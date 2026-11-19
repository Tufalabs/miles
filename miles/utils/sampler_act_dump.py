"""Dump SGLang's per-layer activations of generated tokens, per finished request (LLMT-83 mismatch probe).

Called from SGLang's Qwen3MoeDecoderLayer.forward through its source patcher (sglang_act_dump.yaml; with the
integer sampler, sglang_fake_int_act_dump.yaml). Without MILES_SAMPLER_ACT_DUMP the functions return at once.

Decode forwards only: each token there is a generated token at its position in the sequence; prompt tokens (and
anything else computed in prefill) are left out. Per token and layer, three tensors of the hidden size:
  emb    layer 0's input, the token embedding (once per token)
  attn   the attention block's output (after o_proj, before the residual add)
  moe    the MoE block's output (before the residual add)
The residual stream after layer l is emb + sum over layers <= l of (attn + moe), so it is rebuilt from these on
both sides in the same way.

MILES_SAMPLER_ACT_DUMP=<dir>       one file per request, named by the SHA-1 of its token ids (int32): {"tokens",
    "positions", "emb" [n, H], "attn" [L, n, H], "moe" [L, n, H]}, bf16, for the n generated tokens at positions
    len(prompt) .. len(tokens) - 2 (the last sampled token never enters a forward).
MILES_SAMPLER_ACT_DUMP_MAX=<n>     requests dumped per engine (default 8).
MILES_SAMPLER_ACT_DUMP_SELECT=<k>  only requests whose rid's SHA-1 is 0 mod k (default 1, every one), so that the
    dumped answers are not just the ones that finish first.
MILES_SAMPLER_ACT_DUMP_ROWS=<n>    decode tokens the host buffer holds per engine (default 12000).
MILES_SAMPLER_ACT_DUMP_LAYERS=<n>  layers (default 48).
MILES_SAMPLER_ACT_DUMP_STRIDE=<s>  only tokens at positions that are multiples of s (default 1, all).
Needs eager decode (--sglang-disable-cuda-graph): a captured CUDA graph replays without this Python.
"""

import hashlib
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

DIR = os.environ.get("MILES_SAMPLER_ACT_DUMP", "")
MAX = int(os.environ.get("MILES_SAMPLER_ACT_DUMP_MAX", "") or 8)
ROWS = int(os.environ.get("MILES_SAMPLER_ACT_DUMP_ROWS", "") or 12000)
LAYERS = int(os.environ.get("MILES_SAMPLER_ACT_DUMP_LAYERS", "") or 48)
SELECT = int(os.environ.get("MILES_SAMPLER_ACT_DUMP_SELECT", "") or 1)
STRIDE = int(os.environ.get("MILES_SAMPLER_ACT_DUMP_STRIDE", "") or 1)

_state: dict = {}
_by_idx: dict[int, list[int]] = {}
_counts = {"dumped": 0, "incomplete": 0, "forwards": 0}


def _stop(reason: str) -> None:
    if not _state.get("stopped"):
        _state["stopped"] = True
        logger.warning("sampler act dump: capture stopped (%s), pid %d, counts %s", reason, os.getpid(), _counts)


def _decode(forward_batch) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    return mode is not None and mode.is_decode()


def layer_input(layer, hidden: torch.Tensor, forward_batch) -> None:
    """At layer 0 of a decode forward: give the batch's tokens their rows and store their embeddings."""
    if not DIR or layer.layer_id != 0:
        return
    _state["row0"] = None
    if _state.get("stopped") or not _decode(forward_batch):
        return
    if not torch.is_tensor(hidden) or hidden.shape[0] == 0:
        return
    if "emb" not in _state:
        size = hidden.shape[-1]
        _state.update(
            emb=torch.empty((ROWS, size), dtype=torch.bfloat16, pin_memory=True),
            attn=torch.empty((LAYERS, ROWS, size), dtype=torch.bfloat16, pin_memory=True),
            moe=torch.empty((LAYERS, ROWS, size), dtype=torch.bfloat16, pin_memory=True),
            idx=np.full(ROWS, -1, dtype=np.int64),
            pos=np.full(ROWS, -1, dtype=np.int64),
            next=0,
        )
        os.makedirs(DIR, exist_ok=True)
        logger.warning("sampler act dump: on, %d layers, hidden %d, %d rows, %d requests, into %s, pid %d", LAYERS, size, ROWS, MAX, DIR, os.getpid())
    if _counts["dumped"] >= MAX:
        _stop("enough requests")
        return
    n = hidden.shape[0]
    indices, positions = forward_batch.req_pool_indices, forward_batch.positions
    if indices is None or positions is None or indices.shape[0] != n or positions.shape[0] != n:
        _stop(f"decode batch of {n} tokens without one request index and position each")
        return
    indices, positions = indices.cpu().numpy(), positions.cpu().numpy()
    keep = np.nonzero(positions % STRIDE == 0)[0]
    kept = len(keep)
    if kept == 0:
        return
    if _state["next"] + kept > ROWS:
        _stop("host buffer full")
        return
    row0 = _state["next"]
    _state["idx"][row0 : row0 + kept] = indices[keep]
    _state["pos"][row0 : row0 + kept] = positions[keep]
    for offset, idx in enumerate(indices[keep].tolist()):
        _by_idx.setdefault(int(idx), []).append(row0 + offset)
    keep_t = torch.as_tensor(keep, device=hidden.device)
    _state["emb"][row0 : row0 + kept].copy_(hidden.reshape(n, -1)[keep_t], non_blocking=True)
    _state.update(row0=row0, n=n, kept=kept, keep=keep_t, next=row0 + kept)
    _counts["forwards"] += 1


def capture(layer, kind: str, hidden: torch.Tensor, forward_batch) -> None:
    """A decode forward's attention ("attn") or MoE ("moe") output of one layer."""
    row0 = _state.get("row0")
    if row0 is None or not _decode(forward_batch):
        return
    n = _state["n"]
    if not torch.is_tensor(hidden) or hidden.shape[0] != n or layer.layer_id >= LAYERS:
        _stop(f"layer {layer.layer_id} {kind}: {type(hidden).__name__} {getattr(hidden, 'shape', None)} for {n} tokens")
        return
    _state[kind][layer.layer_id, row0 : row0 + _state["kept"]].copy_(hidden.reshape(n, -1)[_state["keep"]], non_blocking=True)


def on_release(req, tree_cache) -> None:
    """Write a finished request's generated tokens' activations; forget its rows either way."""
    if not DIR or "emb" not in _state:
        return
    kv = req.kv
    idx = kv.req_pool_idx
    rows = _by_idx.pop(int(idx), []) if idx is not None else []
    if not rows or not req.finished() or _counts["dumped"] >= MAX:
        return
    if "abort" in type(req.finished_reason).__name__.lower():
        return
    if int(hashlib.sha1(str(req.rid).encode()).hexdigest(), 16) % SELECT:
        return
    output = list(getattr(req, "output_ids_through_stop", None) or req.output_ids)
    tokens = list(req.origin_input_ids) + output
    wanted = [p for p in range(len(req.origin_input_ids), len(tokens) - 1) if p % STRIDE == 0]
    if not wanted:
        return
    latest = {}
    for row in rows:
        latest[int(_state["pos"][row])] = row
    if any(position not in latest for position in wanted):
        _counts["incomplete"] += 1
        return
    torch.cuda.synchronize()
    take = torch.tensor([latest[position] for position in wanted], dtype=torch.long)
    record = {
        "tokens": tokens,
        "positions": wanted,
        "rid": str(req.rid),
        "emb": _state["emb"][take].clone(),
        "attn": _state["attn"][:, take].clone(),
        "moe": _state["moe"][:, take].clone(),
    }
    name = hashlib.sha1(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()
    torch.save(record, os.path.join(DIR, f"{name}.pt"))
    _counts["dumped"] += 1
    logger.warning("sampler act dump: %s, pid %d (last: %d generated tokens)", _counts, os.getpid(), len(wanted))
