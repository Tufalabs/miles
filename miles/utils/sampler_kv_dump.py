"""Dump the keys and values SGLang's attention reads, per finished request (LLMT-83 fetched-KV probe).

Called from SGLang through its source patcher (DUMPER_SOURCE_PATCHER_CONFIG; sglang_kv_dump.yaml and
sglang_fake_int_kv_dump.yaml in rave's LLMT-83 artifacts folder). Without MILES_SAMPLER_KV_DUMP the functions
return at once.

MILES_SAMPLER_KV_DUMP=<dir>        where each finished request's K and V go, one file per request named by the
    SHA-1 of its token ids (prompt then output, int32): {"tokens", "k", "v", "rid"}, k and v [layers, n,
    kv_heads * head_dim] bf16 for the first n = len(tokens) - 1 tokens (the last sampled token never enters
    the cache).
MILES_SAMPLER_KV_DUMP_MAX=<n>      requests dumped per engine (default 64); capture stops after.
MILES_SAMPLER_KV_DUMP_ROWS=<n>     tokens the host buffer holds per engine (default 60000); capture stops when full.
MILES_SAMPLER_KV_DUMP_LAYERS=<n>   the model's layers (default 48, Qwen3-30B-A3B).

What is captured: in Qwen3MoeAttention.forward_core, the k and v handed to the attention layer, after q/k norm,
RoPE and (with MILES_SAMPLER_FAKE_INT_KV) the integer rounding, for every token of every forward, prefill and
decode. With a bf16 cache that is what the cache stores. With an NVFP4 cache the cache stores these rounded to
NVFP4, which the trainer side applies (sampler_kv_substitute.py, fetch_nvfp4). Each forward's rows go to a
pinned host buffer, and a map from KV-pool slot to row lets a request's tokens be gathered at release, prompt
prefixes computed by another request included. Needs eager decode (--sglang-disable-cuda-graph): a captured
CUDA graph replays without this Python.
"""

import hashlib
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

DIR = os.environ.get("MILES_SAMPLER_KV_DUMP", "")
MAX = int(os.environ.get("MILES_SAMPLER_KV_DUMP_MAX", "") or 64)
ROWS = int(os.environ.get("MILES_SAMPLER_KV_DUMP_ROWS", "") or 60000)
LAYERS = int(os.environ.get("MILES_SAMPLER_KV_DUMP_LAYERS", "") or 48)

_state: dict = {}
_counts = {"dumped": 0, "skipped": 0, "forwards": 0}


def _stop(reason: str) -> None:
    if not _state.get("stopped"):
        _state["stopped"] = True
        logger.warning("sampler kv dump: capture stopped (%s), pid %d, counts %s", reason, os.getpid(), _counts)


def _init(attention, forward_batch) -> None:
    width = attention.num_kv_heads * attention.head_dim
    _state.update(
        layers=LAYERS,
        width=width,
        k=torch.empty((LAYERS, ROWS, width), dtype=torch.bfloat16, pin_memory=True),
        v=torch.empty((LAYERS, ROWS, width), dtype=torch.bfloat16, pin_memory=True),
        map=torch.full((1 << 22,), -1, dtype=torch.int64, device=forward_batch.out_cache_loc.device),
        next=0,
        row0=0,
        n=0,
    )
    os.makedirs(DIR, exist_ok=True)
    logger.warning(
        "sampler kv dump: on, %d layers, width %d, %d rows, at most %d requests, into %s, pid %d",
        LAYERS,
        width,
        ROWS,
        MAX,
        DIR,
        os.getpid(),
    )


def _fit_map(loc: torch.Tensor) -> None:
    top = int(loc.max()) + 1
    if top > _state["map"].shape[0]:
        grow = torch.full((top - _state["map"].shape[0] + (1 << 20),), -1, dtype=torch.int64, device=loc.device)
        _state["map"] = torch.cat([_state["map"], grow])


def capture(attention, k: torch.Tensor, v: torch.Tensor, forward_batch) -> None:
    """Store one layer's k and v [tokens, kv_heads * head_dim] of this forward."""
    if not DIR or _state.get("stopped") or k is None or k.shape[0] == 0:
        return
    if not _state:
        _init(attention, forward_batch)
    layer = attention.attn.layer_id
    if layer >= LAYERS:
        _stop(f"layer {layer} past MILES_SAMPLER_KV_DUMP_LAYERS={LAYERS}")
        return
    loc = forward_batch.out_cache_loc
    n = k.shape[0]
    if loc is None or loc.shape[0] != n:
        _stop(f"out_cache_loc {None if loc is None else tuple(loc.shape)} against {n} tokens")
        return
    if layer == 0:
        if _counts["dumped"] >= MAX:
            _stop("enough requests")
            return
        if _state["next"] + n > ROWS:
            _stop("host buffer full")
            return
        row0 = _state["next"]
        _fit_map(loc)
        _state["map"][loc.long()] =torch.arange(row0, row0 + n, device=loc.device)
        _state.update(row0=row0, n=n, next=row0 + n)
        _counts["forwards"] += 1
    elif n != _state["n"]:
        _stop(f"layer {layer} has {n} tokens, layer 0 had {_state['n']}")
        return
    row0 = _state["row0"]
    _state["k"][layer, row0 : row0 + n].copy_(k.reshape(n, -1), non_blocking=True)
    _state["v"][layer, row0 : row0 + n].copy_(v.reshape(n, -1), non_blocking=True)


def on_release(req, tree_cache) -> None:
    """Write a finished request's K and V before its KV row is given back."""
    if not DIR or not _state or _counts["dumped"] >= MAX:
        return
    kv = req.kv
    if not kv.holds_kv or kv.req_pool_idx is None or not req.finished():
        return
    if "abort" in type(req.finished_reason).__name__.lower():
        return
    output = list(getattr(req, "output_ids_through_stop", None) or req.output_ids)
    tokens = list(req.origin_input_ids) + output
    n = min(len(tokens) - 1, int(kv.kv_committed_len))
    if n <= 0:
        return
    slots = tree_cache.req_to_token_pool.req_to_token[kv.req_pool_idx, :n].long()
    rows = _state["map"][slots]
    if bool((rows < 0).any()):
        _counts["skipped"] += 1
        return
    torch.cuda.synchronize()
    rows = rows.cpu()
    record = {
        "tokens": tokens,
        "n": n,
        "rid": str(req.rid),
        "k": _state["k"][:, rows].clone(),
        "v": _state["v"][:, rows].clone(),
    }
    name = hashlib.sha1(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()
    torch.save(record, os.path.join(DIR, f"{name}.pt"))
    _counts["dumped"] += 1
    if _counts["dumped"] in (1, 8, MAX) or _counts["dumped"] % 32 == 0:
        logger.warning("sampler kv dump: %s, pid %d (last: %d tokens, %d with KV)", _counts, os.getpid(), len(tokens), n)
