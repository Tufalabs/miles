"""Put the sampler's own keys and values, or their rounding alone, into the trainer's attention (LLMT-83 probe).

MILES_KV_SUBSTITUTE=<rollout>:<mode>[:<dump>],...  what each rollout's training forward does to K and V (after
    q/k norm and RoPE) in every layer's core attention; rollouts not listed are left alone. Modes:
      none          nothing (the plain trainer, for the same replayed batch)
      emul_nvfp4    the trainer's own K and V through NVFP4 as SGLang's cache stores them (sampler_emulation)
      emul_int<b>g<n>  the trainer's own K and V rounded to b-bit integers, one scale per n values
                    (sampler_fake_int.round_last_dim, what the integer sampler does before its cache)
      fetch         the sampler's K and V from the dump named <dump>, as dumped (an integer sampler's, rounded)
      fetch_nvfp4   the same, then through NVFP4 as an NVFP4 cache stores them
    Every replacement is straight-through: the forward uses the new values, the gradient flows as for the
    trainer's own K and V.
MILES_KV_SUBSTITUTE_DIRS=<dump>=<dir>,...  the dump directories (sampler_kv_dump.py's files, named by the SHA-1
    of a sequence's token ids). A sequence with no file keeps its own K and V; the first n = len - 1 positions
    of a sequence with one are replaced, the last token (never in the sampler's cache) keeps its own.
Each fetch step logs, for its first micro-batch with a dump and per layer, how far the sampler's K and V are from the
trainer's rounded the same way (relative RMS, overall and by position), and how many sequences were found.
Needs packed sequences (qkv_format thd), tensor- and context-parallel size 1.
"""

import hashlib
import logging
import os
import re

import numpy as np
import torch

from miles.backends.megatron_utils.sampler_emulation import _nvfp4_round_trip
from miles.backends.training_utils.metrics import debug_dump

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_KV_SUBSTITUTE"
DIRS_ENV_VAR = "MILES_KV_SUBSTITUTE_DIRS"
_POSITION_BUCKETS = ((0, 64), (64, 256), (256, 10**9))

_schedule: dict[int, tuple[str, str]] = {}
_dirs: dict[str, str] = {}
_index: dict[str, dict[str, str]] = {}
_batch: dict = {}
_logged: set[tuple[int, int]] = set()
_totals: dict[int, tuple[int, int]] = {}


def enabled() -> bool:
    return bool(_schedule)


def _int_round(x: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    top = 2 ** (bits - 1) - 1
    values = x.float().reshape(-1, group)
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / top
    return (torch.round(values / scale).clamp_(-top, top) * scale).to(x.dtype).reshape(x.shape)


def _rounding(mode: str):
    """The rounding a mode applies to the trainer's K and V (emul_*) or that the cache applies (fetch_nvfp4)."""
    if mode.endswith("nvfp4"):
        return _nvfp4_round_trip
    if mode.startswith("emul_int"):
        bits, group = (int(part) for part in mode.removeprefix("emul_int").split("g"))
        return lambda x: _int_round(x, bits, group)
    return lambda x: x


def _parse() -> None:
    spec = os.environ.get(ENV_VAR, "").strip()
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        rollout, mode, *dump = entry.split(":")
        assert mode in ("none", "emul_nvfp4", "fetch", "fetch_nvfp4") or mode.startswith("emul_int"), entry
        assert mode.startswith("fetch") == bool(dump), f"{ENV_VAR}: {entry}: fetch modes name a dump, others none"
        _schedule[int(rollout)] = (mode, dump[0] if dump else "")
    for entry in filter(None, (part.strip() for part in os.environ.get(DIRS_ENV_VAR, "").split(","))):
        name, path = entry.split("=", 1)
        _dirs[name] = path
    missing = {dump for _, dump in _schedule.values() if dump} - set(_dirs)
    assert not missing, f"{DIRS_ENV_VAR} lacks {sorted(missing)}"


def _files(dump: str) -> dict[str, str]:
    if dump not in _index:
        folder = _dirs[dump]
        _index[dump] = {name[:-3]: os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".pt")}
        logger.info("kv substitute: dump %s has %d sequences in %s", dump, len(_index[dump]), folder)
    return _index[dump]


def _current() -> tuple[int | None, str, str]:
    rollout = debug_dump._training_rollout_id
    mode, dump = _schedule.get(rollout, ("none", "")) if rollout is not None else ("none", "")
    return rollout, mode, dump


def set_batch(unconcat_tokens) -> None:
    """Called with each micro-batch's sequences before the model's forward: loads the dumps a fetch step needs."""
    _batch.clear()
    if not _schedule:
        return
    rollout, mode, dump = _current()
    _batch.update(rollout=rollout, mode=mode, lengths=[int(t.numel()) for t in unconcat_tokens])
    if not mode.startswith("fetch"):
        return
    files = _files(dump)
    found = []
    for tokens in unconcat_tokens:
        name = hashlib.sha1(tokens.detach().cpu().numpy().astype(np.int32).tobytes()).hexdigest()
        path = files.get(name)
        if path is None:
            found.append(None)
            continue
        record = torch.load(path, map_location="cpu", weights_only=False)
        assert record["tokens"] == tokens.tolist(), f"{path}: tokens differ from the sequence with the same hash"
        found.append(record)
    _batch["records"] = found
    hits = sum(record is not None for record in found)
    _totals[rollout] = (_totals.get(rollout, (0, 0))[0] + hits, _totals.get(rollout, (0, 0))[1] + len(found))
    logger.info("kv substitute rollout=%d sequences found so far %d/%d", rollout, *_totals[rollout])
    _batch["first"] = hits > 0 and (rollout, 0) not in _logged
    if _batch["first"]:
        _logged.add((rollout, 0))


def _replace(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return x + (target - x).detach()


def _fetched(layer: int, which: str, like: torch.Tensor, bounds) -> tuple[torch.Tensor, torch.Tensor]:
    """The sampler's values of one layer laid out like the packed `like` [tokens, groups, head_dim], and a mask."""
    out = like.detach().clone()
    mask = torch.zeros(like.shape[0], dtype=torch.bool, device=like.device)
    for i, record in enumerate(_batch["records"]):
        if record is None:
            continue
        start, n = int(bounds[i]), record["n"]
        values = record[which][layer].to(device=like.device, dtype=like.dtype)
        out[start : start + n] = values.reshape(n, *like.shape[1:])
        mask[start : start + n] = True
    return out, mask


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()).pow(2).mean() / b.float().pow(2).mean().clamp_min(1e-30)).sqrt())


def _log_distance(layer: int, key, value, fk, fv, mask, bounds, rounding) -> None:
    pieces = []
    for name, own, fetched in (("k", key, fk), ("v", value, fv)):
        own_rounded = rounding(own.detach())
        pieces.append(f"{name}_rel={_rel(fetched[mask], own_rounded[mask]):.4e}")
        position = torch.cat([torch.arange(int(e) - int(s), device=own.device) for s, e in zip(bounds[:-1], bounds[1:], strict=True)])
        position = torch.cat([position, torch.zeros(own.shape[0] - position.numel(), dtype=position.dtype, device=own.device)])
        for low, high in _POSITION_BUCKETS:
            sel = mask & (position >= low) & (position < high)
            if bool(sel.any()):
                pieces.append(f"{name}_rel_pos{low}={_rel(fetched[sel], own_rounded[sel]):.4e}")
    found = sum(record is not None for record in _batch["records"])
    logger.info(
        "kv substitute rollout=%d mode=%s layer=%d sequences=%d/%d tokens=%d %s",
        _batch["rollout"],
        _batch["mode"],
        layer,
        found,
        len(_batch["records"]),
        int(mask.sum()),
        " ".join(pieces),
    )


def _wrap(core_attention: torch.nn.Module, layer: int) -> None:
    original_forward = core_attention.forward

    def forward(query, key, value, *args, **kwargs):
        mode = _batch.get("mode", "none")
        if mode == "none" or not _batch:
            return original_forward(query, key, value, *args, **kwargs)
        packed = kwargs.get("packed_seq_params")
        assert packed is not None and key.dim() == 3, f"{ENV_VAR} needs packed sequences (qkv_format thd)"
        bounds = packed.cu_seqlens_host
        rounding = _rounding(mode)
        if mode.startswith("emul"):
            key, value = _replace(key, rounding(key.detach())), _replace(value, rounding(value.detach()))
        else:
            fk, mask = _fetched(layer, "k", key, bounds)
            fv, _ = _fetched(layer, "v", value, bounds)
            fk, fv = rounding(fk), rounding(fv)
            if _batch.get("first"):
                _log_distance(layer, key, value, fk, fv, mask, bounds, rounding)
            key = _replace(key, torch.where(mask[:, None, None], fk, rounding(key.detach())))
            value = _replace(value, torch.where(mask[:, None, None], fv, rounding(value.detach())))
        return original_forward(query, key, value, *args, **kwargs)

    core_attention.forward = forward


def maybe_install(model_chunks) -> None:
    _parse()
    if not _schedule:
        return
    wrapped = 0
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if name.endswith("self_attention") and hasattr(module, "core_attention"):
                _wrap(module.core_attention, int(re.search(r"(?:^|\.)layers\.(\d+)\.", name).group(1)))
                wrapped += 1
    assert wrapped > 0, f"{ENV_VAR} is set but no self_attention.core_attention module was found"
    logger.info("kv substitute installed in %d layers: %s, dumps %s", wrapped, _schedule, _dirs)
