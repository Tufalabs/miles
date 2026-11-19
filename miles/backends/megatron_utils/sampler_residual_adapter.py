"""A residual correction of the attention and MoE outputs fitted on SGLang's activations (LLMT-83 mismatch probe).

Per layer and block (attention, MoE): a linear map W from the block's input in the trainer (with a bias column) to
the sampler-minus-trainer difference of the block's output, h <- h + W [x; 1], fitted by ridge regression on the
generated tokens SGLang dumped (sampler_act_dump.py). Offline, at a frozen policy. Fit rollouts only accumulate
X^T X, X^T Y and Y^T Y per block (float64); the first rollout after a sweep's fit rollouts all-reduces them over the
ranks and solves. Corrections are computed and added in float32.

Sweeps: maps fitted on uncorrected inputs overcorrect when all are applied at once. Sweep k > 1 refits the remainder
with the corrections of the earlier sweeps in place: its fit forwards apply them, the inputs x are the corrected
ones, the target is the sampler's output minus the corrected output; the sweep's map adds to the earlier ones.

MILES_RESID_DIR=<dir>               SGLang's dumps, named by the SHA-1 of a sequence's token ids
MILES_RESID_FIT=<rollout>,...       sweep 1's fit rollouts; MILES_RESID_FIT2, MILES_RESID_FIT3 the next sweeps'
MILES_RESID_HOLDOUT=<rollout>,...   uncorrected forwards whose dumped tokens score sweep 1's maps: held-out R^2 per
                                    block at each ridge of MILES_RESID_RIDGES (default "1e-4,1e-2,1"), logged at the
                                    first rollout after them
MILES_RESID_EVAL=<rollout>:<variant>[:<ridge>],...  variants: off, full, r<k> (the map cut to rank k), attn, moe.
                                    Without a ridge: the sum of the sweeps solved so far at MILES_RESID_RIDGE; with
                                    one: sweep 1 alone at that ridge
MILES_RESID_RIDGE=<float>           the ridge of the sweeps, relative to the mean diagonal of X^T X (default 1e-4)
Logged per block and sweep after the solve: the fit R^2, ||W||, the tokens. Install before sampler_act_compare so
that the comparison sees the corrected outputs.
"""

import hashlib
import logging
import os
import re

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.metrics import debug_dump

logger = logging.getLogger(__name__)


def _ids(name: str) -> list[int]:
    return [int(r) for r in os.environ.get(name, "").split(",") if r.strip()]


DIR = os.environ.get("MILES_RESID_DIR", "")
SWEEPS = [ids for ids in (_ids("MILES_RESID_FIT"), _ids("MILES_RESID_FIT2"), _ids("MILES_RESID_FIT3")) if ids]
HOLDOUT = set(_ids("MILES_RESID_HOLDOUT"))
RIDGE = float(os.environ.get("MILES_RESID_RIDGE", "") or 1e-4)
RIDGES = [float(r) for r in (os.environ.get("MILES_RESID_RIDGES", "") or "1e-4,1e-2,1").split(",")]
EVAL: dict[int, tuple[str, float | None]] = {}
for _entry in filter(None, (p.strip() for p in os.environ.get("MILES_RESID_EVAL", "").split(","))):
    _rollout, _variant, *_ridge = _entry.split(":")
    assert _variant in ("off", "full", "attn", "moe") or re.fullmatch(r"r\d+", _variant), _entry
    EVAL[int(_rollout)] = (_variant, float(_ridge[0]) if _ridge else None)
FIT_SWEEP = {rollout: k for k, ids in enumerate(SWEEPS) for rollout in ids}

_files: dict[str, str] = {}
_stats: list[dict] = [dict() for _ in SWEEPS]  # per sweep: (layer, block) -> [xtx, xty, yty, n]
_holdout: dict = {}
_solved: list[dict] = []  # per solved sweep: (layer, block) -> W at RIDGE
_ridged: dict[tuple, torch.Tensor] = {}  # (layer, block, ridge) -> sweep 1's W at that ridge
_ranked: dict[tuple, tuple] = {}
_batch: dict = {}
_reduced = {"holdout": False}


def enabled() -> bool:
    return bool(DIR and (SWEEPS or EVAL))


def _reduce(stats: dict) -> None:
    if not dist.is_initialized():
        return
    for key in sorted(stats):
        entry = stats[key]
        for tensor in entry[:3]:
            dist.all_reduce(tensor)
        count = torch.tensor([entry[3]], dtype=torch.float64, device=entry[0].device)
        dist.all_reduce(count)
        entry[3] = int(count.item())


def _ridge_solve(entry, ridge: float) -> torch.Tensor:
    xtx, xty = entry[0], entry[1]
    scale = ridge * float(xtx.diagonal().mean())
    return torch.linalg.solve(xtx + scale * torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device), xty)


def _r2(entry, w) -> float:
    xtx, xty, yty = entry[0], entry[1], entry[2]
    sse = yty - 2 * (w * xty).sum() + (w * (xtx @ w)).sum()
    return float(1 - sse / yty.clamp_min(1e-30))


def _advance(rollout: int) -> None:
    """Solve the sweeps whose fit rollouts are over, and score the held-out tokens, before this rollout runs."""
    sweep = FIT_SWEEP.get(rollout)
    while len(_solved) < len(SWEEPS) and (sweep is None or sweep > len(_solved)) and _stats[len(_solved)]:
        k = len(_solved)
        _reduce(_stats[k])
        maps = {}
        for key in sorted(_stats[k]):
            entry = _stats[k][key]
            maps[key] = _ridge_solve(entry, RIDGE)
            logger.info("resid adapter sweep=%d layer=%d block=%s tokens=%d r2=%.4f w_norm=%.4e target_rms=%.4e", k + 1, key[0], key[1], entry[3], _r2(entry, maps[key]), float(maps[key].norm()), float((entry[2] / max(1, entry[3] * maps[key].shape[1])).sqrt()))
        _solved.append(maps)
    if _holdout and not _reduced["holdout"] and rollout not in HOLDOUT and _solved:
        _reduce(_holdout)
        _reduced["holdout"] = True
        for key in sorted(_holdout):
            values = []
            for ridge in RIDGES:
                w = _ridged.setdefault((*key, ridge), _ridge_solve(_stats[0][key], ridge))
                values.append(f"{ridge:g}:{_r2(_holdout[key], w):.4f}")
            logger.info("resid adapter holdout layer=%d block=%s tokens=%d r2 %s", key[0], key[1], _holdout[key][3], " ".join(values))


def _load(unconcat_tokens):
    if not _files:
        _files.update({name[:-3]: os.path.join(DIR, name) for name in os.listdir(DIR) if name.endswith(".pt")})
    lengths = [int(t.numel()) for t in unconcat_tokens]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    matched = []
    for start, tokens in zip(starts, unconcat_tokens, strict=True):
        name = hashlib.sha1(tokens.detach().cpu().numpy().astype(np.int32).tobytes()).hexdigest()
        if name in _files:
            matched.append((start, torch.load(_files[name], map_location="cpu", weights_only=False)))
    if not matched:
        return None
    device = torch.cuda.current_device()
    return {
        "index": torch.cat([torch.tensor(r["positions"], dtype=torch.long) + s for s, r in matched]).to(device),
        "attn": torch.cat([r["attn"] for _, r in matched], dim=1),
        "moe": torch.cat([r["moe"] for _, r in matched], dim=1),
        "done": set(),
    }


def set_batch(unconcat_tokens) -> None:
    _batch.clear()
    rollout = debug_dump._training_rollout_id
    if not enabled() or rollout is None:
        return
    _advance(rollout)
    if rollout in EVAL:
        variant, ridge = EVAL[rollout]
        _batch.update(mode="eval", variant=variant, ridge=ridge)
        return
    if rollout in FIT_SWEEP or rollout in HOLDOUT:
        loaded = _load(unconcat_tokens)
        if loaded is not None:
            _batch.update(loaded)
            _batch.update(mode="fit" if rollout in FIT_SWEEP else "holdout", sweep=FIT_SWEEP.get(rollout))


def _with_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones_like(x[:, :1])], dim=1)


def _apply(key, x: torch.Tensor, variant: str, ridge: float | None) -> torch.Tensor | None:
    """The correction for block `key` on inputs x [tokens, hidden], float32, or None."""
    if ridge is not None:
        if key not in _stats[0]:
            return None
        w = _ridged.setdefault((*key, ridge), _ridge_solve(_stats[0][key], ridge))
    else:
        maps = [m[key] for m in _solved if key in m]
        if not maps:
            return None
        w = maps[0] if len(maps) == 1 else torch.stack(maps).sum(0)
    xb = _with_bias(x.float())
    if variant.startswith("r"):
        rank = int(variant[1:])
        cut = (*key, rank, ridge, len(_solved))
        if cut not in _ranked:
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            _ranked[cut] = ((u[:, :rank] * s[:rank]).float(), vh[:rank].float())
        left, right = _ranked[cut]
        return (xb @ left) @ right
    return xb @ w.float()


def _hook(layer: int, block: str):
    key = (layer, block)

    def hook(module, args, output):
        if not _batch or not args or not torch.is_tensor(args[0]):
            return None
        out = output[0] if isinstance(output, tuple) else output
        x = args[0].reshape(-1, args[0].shape[-1])
        flat = out.reshape(-1, out.shape[-1])
        if _batch["mode"] == "eval":
            variant, ridge = _batch["variant"], _batch["ridge"]
            if variant == "off" or (variant in ("attn", "moe") and variant != block):
                return None
            correction = _apply(key, x, "full" if variant in ("attn", "moe") else variant, ridge)
            if correction is None:
                return None
            corrected = (flat.float() + correction).to(out.dtype).reshape(out.shape)
            return (corrected, *output[1:]) if isinstance(output, tuple) else corrected
        if key in _batch["done"]:
            return None
        _batch["done"].add(key)
        corrected = flat.float()
        if _batch["mode"] == "fit" and _batch["sweep"] > 0 and _solved:
            # earlier sweeps' corrections are in place in this forward
            correction = _apply(key, x, "full", None)
            if correction is not None:
                corrected = corrected + correction
        index = _batch["index"]
        xs = _with_bias(x[index].double())
        target = _batch[block][layer].to(out.device).double() - corrected[index].double()
        stats = _stats[_batch["sweep"]] if _batch["mode"] == "fit" else _holdout
        if key not in stats:
            d, h = xs.shape[1], target.shape[1]
            stats[key] = [torch.zeros(d, d, dtype=torch.float64, device=out.device), torch.zeros(d, h, dtype=torch.float64, device=out.device), torch.zeros((), dtype=torch.float64, device=out.device), 0]
        entry = stats[key]
        entry[0] += xs.T @ xs
        entry[1] += xs.T @ target
        entry[2] += (target * target).sum()
        entry[3] += xs.shape[0]
        if corrected is not flat and _batch["mode"] == "fit":
            fixed = corrected.to(out.dtype).reshape(out.shape)
            return (fixed, *output[1:]) if isinstance(output, tuple) else fixed
        return None

    return hook


def maybe_install(model_chunks) -> None:
    if not enabled():
        return
    hooks = 0
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.(self_attention|mlp)$", name)
            if match:
                module.register_forward_hook(_hook(int(match.group(1)), "attn" if match.group(2) == "self_attention" else "moe"))
                hooks += 1
    logger.info("resid adapter installed: %d hooks, sweeps %s, holdout %s, eval %s, ridge %g, ridges %s, dumps in %s", hooks, SWEEPS, sorted(HOLDOUT), EVAL, RIDGE, RIDGES, DIR)
