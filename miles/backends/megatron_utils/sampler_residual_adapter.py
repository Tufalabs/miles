"""A residual correction of the attention and MoE outputs fitted on SGLang's activations (LLMT-83 mismatch probe).

Per layer and block (attention, MoE): a linear map W from the block's input in the trainer (with a bias column) to
the sampler-minus-trainer difference of the block's output, h <- h + W [x; 1], fitted by ridge regression on the
generated tokens that SGLang dumped (sampler_act_dump.py). Offline, at a frozen policy: the fit rollouts' forwards
run uncorrected and only accumulate X^T X, X^T Y and Y^T Y per block (float64); at the first eval rollout every rank
all-reduces them and solves; eval rollouts' forwards add the correction to every token's block output.

MILES_RESID_DIR=<dir>               SGLang's dumps, named by the SHA-1 of a sequence's token ids
MILES_RESID_FIT=<rollout>,...       the fit rollouts
MILES_RESID_EVAL=<rollout>:<variant>,...  variants: off, full, r<k> (W cut to rank k by its SVD), attn or moe (full
                                    rank, that block only)
MILES_RESID_RIDGE=<float>           ridge, relative to the mean diagonal of X^T X (default 1e-4)
Logged once per block after the solve: the fit's R^2 on the fit tokens, ||W||, the tokens. Install before
sampler_act_compare so that the comparison sees the corrected outputs.
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

DIR = os.environ.get("MILES_RESID_DIR", "")
FIT = {int(r) for r in os.environ.get("MILES_RESID_FIT", "").split(",") if r.strip()}
EVAL = {}
for _entry in filter(None, (p.strip() for p in os.environ.get("MILES_RESID_EVAL", "").split(","))):
    _rollout, _variant = _entry.split(":")
    assert _variant in ("off", "full", "attn", "moe") or re.fullmatch(r"r\d+", _variant), _entry
    EVAL[int(_rollout)] = _variant
RIDGE = float(os.environ.get("MILES_RESID_RIDGE", "") or 1e-4)

_files: dict[str, str] = {}
_stats: dict[tuple[int, str], list] = {}
_maps: dict[tuple[int, str], torch.Tensor] = {}
_ranked: dict[tuple[int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}
_batch: dict = {}
_solved = [False]


def enabled() -> bool:
    return bool(DIR and (FIT or EVAL))


def _solve() -> None:
    keys = sorted(_stats)
    for key in keys:
        xtx, xty, yty, n = _stats[key]
        if dist.is_initialized():
            for tensor in (xtx, xty, yty):
                dist.all_reduce(tensor)
            count = torch.tensor([n], dtype=torch.float64, device=xtx.device)
            dist.all_reduce(count)
            n = int(count.item())
        ridge = RIDGE * float(xtx.diagonal().mean())
        w = torch.linalg.solve(xtx + ridge * torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device), xty)
        sse = yty - 2 * (w * xty).sum() + (w * (xtx @ w)).sum()
        _maps[key] = w
        logger.info("resid adapter layer=%d block=%s tokens=%d r2=%.4f w_norm=%.4e target_rms=%.4e", key[0], key[1], n, float(1 - sse / yty.clamp_min(1e-30)), float(w.norm()), float((yty / max(1, n * w.shape[1])).sqrt()))
    _solved[0] = True


def set_batch(unconcat_tokens) -> None:
    _batch.clear()
    rollout = debug_dump._training_rollout_id
    if not enabled() or rollout is None:
        return
    if rollout in EVAL:
        if not _solved[0]:
            _solve()
        lengths = [int(t.numel()) for t in unconcat_tokens]
        _batch.update(rollout=rollout, variant=EVAL[rollout], lengths=lengths)
        return
    if rollout not in FIT:
        return
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
        return
    device = torch.cuda.current_device()
    _batch.update(
        rollout=rollout,
        variant="fit",
        index=torch.cat([torch.tensor(r["positions"], dtype=torch.long) + s for s, r in matched]).to(device),
        attn=torch.cat([r["attn"] for _, r in matched], dim=1),
        moe=torch.cat([r["moe"] for _, r in matched], dim=1),
        done=set(),
    )


def _with_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones_like(x[:, :1])], dim=1)


def _correction(key, x: torch.Tensor, variant: str) -> torch.Tensor:
    w = _maps[key]
    if variant.startswith("r"):
        rank = int(variant[1:])
        cut = (key[0], key[1], rank)
        if cut not in _ranked:
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            _ranked[cut] = ((u[:, :rank] * s[:rank]).to(torch.bfloat16), vh[:rank].to(torch.bfloat16))
        left, right = _ranked[cut]
        return (_with_bias(x.to(torch.bfloat16)) @ left) @ right
    return _with_bias(x.to(torch.bfloat16)) @ w.to(torch.bfloat16)


def _hook(layer: int, block: str):
    key = (layer, block)

    def hook(module, args, output):
        if not _batch or not args or not torch.is_tensor(args[0]):
            return None
        out = output[0] if isinstance(output, tuple) else output
        x = args[0].reshape(-1, args[0].shape[-1])
        if _batch["variant"] == "fit":
            if key in _batch["done"]:
                return None
            _batch["done"].add(key)
            index = _batch["index"]
            xs = _with_bias(x[index].double())
            target = _batch[block][layer].to(out.device).double() - out.reshape(-1, out.shape[-1])[index].double()
            if key not in _stats:
                d, h = xs.shape[1], target.shape[1]
                _stats[key] = [torch.zeros(d, d, dtype=torch.float64, device=out.device), torch.zeros(d, h, dtype=torch.float64, device=out.device), torch.zeros((), dtype=torch.float64, device=out.device), 0]
            stats = _stats[key]
            stats[0] += xs.T @ xs
            stats[1] += xs.T @ target
            stats[2] += (target * target).sum()
            stats[3] += xs.shape[0]
            return None
        variant = _batch["variant"]
        if variant == "off" or (variant in ("attn", "moe") and variant != block) or key not in _maps:
            return None
        corrected = out + _correction(key, x, "full" if variant in ("attn", "moe") else variant).reshape(out.shape).to(out.dtype)
        return (corrected, *output[1:]) if isinstance(output, tuple) else corrected

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
    logger.info("resid adapter installed: %d hooks, fit %s, eval %s, ridge %g, dumps in %s", hooks, sorted(FIT), EVAL, RIDGE, DIR)
