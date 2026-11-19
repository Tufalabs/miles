"""Sensitivity control (LLMT-83 mismatch probe): how far does a small realization noise move the trainer's own output?

MILES_SENS_NOISE=<rollout>:<relative>,...  rollouts whose forward adds Gaussian noise to every attention and MoE
    output, per token of standard deviation <relative> times that token's root mean square over the hidden size
MILES_SENS_PAIRS=<clean>:<noisy>,...       pairs of rollouts on the same replayed batch: the clean one keeps the
    trainer's top-k log-probs (k = MILES_SENS_TOPK, default 128) at every answer position, the noisy one scores its
    own log-probs at those ids, and KL(clean || noisy) over those ids plus one bucket for the rest is logged per pair as running totals per rank:
    "sensitivity clean=<id> noisy=<id> kl_sum=<s> tokens=<n> applied_rel=<r>", where applied_rel is the root mean
    square of the noise that survived the cast to the output's dtype relative to that of the outputs, all blocks
The batches of the two rollouts must be the same data in the same order (the same rollout file), so that
micro-batch i of one is micro-batch i of the other. Any emulation (MILES_MISMATCH_EMULATION, MILES_KV_SUBSTITUTE)
of a rollout applies as usual, so a pair of emulated rollouts measures the noise floor around the emulated forward.
"""

import logging
import os
import re

import torch

from miles.backends.training_utils.metrics import debug_dump

logger = logging.getLogger(__name__)

NOISE = {int(r): float(v) for r, v in (e.split(":") for e in os.environ.get("MILES_SENS_NOISE", "").split(",") if e.strip())}
PAIRS = [tuple(int(v) for v in e.split(":")) for e in os.environ.get("MILES_SENS_PAIRS", "").split(",") if e.strip()]
TOPK = int(os.environ.get("MILES_SENS_TOPK", "") or 128)
CLEAN = {clean: noisy for clean, noisy in PAIRS}
NOISY = {noisy: clean for clean, noisy in PAIRS}

_batch: dict = {}
_kept: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_counter: dict[int, int] = {}
_totals: dict[int, list] = {}
_generator: dict = {}
_applied: dict[int, list] = {}


def enabled() -> bool:
    return bool(NOISE or PAIRS)


def set_batch(total_lengths, response_lengths) -> None:
    _batch.clear()
    rollout = debug_dump._training_rollout_id
    if rollout is None or not enabled():
        return
    index = _counter.get(rollout, 0)
    _counter[rollout] = index + 1
    positions, start = [], 0
    for total, response in zip(total_lengths, response_lengths, strict=True):
        total, response = int(total), int(response)
        positions.append(torch.arange(start + total - response - 1, start + total - 1))
        start += total
    _batch.update(rollout=rollout, micro=index, positions=torch.cat(positions) if positions else torch.zeros(0, dtype=torch.long))


def _noise_hook(module, args, output):
    rollout = _batch.get("rollout")
    if rollout not in NOISE:
        return None
    out = output[0] if isinstance(output, tuple) else output
    if "gen" not in _generator:
        _generator["gen"] = torch.Generator(device=out.device).manual_seed(1234)
    rms = out.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    noisy = (out.float() + NOISE[rollout] * rms * torch.randn(out.shape, generator=_generator["gen"], device=out.device)).to(out.dtype)
    # what survives the cast to the output's dtype (bf16 rounds part of a 1e-3 noise away)
    applied = _applied.setdefault(rollout, [0.0, 0.0])
    applied[0] += float((noisy.float() - out.float()).square().sum())
    applied[1] += float(out.float().square().sum())
    return (noisy, *output[1:]) if isinstance(output, tuple) else noisy


def _logits_hook(module, args, output):
    rollout = _batch.get("rollout")
    if rollout not in CLEAN and rollout not in NOISY:
        return None
    logits = output[0] if isinstance(output, tuple) else output
    flat = logits.reshape(-1, logits.shape[-1])
    positions = _batch["positions"].to(flat.device)
    if rollout in CLEAN:
        ids, logps = [], []
        for chunk in positions.split(1024):
            lp = torch.log_softmax(flat[chunk].float(), dim=-1)
            top = lp.topk(TOPK, dim=-1)
            ids.append(top.indices.cpu())
            logps.append(top.values.cpu())
        _kept[(rollout, _batch["micro"])] = (torch.cat(ids), torch.cat(logps))
        return None
    clean = NOISY[rollout]
    kept = _kept.pop((clean, _batch["micro"]), None)
    if kept is None or kept[0].shape[0] != positions.shape[0]:
        logger.warning("sensitivity: micro-batch %d of rollout %d does not match the clean rollout %d", _batch["micro"], rollout, clean)
        return None
    total = _totals.setdefault(rollout, [0.0, 0])
    for chunk, ids, logps in zip(positions.split(1024), kept[0].split(1024), kept[1].split(1024), strict=True):
        lp = torch.log_softmax(flat[chunk].float(), dim=-1).gather(1, ids.to(flat.device))
        q = logps.to(flat.device)
        # the top-k ids plus one bucket for the rest: a proper KL (>= 0), a lower bound on the full one
        q_rest = (1 - q.exp().sum(-1)).clamp_min(1e-30)
        p_rest = (1 - lp.exp().sum(-1)).clamp_min(1e-30)
        total[0] += float((q.exp() * (q - lp)).sum() + (q_rest * (q_rest.log() - p_rest.log())).sum())
        total[1] += chunk.shape[0]
    applied = _applied.get(rollout, [0.0, 1.0])
    logger.info("sensitivity clean=%d noisy=%d kl_sum=%.6e tokens=%d applied_rel=%.3e", clean, rollout, total[0], total[1], (applied[0] / max(applied[1], 1e-30)) ** 0.5)
    return None


def maybe_install(model_chunks) -> None:
    if not enabled():
        return
    hooks = 0
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if re.search(r"(?:^|\.)layers\.\d+\.(self_attention|mlp)$", name):
                module.register_forward_hook(_noise_hook)
                hooks += 1
            elif name.endswith("output_layer"):
                module.register_forward_hook(_logits_hook)
                hooks += 1
    logger.info("sensitivity installed: %d hooks, noise %s, pairs %s", hooks, NOISE, PAIRS)
