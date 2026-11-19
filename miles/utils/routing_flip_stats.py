"""Routing flips between the sampler and the trainer, and router replay per rollout (LLMT-83 mismatch probe).

Needs --use-rollout-routing-replay: the sampler returns each token's top-k experts per layer and the trainer's
forward has them (replay_base.py). With MILES_ROUTING_FLIP_STATS=1, in every replayed forward each MoE layer's own
top-k is computed as well and compared with the sampler's, per token: a flip is a token whose two expert sets
differ. Counted per layer, apart for generated tokens (positions from the prompt's length to the second-to-last,
whose sampler routing came from decode forwards) and prompt tokens (prefill).
MILES_ROUTING_REPLAY_ROLLOUTS=<id>,...  the rollouts whose forward uses the sampler's routing; the others use the
trainer's own (default: all replayed, as --use-rollout-routing-replay alone does).
The score-centering loss adds, per rollout, the head KL of generated tokens at positions where any layer flipped
against the rest, and binned by the number of layers that flipped at the position (0, 1, 2-3, 4-7, 8-15, 16+) and by
the swapped gate mass there (the trainer's router probability, summed over layers, on its experts that the sampler
did not pick: 0, up to 0.01, 0.03, 0.1, 0.3, above). Each rank logs its running totals after every micro-batch: "routing flips rollout=<id> {json}"
with, per layer, [generated tokens, flipped, prompt tokens, flipped], and kl_flip/kl_all/n_flip/n_all.
"""

import json
import logging
import os

import torch

from miles.backends.training_utils.metrics import debug_dump

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("MILES_ROUTING_FLIP_STATS", "") == "1"
_REPLAY = os.environ.get("MILES_ROUTING_REPLAY_ROLLOUTS", "").strip()
REPLAY_ROLLOUTS = {int(r) for r in _REPLAY.split(",") if r.strip()} if _REPLAY else None

_batch: dict = {}
_totals: dict[int, dict] = {}


def enabled() -> bool:
    return ENABLED


def use_replay() -> bool:
    """Whether this forward uses the sampler's routing."""
    if REPLAY_ROLLOUTS is None:
        return True
    return debug_dump._training_rollout_id in REPLAY_ROLLOUTS


def set_batch(total_lengths, response_lengths) -> None:
    """Called with each micro-batch's sequence lengths (packed in this order) before its forward."""
    _batch.clear()
    if not ENABLED:
        return
    generated = []
    for total, response in zip(total_lengths, response_lengths, strict=True):
        total, response = int(total), int(response)
        mask = torch.zeros(total, dtype=torch.bool)
        mask[total - response : total - 1] = True
        generated.append(mask)
    generated = torch.cat(generated) if generated else torch.zeros(0, dtype=torch.bool)
    _batch.update(generated=generated, flip=None, total_lengths=[int(t) for t in total_lengths], response_lengths=[int(r) for r in response_lengths])


COUNT_BINS = (0, 1, 2, 4, 8, 16)
MASS_BINS = (0.0, 0.01, 0.03, 0.1, 0.3)


def record(layer: int, own: torch.Tensor, sampler: torch.Tensor, scores: torch.Tensor | None = None) -> None:
    """own, sampler: [tokens, k] expert ids of one MoE layer for the packed micro-batch (padding: -1); scores: the
    router's scores [tokens, experts] (probabilities, or logits that are softmaxed here), for the swapped gate mass:
    the trainer's probability on its own experts that the sampler did not pick."""
    if not _batch:
        return
    own, sampler = own.reshape(-1, own.shape[-1]), sampler.reshape(-1, sampler.shape[-1])
    valid = (sampler >= 0).all(dim=-1)
    same = (own.sort(dim=-1).values == sampler.sort(dim=-1).values).all(dim=-1)
    flip = valid & ~same
    n = flip.shape[0]
    generated = torch.zeros(n, dtype=torch.bool, device=flip.device)
    known = min(n, _batch["generated"].shape[0])
    generated[:known] = _batch["generated"][:known].to(flip.device)
    prompt = valid & ~generated
    generated = generated & valid
    rollout = debug_dump._training_rollout_id
    totals = _totals.setdefault(rollout, {"layers": {}, "kl": [0.0, 0.0, 0, 0]})
    entry = totals["layers"].setdefault(layer, [0, 0, 0, 0])
    entry[0] += int(generated.sum())
    entry[1] += int((flip & generated).sum())
    entry[2] += int(prompt.sum())
    entry[3] += int((flip & prompt).sum())
    _batch["flip"] = flip if _batch["flip"] is None else (_batch["flip"] | flip)
    count = flip.to(torch.int32)
    _batch["count"] = count if _batch.get("count") is None else _batch["count"] + count
    if scores is not None:
        probs = scores.reshape(own.shape[0], -1).float()
        # the patched top-k may receive the score function's output (probabilities) or raw logits
        if bool((probs < 0).any()) or not torch.allclose(probs.sum(-1), torch.ones_like(probs[:, 0]), atol=1e-2):
            probs = torch.softmax(probs, dim=-1)
        picked = (own.unsqueeze(-1) == sampler.unsqueeze(-2)).any(-1)
        mass = (probs.gather(1, own.clamp_min(0).long()) * (~picked).float()).sum(-1) * flip.float()
        _batch["mass"] = mass if _batch.get("mass") is None else _batch["mass"] + mass


def _per_response(values: torch.Tensor, device) -> torch.Tensor:
    values = values.to(device)
    out, start = [], 0
    for total, response in zip(_batch["total_lengths"], _batch["response_lengths"], strict=True):
        first = start + total - response - 1
        out.append(values[first : first + response])
        start += total
    return torch.cat(out)


def response_flips(device) -> torch.Tensor | None:
    """Per response token of the micro-batch (the loss's order): whether any layer flipped at the position whose
    forward predicts it."""
    if not _batch or _batch["flip"] is None:
        return None
    return _per_response(_batch["flip"], device)


def add_kl(per_token_kl: torch.Tensor, active: torch.Tensor) -> None:
    """The head KL per response token (the loss's order), split at flipped positions; logs the running totals."""
    flips = response_flips(per_token_kl.device)
    rollout = debug_dump._training_rollout_id
    if flips is None or rollout not in _totals or flips.shape[0] != per_token_kl.shape[0]:
        return
    kl = per_token_kl.detach().float() * active
    binned = _totals[rollout].setdefault("bins", {"count": [[0.0, 0] for _ in COUNT_BINS], "mass": [[0.0, 0] for _ in range(len(MASS_BINS) + 1)]})
    count = _per_response(_batch["count"], kl.device)
    for i, low in enumerate(COUNT_BINS):
        high = COUNT_BINS[i + 1] if i + 1 < len(COUNT_BINS) else 10**9
        sel = (count >= low) & (count < high) & active.bool()
        binned["count"][i][0] += float(kl[sel].sum())
        binned["count"][i][1] += int(sel.sum())
    if _batch.get("mass") is not None:
        mass = _per_response(_batch["mass"], kl.device)
        edges = [0.0, *MASS_BINS[1:], float("inf")]
        selections = [mass <= 0] + [(mass > edges[i]) & (mass <= edges[i + 1]) for i in range(len(edges) - 1)]
        for i, sel in enumerate(selections):
            sel = sel & active.bool()
            binned["mass"][i][0] += float(kl[sel].sum())
            binned["mass"][i][1] += int(sel.sum())
    totals = _totals[rollout]["kl"]
    totals[0] += float(kl[flips].sum())
    totals[1] += float(kl.sum())
    totals[2] += int((flips & active.bool()).sum())
    totals[3] += int(active.sum())
    totals_out = _totals[rollout]
    logger.info(
        "routing flips rollout=%d replay=%d %s",
        rollout,
        int(use_replay()),
        json.dumps({"layers": totals_out["layers"], "kl_flip": totals[0], "kl_all": totals[1], "n_flip": totals[2], "n_all": totals[3], "bins": totals_out.get("bins")}),
    )
