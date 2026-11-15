"""Score centering from the sampled token's rollout log-prob alone (experiment, LLMT-83).

Plain policy gradient moves the logits of a sampled token a by A * (onehot(a) - p), with p the trainer's
distribution. Score centering uses the sampler's distribution q there, A * (onehot(a) - q), which needs the
sampler's top-k log-probs. MILES_SAMPLED_TOKEN_CENTERING=1 uses the one number every rollout already carries,
q(a): the distribution that has the sampler's probability on the sampled token and the trainer's, rescaled,
on every other token. The logit gradient is then

    A * (onehot(a) - p) * (1 - q(a)) / (1 - p(a)),

plain PG's gradient with a per-token weight. A token the sampler was sure of (q(a) near 1) and the trainer
finds unlikely gets a small step in place of a large one. Logged: `stc_weight` (the weight).
"""

import os

import torch

ENV_VAR = "MILES_SAMPLED_TOKEN_CENTERING"
_ENABLED = os.environ.get(ENV_VAR, "") == "1"
_EPS = 1e-6


def enabled() -> bool:
    return _ENABLED


def apply(pg_loss: torch.Tensor, log_probs: torch.Tensor, rollout_log_probs: torch.Tensor, active_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The per-token policy loss times (1 - q(a)) / (1 - p(a)), and the weight for logging."""
    with torch.no_grad():
        sampler_rest = (-torch.expm1(rollout_log_probs.float())).clamp(_EPS, 1.0)
        trainer_rest = (-torch.expm1(log_probs.float())).clamp(_EPS, 1.0)
        weight = torch.where(active_tokens, torch.nan_to_num(sampler_rest / trainer_rest, nan=1.0), torch.ones_like(sampler_rest))
    return pg_loss * weight.to(pg_loss.dtype), {"stc_weight": weight}
