"""Score centering whose update leaves the trainer-sampler KL unchanged to first order (experiment, LLMT-83).

Score centering moves the logits at a sampled token a by A * (onehot(a) - q), q the sampler's distribution.
With q held fixed, that step changes KL(q || p) at the same prefix by A * kappa to first order, where

    kappa = (p(a) - q(a)) - sum_v (p(v) - q(v)) q(v).

kappa has zero mean under the sampler, so the expected change is Cov_q(A, kappa): it is positive when the
tokens the sampler prefers more than the trainer does fail more often, and score centering leaves it in.
`kl_push` (logged as `sc_kl_push`) is that sum over the tokens of a micro-batch, A * kappa under the loss
reducer; it is always logged.

MILES_SC_KL_NEUTRAL=1 takes the linear fit of the advantages on kappa out of the advantages, per micro-batch:
A' = A - (sum A kappa / sum kappa^2) kappa, so the micro-batch's kl_push is zero. MILES_SC_KL_NEUTRAL=positive
does so only when the sum is positive (the update would raise the KL). The sums use the head candidates; the
tail outside them holds under 0.1% of the mass.
"""

import os
from collections.abc import Callable

import torch

ENV_VAR = "MILES_SC_KL_NEUTRAL"
_MODE = os.environ.get(ENV_VAR, "")
assert _MODE in ("", "1", "positive"), f"{ENV_VAR}={_MODE!r}: expected 1 or positive"


@torch.no_grad()
def adjust(
    advantages: torch.Tensor,
    train_log_probs: torch.Tensor,
    train_head_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    rollout_head_log_probs: torch.Tensor,
    head_mask: torch.Tensor,
    active: torch.Tensor,
    reduce: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The advantages to use, and per-token tensors to log: A * kappa before the change, and the fit's slope."""
    p_head = train_head_log_probs.float().exp().masked_fill(~head_mask, 0.0)
    q_head = rollout_head_log_probs.float().exp().masked_fill(~head_mask, 0.0)
    kappa = (train_log_probs.float().exp() - rollout_log_probs.float().exp()) - ((p_head - q_head) * q_head).sum(-1)
    kappa = torch.where(active, torch.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0), 0.0).to(advantages.dtype)
    push = advantages * kappa
    slope = advantages.new_zeros(())
    if _MODE:
        total, square = reduce(push), reduce(kappa * kappa)
        if square > 1e-20 and (_MODE == "1" or total > 0):
            slope = total / square
            advantages = torch.where(active, advantages - slope * kappa, advantages)
    return advantages, {"sc_kl_push": push, "sc_kl_slope": slope.expand_as(push)}
