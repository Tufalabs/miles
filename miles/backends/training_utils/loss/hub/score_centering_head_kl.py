"""A pull of the trainer toward the sampler's returned candidates, added to score centering (experiment, LLMT-83).

Score centering moves the logits at a sampled token a by A * (onehot(a) - q): the step holds the sampler's q
and never the trainer's p, so nothing in it reacts to a candidate the trainer rates differently from the
sampler. MILES_SC_HEAD_KL=<c> adds, per prefix, c times a divergence between the two distributions over the
candidates the sampler returned, with q held fixed. MILES_SC_HEAD_KL_KIND selects it:

    forward (default)  KL(q || p). Its descent step on the logits is c * (q - p), the term by which plain
                       policy gradient differs from score centering, with a fixed sign in place of A.
    reverse            KL(p || q). Its descent step lowers candidate v by c * p_v * (log(p_v / q_v) - KL):
                       it acts on the candidates the trainer rates above the sampler.

Both distributions are renormalized over the valid candidates (the mass outside them is under 0.1%). The
divergence is logged per token as `sc_head_kl` whether or not the term is on.
"""

import os

import torch

ENV_VAR = "MILES_SC_HEAD_KL"
KIND_ENV_VAR = "MILES_SC_HEAD_KL_KIND"
_COEFFICIENT = float(os.environ.get(ENV_VAR, "") or 0.0)
_KIND = os.environ.get(KIND_ENV_VAR, "") or "forward"
assert _KIND in ("forward", "reverse"), f"{KIND_ENV_VAR}={_KIND!r}: expected forward or reverse"

# A finite stand-in for minus infinity: exp(-1e4) is 0 in float32 and differences stay finite.
_MASKED = -1.0e4


def coefficient() -> float:
    return _COEFFICIENT


def divergence(
    train_head_log_probs: torch.Tensor, rollout_head_log_probs: torch.Tensor, head_mask: torch.Tensor
) -> torch.Tensor:
    """Per prefix, the divergence over the valid candidates; zero where there are none. The gradient flows
    through the trainer's log-probs only."""
    rollout = rollout_head_log_probs.detach().float()
    train = train_head_log_probs.float()
    valid = head_mask & torch.isfinite(rollout) & torch.isfinite(train.detach())
    log_p = torch.where(valid, train, _MASKED).log_softmax(-1)
    log_q = torch.where(valid, rollout, _MASKED).log_softmax(-1)
    if _KIND == "forward":
        terms = log_q.exp() * (log_q - log_p)
    else:
        terms = log_p.exp() * (log_p - log_q)
    return torch.where(valid, terms, 0.0).sum(-1) * valid.any(-1)
