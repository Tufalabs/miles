"""Pull the trainer toward the rollout engine's distribution on the sampled tokens (experiment, LLMT-83).

MILES_ROLLOUT_KL=<coefficient> adds to the policy loss, per supervised token, a term in the difference
d = log p_trainer(token) - log p_rollout(token), with the rollout engine's log-prob as a constant:

  MILES_ROLLOUT_KL_TYPE=logratio  coefficient * (-d). Its mean over sampled tokens is the usual estimate of
                                  KL(rollout || trainer); its gradient is -coefficient * grad log p_trainer on
                                  every sampled token, i.e. every advantage shifted up by the coefficient.
  MILES_ROLLOUT_KL_TYPE=huber     (default) coefficient * huber(d), threshold MILES_ROLLOUT_KL_DELTA (1 nat):
                                  each token's trainer log-prob is pulled toward the rollout engine's, up or down.

Logged: `rollout_kl_penalty` (the term before the coefficient).
"""

import os

import torch

ENV_VAR = "MILES_ROLLOUT_KL"
TYPE_ENV_VAR = "MILES_ROLLOUT_KL_TYPE"
DELTA_ENV_VAR = "MILES_ROLLOUT_KL_DELTA"
_COEFFICIENT = float(os.environ.get(ENV_VAR, "0"))
_TYPE = os.environ.get(TYPE_ENV_VAR, "huber")
_DELTA = float(os.environ.get(DELTA_ENV_VAR, "1"))
assert _TYPE in ("huber", "logratio"), f"{TYPE_ENV_VAR}={_TYPE!r}: expected huber or logratio"


def enabled() -> bool:
    return _COEFFICIENT > 0


def apply(pg_loss: torch.Tensor, log_probs: torch.Tensor, rollout_log_probs: torch.Tensor, active_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The per-token policy loss with the term added, and the per-token term for logging."""
    d = log_probs - rollout_log_probs.detach()
    d = torch.where(active_tokens, torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), d.new_zeros(()))
    if _TYPE == "logratio":
        penalty = -d
    else:
        magnitude = d.abs()
        penalty = torch.where(magnitude <= _DELTA, 0.5 * d * d, _DELTA * (magnitude - 0.5 * _DELTA))
    return pg_loss + _COEFFICIENT * penalty, {"rollout_kl_penalty": penalty.detach()}
