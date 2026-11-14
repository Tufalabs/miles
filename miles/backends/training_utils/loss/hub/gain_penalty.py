"""Penalize how much the sampler's K rounding moves the trainer's log-probs (experiment, LLMT-83).

With MILES_GAIN_PENALTY=<coefficient> (and MILES_SAMPLER_EMULATION naming what to round, e.g. k_nvfp4) the
policy loss gains the term

    coefficient * huber(log p_plain(token) - log p_rounded(token))        per supervised token

where p_plain is the trainer's forward as it is and p_rounded the same weights with the sampler emulation on.
Both forwards carry gradient, in two passes over each micro-batch that accumulate into the same gradient:

  rounded pass  the emulation is on; the loss is the penalty against the plain log-probs of the no-grad
                scoring pass (`batch["log_probs"]`); its log-probs are kept on the batch
  plain pass    the usual forward; the loss is the policy loss plus the penalty against the kept log-probs

Outside the rounded pass the emulation is off, so scoring and the policy term see the plain model. The huber
threshold is MILES_GAIN_PENALTY_DELTA (default 1 nat). Metrics, from the plain pass: `gain_d2` (mean squared
difference), `gain_abs_d` (mean absolute difference), `gain_penalty` (the term before the coefficient's sum).
"""

import contextlib
import os

import torch

ENV_VAR = "MILES_GAIN_PENALTY"
DELTA_ENV_VAR = "MILES_GAIN_PENALTY_DELTA"
_COEFFICIENT = float(os.environ.get(ENV_VAR, "0"))
_DELTA = float(os.environ.get(DELTA_ENV_VAR, "1"))
_ROUNDED_LOG_PROBS_KEY = "gain_penalty_rounded_log_probs"
_rounding_active = False


def enabled() -> bool:
    return _COEFFICIENT > 0


def rounding_active() -> bool:
    """True inside the rounded pass: the sampler emulation applies only then when the penalty is enabled."""
    return _rounding_active


@contextlib.contextmanager
def rounded_pass():
    global _rounding_active
    _rounding_active = True
    try:
        yield
    finally:
        _rounding_active = False


def _huber(d: torch.Tensor) -> torch.Tensor:
    magnitude = d.abs()
    return torch.where(magnitude <= _DELTA, 0.5 * d * d, _DELTA * (magnitude - 0.5 * _DELTA))


def apply(
    batch: dict,
    pg_loss: torch.Tensor,
    log_probs: torch.Tensor,
    plain_log_probs: torch.Tensor,
    active_tokens: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The per-token loss of the current pass and, in the plain pass, per-token metrics.

    `log_probs`: this forward's log-probs of the sampled tokens (with gradient). `plain_log_probs`: the plain
    trainer's from the no-grad scoring pass.
    """
    if rounding_active():
        batch[_ROUNDED_LOG_PROBS_KEY] = log_probs.detach()
        d = plain_log_probs.detach() - log_probs
        d = torch.where(active_tokens, torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), d.new_zeros(()))
        return _COEFFICIENT * _huber(d), {}
    rounded = batch.get(_ROUNDED_LOG_PROBS_KEY)
    assert rounded is not None, "gain penalty: the rounded pass did not run before the plain pass"
    d = log_probs - rounded
    d = torch.where(active_tokens, torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), d.new_zeros(()))
    penalty = _huber(d)
    detached = d.detach()
    metrics = {"gain_d2": detached * detached, "gain_abs_d": detached.abs(), "gain_penalty": penalty.detach()}
    return pg_loss + _COEFFICIENT * penalty, metrics
