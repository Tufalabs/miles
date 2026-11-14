"""Penalize how much the sampler's K rounding moves the trainer's log-probs (experiment, LLMT-83).

With MILES_GAIN_PENALTY=<coefficient> (and MILES_SAMPLER_EMULATION naming what to round, e.g. k_nvfp4) the
policy loss gains the term

    coefficient * huber(log p_plain(token) - log p_rounded(token))        per supervised token

where p_plain is the trainer's forward as it is and p_rounded the same weights with the sampler emulation on.
Both forwards carry gradient, in two passes over each micro-batch that accumulate into the same gradient:

  scoring pass  a plain forward without gradient that keeps the plain log-probs on the batch (the run may have
                no separate scoring pass: --skip-actor-forward-only)
  rounded pass  the emulation is on; the loss is the penalty against the kept plain log-probs; its log-probs
                are kept on the batch too
  plain pass    the usual forward; the loss is the policy loss plus the penalty against the rounded log-probs

Outside the rounded pass the emulation is off, so scoring and the policy term see the plain model. The huber
threshold is MILES_GAIN_PENALTY_DELTA (default 1 nat). Metrics, from the plain pass: `gain_d2` (mean squared
difference), `gain_abs_d` (mean absolute difference), `gain_penalty` (the term before the coefficient's sum).

MILES_GAIN_PROJECTION=1 (in place of the coefficient) does not add the penalty to the loss. Each training step
first computes the penalty's own gradient (the penalty phase: both passes with the policy term dropped), then
the policy gradient as usual, and removes from the policy gradient its component along the penalty's gradient
when that component would raise the penalty (megatron_utils/gain_projection.py).
"""

import contextlib
import os

import torch

ENV_VAR = "MILES_GAIN_PENALTY"
DELTA_ENV_VAR = "MILES_GAIN_PENALTY_DELTA"
_COEFFICIENT = float(os.environ.get(ENV_VAR, "0"))
_DELTA = float(os.environ.get(DELTA_ENV_VAR, "1"))
PROJECTION_ENV_VAR = "MILES_GAIN_PROJECTION"
_PROJECTION = os.environ.get(PROJECTION_ENV_VAR, "") == "1"
assert not (_PROJECTION and _COEFFICIENT > 0), (
    f"set {ENV_VAR} or {PROJECTION_ENV_VAR}, not both"
)
_ROUNDED_LOG_PROBS_KEY = "gain_penalty_rounded_log_probs"
_PLAIN_LOG_PROBS_KEY = "gain_penalty_plain_log_probs"
_rounding_active = False
_scoring = False
_penalty_phase = False
_phase_sums = [
    0.0,
    0.0,
]  # of the current penalty phase: sum of squared differences, supervised tokens


def configured() -> bool:
    """The penalty or the projection is on for this run: the sampler emulation then applies only in rounded passes."""
    return _COEFFICIENT > 0 or _PROJECTION


def projection_enabled() -> bool:
    return _PROJECTION


def in_penalty_phase() -> bool:
    return _penalty_phase


def enabled() -> bool:
    """The penalty is part of the loss of the pass that runs now."""
    return _penalty_phase if _PROJECTION else _COEFFICIENT > 0


@contextlib.contextmanager
def penalty_phase():
    """The projection's first phase: forward/backward passes whose loss is the penalty alone."""
    global _penalty_phase
    _penalty_phase = True
    _phase_sums[0] = _phase_sums[1] = 0.0
    try:
        yield
    finally:
        _penalty_phase = False


def phase_sums() -> tuple[float, float]:
    """Sum of squared log-prob differences and supervised tokens of the last penalty phase, on this rank."""
    return _phase_sums[0], _phase_sums[1]


def rounding_active() -> bool:
    """True inside the rounded pass: the sampler emulation applies only then when the penalty is enabled."""
    return _rounding_active


@contextlib.contextmanager
def scoring_pass():
    """A plain forward whose only purpose is to leave the plain log-probs on the batch."""
    global _scoring
    _scoring = True
    try:
        yield
    finally:
        _scoring = False


def quiet() -> bool:
    """True in the penalty's own passes, which must not write the per-token debug dumps."""
    return _scoring or _rounding_active or _penalty_phase


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
    return torch.where(
        magnitude <= _DELTA, 0.5 * d * d, _DELTA * (magnitude - 0.5 * _DELTA)
    )


def apply(
    batch: dict,
    pg_loss: torch.Tensor,
    log_probs: torch.Tensor,
    active_tokens: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The per-token loss of the current pass and, in the plain pass, per-token metrics.

    `log_probs`: this forward's log-probs of the sampled tokens.
    """
    if _scoring:
        batch[_PLAIN_LOG_PROBS_KEY] = log_probs.detach()
        return pg_loss, {}
    coefficient = 1.0 if _PROJECTION else _COEFFICIENT
    if rounding_active():
        batch[_ROUNDED_LOG_PROBS_KEY] = log_probs.detach()
        plain = batch.get(_PLAIN_LOG_PROBS_KEY)
        assert plain is not None, (
            "gain penalty: the scoring pass did not run before the rounded pass"
        )
        d = plain - log_probs
        d = torch.where(
            active_tokens,
            torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0),
            d.new_zeros(()),
        )
        return coefficient * _huber(d), {}
    rounded = batch.get(_ROUNDED_LOG_PROBS_KEY)
    assert rounded is not None, (
        "gain penalty: the rounded pass did not run before the plain pass"
    )
    d = log_probs - rounded
    d = torch.where(
        active_tokens,
        torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0),
        d.new_zeros(()),
    )
    penalty = _huber(d)
    detached = d.detach()
    metrics = {
        "gain_d2": detached * detached,
        "gain_abs_d": detached.abs(),
        "gain_penalty": penalty.detach(),
    }
    if _PROJECTION:  # the penalty phase: the policy term is dropped
        _phase_sums[0] += (detached * detached).sum().item()
        _phase_sums[1] += active_tokens.sum().item()
        return 0.0 * pg_loss + penalty, metrics
    return pg_loss + coefficient * penalty, metrics
