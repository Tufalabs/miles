"""Remove from the policy gradient its component along the gain penalty's gradient (experiment, LLMT-83).

With MILES_GAIN_PROJECTION=1 (see training_utils/loss/hub/gain_penalty.py) a training step first computes the
gradient of the gain penalty alone on the step's micro-batches, then the policy gradient as usual. If the two
have a negative dot product, a step along the policy gradient would raise the penalty to first order; the
policy gradient then loses that component: g <- g - (g . p / p . p) p. Nothing is done when the dot product is
positive. The penalty itself is never added to the loss.

The penalty's gradient costs three passes, and its direction changes little from batch to batch and from step
to step, so it can be kept across steps: MILES_GAIN_PROJECTION_EVERY=N computes it on every N-th training step
and projects the steps between off the kept direction. MILES_GAIN_PROJECTION_EMA=b makes the kept direction an
average over the computed ones, d <- b d + (1 - b) p / |p|. Each newly computed gradient logs its cosine with
the direction kept until then (refresh_cos).

The projection is first order and its direction is an estimate, so the gain still creeps up. MILES_GAIN_SERVO=k
adds a restoring step: with gain the penalty phases' mean squared log-prob difference (smoothed over the
computed steps) and MILES_GAIN_SERVO_TARGET its value at the start of the run, the gradient gains
min(k * max(gain / target - 1, 0), 1) * |policy gradient| along the unit penalty direction, a descent step on
the penalty. At or below the target nothing is added.
"""

import logging
import math
import os
from collections.abc import Callable, Sequence

import torch

from miles.backends.training_utils.loss.hub import gain_penalty

logger = logging.getLogger(__name__)

FRACTION_ENV_VAR = "MILES_GAIN_PROJECTION_FRACTION"
_FRACTION = float(os.environ.get(FRACTION_ENV_VAR, "1"))
assert 0 < _FRACTION <= 1, f"{FRACTION_ENV_VAR}={_FRACTION}: expected a share in (0, 1]"
EVERY_ENV_VAR = "MILES_GAIN_PROJECTION_EVERY"
_EVERY = int(os.environ.get(EVERY_ENV_VAR, "1"))
assert _EVERY >= 1, f"{EVERY_ENV_VAR}={_EVERY}: expected a number of steps, at least 1"
EMA_ENV_VAR = "MILES_GAIN_PROJECTION_EMA"
_EMA = float(os.environ.get(EMA_ENV_VAR, "0"))
assert 0 <= _EMA < 1, f"{EMA_ENV_VAR}={_EMA}: expected a decay in [0, 1)"
SERVO_ENV_VAR = "MILES_GAIN_SERVO"
_SERVO = float(os.environ.get(SERVO_ENV_VAR, "0"))
SERVO_TARGET_ENV_VAR = "MILES_GAIN_SERVO_TARGET"
_SERVO_TARGET = float(os.environ.get(SERVO_TARGET_ENV_VAR, "0"))
assert _SERVO >= 0 and (_SERVO == 0 or _SERVO_TARGET > 0), f"{SERVO_ENV_VAR}={_SERVO} needs {SERVO_TARGET_ENV_VAR} > 0"
_SERVO_SMOOTHING = 0.8  # decay of the gain's average over the steps that compute the penalty
# The direction kept across steps, the training steps it has served, what the last computed gradient's cosine
# with the direction before it was, and the smoothed gain.
_kept: dict = {"direction": None, "age": 0, "refresh_cos": float("nan"), "refreshed": False, "gain": None}


def _named_grads(model_chunks) -> list[tuple[str, torch.Tensor]]:
    out = []
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            grad = getattr(param, "main_grad", None)  # config-access-exempt: DDP attaches main_grad at runtime
            if grad is not None:
                out.append((name, grad))
    return out


def _reduced_dots(named: list[tuple[str, torch.Tensor]], others: list[torch.Tensor]) -> tuple[float, float, float]:
    """grad . other, |other|^2 and |grad|^2 over the parameters of all ranks."""
    # the parameters every rank holds (0-2) and the experts (3-5)
    stats = torch.zeros(6, dtype=torch.float64, device=named[0][1].device)
    for (name, grad), other in zip(named, others, strict=True):
        offset = 3 if ".experts." in name else 0
        g, o = grad.reshape(-1).float(), other.reshape(-1).float()
        stats[offset] += torch.dot(g, o).double()
        stats[offset + 1] += torch.dot(o, o).double()
        stats[offset + 2] += torch.dot(g, g).double()
    world = 1
    if torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(stats)
    stats[:3] /= world  # the shared parameters' gradients are the same on every rank after the sync
    dot, other2, grad2 = (float(stats[i] + stats[i + 3]) for i in range(3))
    return dot, other2, grad2


@torch.no_grad()
def _keep(named: list[tuple[str, torch.Tensor]]) -> list[torch.Tensor]:
    """The direction to project off, from the penalty gradient now in the gradient buffers and the kept one."""
    previous = _kept["direction"]
    new2 = 0.0
    if previous is not None:
        dot, previous2, new2 = _reduced_dots(named, previous)
        _kept["refresh_cos"] = dot / max((previous2 * new2) ** 0.5, 1e-30)
    if previous is None:
        scale = 1.0
        if _EMA > 0:  # the average is over unit vectors
            scale = 1 / max(_reduced_dots(named, [grad for _, grad in named])[2] ** 0.5, 1e-30)
        # bf16 halves the copy of an fp32 gradient buffer; the direction keeps 8 bits per element
        return [(grad * scale).to(torch.bfloat16) for _, grad in named]
    for (_, grad), kept in zip(named, previous, strict=True):  # in place: one copy of the direction at a time
        if _EMA == 0:
            kept.copy_(grad)
        else:
            kept.mul_(_EMA).add_(grad, alpha=(1 - _EMA) / max(new2**0.5, 1e-30))
    return previous


def penalty_gradient(
    run_pass: Callable[[int], object],
    data_iterator: Sequence,
    model_chunks,
    zero_grads: Callable[[], None],
    num_microbatches: int,
) -> list[torch.Tensor]:
    """The gain penalty's gradient on the step's first micro-batches (all of them, or the share
    MILES_GAIN_PROJECTION_FRACTION: the projection needs a direction, and a part of the batch gives one at a
    part of the cost); leaves the gradients zeroed and the data iterators where they were. With
    MILES_GAIN_PROJECTION_EVERY above 1, the steps between two computations return the kept direction."""
    _kept["refreshed"] = _kept["direction"] is None or _kept["age"] >= _EVERY
    if _kept["refreshed"]:
        offsets = [iterator.offset for iterator in data_iterator]
        with gain_penalty.penalty_phase():
            run_pass(max(1, math.ceil(_FRACTION * num_microbatches)))
        _kept["direction"] = _keep(_named_grads(model_chunks))
        _kept["age"] = 0
        zero_grads()
        for iterator, offset in zip(data_iterator, offsets, strict=True):
            iterator.offset = offset
    _kept["age"] += 1
    return _kept["direction"]


@torch.no_grad()
def project_out_penalty_gradient(model_chunks, penalty_grads: list[torch.Tensor], rollout_id: int) -> None:
    """Project the policy gradient (now in the gradient buffers) off the penalty's gradient when they oppose."""
    named = _named_grads(model_chunks)
    assert len(named) == len(penalty_grads)
    dot, penalty2, policy2 = _reduced_dots(named, penalty_grads)
    # the last penalty phase's sum of squared log-prob differences and its supervised tokens
    sums = torch.tensor(gain_penalty.phase_sums(), dtype=torch.float64, device=named[0][1].device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sums)
    gain_d2 = float(sums[0] / sums[1].clamp_min(1))
    if _kept["refreshed"]:
        previous = _kept["gain"]
        _kept["gain"] = gain_d2 if previous is None else _SERVO_SMOOTHING * previous + (1 - _SERVO_SMOOTHING) * gain_d2
    projected = dot < 0 and penalty2 > 0
    # in units of the policy gradient's norm, along the unit penalty direction
    servo_push = min(_SERVO * max(_kept["gain"] / _SERVO_TARGET - 1, 0.0), 1.0) if _SERVO > 0 and penalty2 > 0 else 0.0
    coefficient = servo_push * (policy2 / penalty2) ** 0.5 if servo_push > 0 else 0.0
    if projected:
        coefficient -= dot / penalty2
    if coefficient != 0:
        for (_, grad), penalty in zip(named, penalty_grads, strict=True):
            grad.add_(penalty.to(grad.dtype), alpha=coefficient)
    logger.info(
        "gain projection rollout=%d cos=%.4f policy_norm=%.4e penalty_norm=%.4e removed_share=%.4e gain_d2=%.4e projected=%d refreshed=%d refresh_cos=%.4f servo_push=%.4f gain_smoothed=%.4e",
        rollout_id,
        dot / max((penalty2 * policy2) ** 0.5, 1e-30),
        policy2**0.5,
        penalty2**0.5,
        dot * dot / max(penalty2 * policy2, 1e-30) if projected else 0.0,
        gain_d2,
        int(projected),
        int(_kept["refreshed"]),
        _kept["refresh_cos"],
        servo_push,
        _kept["gain"],
    )
