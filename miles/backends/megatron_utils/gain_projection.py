"""Remove from the policy gradient its component along the gain penalty's gradient (experiment, LLMT-83).

With MILES_GAIN_PROJECTION=1 (see training_utils/loss/hub/gain_penalty.py) a training step first computes the
gradient of the gain penalty alone on the step's micro-batches, then the policy gradient as usual. If the two
have a negative dot product, a step along the policy gradient would raise the penalty to first order; the
policy gradient then loses that component: g <- g - (g . p / p . p) p. Nothing is done when the dot product is
positive. The penalty itself is never added to the loss.
"""

import logging
from collections.abc import Callable, Sequence

import torch

from miles.backends.training_utils.loss.hub import gain_penalty

logger = logging.getLogger(__name__)


def _named_grads(model_chunks) -> list[tuple[str, torch.Tensor]]:
    out = []
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            grad = getattr(param, "main_grad", None)  # config-access-exempt: DDP attaches main_grad at runtime
            if grad is not None:
                out.append((name, grad))
    return out


def penalty_gradient(run_pass: Callable[[], object], data_iterator: Sequence, model_chunks, zero_grads: Callable[[], None]) -> list[torch.Tensor]:
    """The gain penalty's gradient on the step's micro-batches; leaves the gradients zeroed and the data
    iterators where they were."""
    offsets = [iterator.offset for iterator in data_iterator]
    with gain_penalty.penalty_phase():
        run_pass()
    # bf16 halves the copy of an fp32 gradient buffer; the direction keeps 8 bits per element
    snapshot = [grad.to(torch.bfloat16) for _, grad in _named_grads(model_chunks)]
    zero_grads()
    for iterator, offset in zip(data_iterator, offsets, strict=True):
        iterator.offset = offset
    return snapshot


@torch.no_grad()
def project_out_penalty_gradient(model_chunks, penalty_grads: list[torch.Tensor], rollout_id: int) -> None:
    """Project the policy gradient (now in the gradient buffers) off the penalty's gradient when they oppose."""
    named = _named_grads(model_chunks)
    assert len(named) == len(penalty_grads)
    # dot, |penalty|^2, |policy|^2 for the parameters every rank holds (0-2) and for the experts (3-5), then
    # the penalty phase's sum of squared log-prob differences and its supervised tokens (6-7)
    stats = torch.zeros(8, dtype=torch.float64, device=named[0][1].device)
    for (name, grad), penalty in zip(named, penalty_grads, strict=True):
        offset = 3 if ".experts." in name else 0
        g, p = grad.reshape(-1).float(), penalty.reshape(-1).float()
        stats[offset] += torch.dot(g, p).double()
        stats[offset + 1] += torch.dot(p, p).double()
        stats[offset + 2] += torch.dot(g, g).double()
    stats[6], stats[7] = gain_penalty.phase_sums()
    world = 1
    if torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(stats)
    stats[:3] /= world  # the shared parameters' gradients are the same on every rank after the sync
    dot, penalty2, policy2 = (float(stats[i] + stats[i + 3]) for i in range(3))
    projected = dot < 0 and penalty2 > 0
    if projected:
        coefficient = dot / penalty2
        for (_, grad), penalty in zip(named, penalty_grads, strict=True):
            grad.sub_(penalty.to(grad.dtype), alpha=coefficient)
    logger.info(
        "gain projection rollout=%d cos=%.4f policy_norm=%.4e penalty_norm=%.4e removed_share=%.4e gain_d2=%.4e projected=%d",
        rollout_id,
        dot / max((penalty2 * policy2) ** 0.5, 1e-30),
        policy2**0.5,
        penalty2**0.5,
        dot * dot / max(penalty2 * policy2, 1e-30) if projected else 0.0,
        float(stats[6] / stats[7].clamp_min(1)),
        int(projected),
    )
