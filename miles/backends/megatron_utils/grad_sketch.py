"""Save a small sketch of every parameter's gradient, to compare gradients across jobs (experiment, LLMT-83).

Enabled by MILES_GRAD_SKETCH=1; needs --dump-details. After the backward pass of each training step every rank
writes `<dump-details>/grad_sketch/rollout_<id>_step_<s>_rank_<r>.pt`: for each parameter its squared norm,
its size and a count sketch of its gradient (each element goes to one of BUCKETS buckets with a random sign;
a tensor with at most BUCKETS elements is stored whole). The buckets and signs come from a generator seeded by
the parameter's name, so two jobs on the same model sketch with the same map, and the dot product of two
sketches estimates the dot product of the gradients (error about 1 / sqrt(BUCKETS) in the cosine).
"""

import logging
import os
import zlib
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_GRAD_SKETCH"
BUCKETS = 4096
_CHUNK = 1 << 26


def _count_sketch(grad: torch.Tensor, name: str) -> torch.Tensor:
    flat = grad.detach().reshape(-1).float()
    if flat.numel() <= BUCKETS:
        return flat.cpu().clone()
    generator = torch.Generator(device=flat.device)
    generator.manual_seed(zlib.crc32(name.encode()))
    sketch = torch.zeros(BUCKETS, device=flat.device, dtype=torch.float64)
    for start in range(0, flat.numel(), _CHUNK):
        chunk = flat[start : start + _CHUNK]
        bucket = torch.randint(0, BUCKETS, (chunk.numel(),), generator=generator, device=flat.device)
        sign = torch.randint(0, 2, (chunk.numel(),), generator=generator, device=flat.device) * 2 - 1
        sketch.index_add_(0, bucket, (chunk * sign).double())
    return sketch.float().cpu()


@torch.no_grad()
def maybe_save_grad_sketch(model_chunks, dump_details: str | None, rollout_id: int, step_id: int) -> None:
    """Write this rank's gradient sketches when MILES_GRAD_SKETCH=1."""
    if os.environ.get(ENV_VAR, "") != "1":
        return
    assert dump_details, f"{ENV_VAR}=1 needs --dump-details"
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    out = {}
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            grad = getattr(param, "main_grad", None)  # config-access-exempt: DDP attaches main_grad at runtime
            if grad is None:
                grad = param.grad
            if grad is None:
                continue
            out[name] = {
                "norm2": grad.detach().double().pow(2).sum().item(),
                "numel": grad.numel(),
                "sketch": _count_sketch(grad, name),
            }
    path = Path(dump_details) / "grad_sketch" / f"rollout_{rollout_id}_step_{step_id}_rank_{rank}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    total = sum(v["norm2"] for v in out.values()) ** 0.5
    logger.info("grad sketch saved: %s tensors=%d local grad norm=%.4e", path, len(out), total)
