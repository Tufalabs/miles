"""A trainer-side adapter that tracks the sampler (experiment, LLMT-83).

MILES_SAMPLER_ADAPTER names a group of parameters (entries as MILES_FREEZE_PARAMS: substrings of parameter
names, and `qkv_rows:<letters of q, k, v>`). The group then has two sets of values. The model's own are the
ones the sampler has: the optimizer, the weight sync and the checkpoints see them. The trainer adds an offset
to them for its forward passes. Each training step:

  1. the offset is added to the group;
  2. a pass whose only loss is the divergence between trainer and sampler over the sampler's returned
     candidates (loss/hub/score_centering_head_kl.py) gives the offset's gradient;
  3. the policy pass runs as usual, through the adapted weights; its gradient goes to the model's own values
     as for any other weight (MILES_SAMPLER_ADAPTER_FREEZE=1 zeroes it for the group, which then stays where
     the run started on the sampler's side);
  4. the group goes back to the sampler's values and the offset takes its step, offset <- offset - lr *
     gradient (MILES_SAMPLER_ADAPTER_LR, default 0.02).

So the sampler never receives the offset and the policy loss never trains it. The offset is kept in fp32; the
model's bf16 values during the passes are the rounded sum of the sampler's bf16 values and the offset. It is
not saved: a resumed run starts it from zero. It doubles the training passes of a step.
"""

import logging
import os
from collections.abc import Callable, Sequence

import torch

from miles.backends.training_utils.loss.hub import score_centering_head_kl

from .weight_swap import _qkv_row_mask

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_SAMPLER_ADAPTER"
LR_ENV_VAR = "MILES_SAMPLER_ADAPTER_LR"
_ROWS_PREFIX = "qkv_rows:"
_ENTRIES = tuple(p for p in os.environ.get(ENV_VAR, "").split(",") if p)
_PATTERNS = tuple(p for p in _ENTRIES if not p.startswith(_ROWS_PREFIX))
_QKV_PARTS = "".join(p[len(_ROWS_PREFIX) :] for p in _ENTRIES if p.startswith(_ROWS_PREFIX))
_LR = float(os.environ.get(LR_ENV_VAR, "") or 0.02)
FREEZE_ENV_VAR = "MILES_SAMPLER_ADAPTER_FREEZE"
_FREEZE = os.environ.get(FREEZE_ENV_VAR, "") == "1"

# Per adapted parameter: the parameter, the index of its adapted rows (a slice for all of it), the sampler's
# values of those rows, the offset (fp32) and the step's gradient of the divergence on them.
_state: list[dict] = []
_ready = False


def enabled() -> bool:
    return bool(_ENTRIES)


def _grad_of(param: torch.Tensor) -> torch.Tensor:
    grad = getattr(param, "main_grad", None)  # config-access-exempt: DDP attaches main_grad at runtime
    return param.grad if grad is None else grad


@torch.no_grad()
def _collect(model_chunks) -> None:
    global _ready
    _ready = True
    whole: set[int] = set()
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if any(pattern in name for pattern in _PATTERNS):
                whole.add(id(param))
                _state.append({"name": name, "param": param, "rows": slice(None)})
        if not _QKV_PARTS:
            continue
        for name, module in chunk.named_modules():
            linear_qkv = getattr(module, "linear_qkv", None)
            if linear_qkv is None or not hasattr(module, "num_query_groups_per_partition"):
                continue
            if id(linear_qkv.weight) in whole:
                continue
            rows = _qkv_row_mask(module, linear_qkv.weight, _QKV_PARTS)
            label = f"{name}.linear_qkv.weight[{_QKV_PARTS}]"
            _state.append({"name": label, "param": linear_qkv.weight, "rows": rows})
    assert _state, f"{ENV_VAR}={_ENTRIES} names no parameter"
    elements = 0
    for entry in _state:
        entry["offset"] = torch.zeros_like(entry["param"].data[entry["rows"]], dtype=torch.float32)
        elements += entry["offset"].numel()
    logger.info(
        "sampler adapter: entries=%s tensors=%d elements=%.4e lr=%g freeze=%d (this rank)",
        _ENTRIES,
        len(_state),
        elements,
        _LR,
        int(_FREEZE),
    )


@torch.no_grad()
def _write(adapted: bool) -> None:
    """Put the adapted values in the weights, keeping the sampler's (the optimizer's last write) aside, or put
    the sampler's back."""
    for entry in _state:
        if adapted:
            entry["sampler"] = entry["param"].data[entry["rows"]].clone()
            values = (entry["sampler"].float() + entry["offset"]).to(entry["sampler"].dtype)
        else:
            values = entry.pop("sampler")
        entry["param"].data[entry["rows"]] = values


def measure(
    run_pass: Callable[[int], object],
    data_iterator: Sequence,
    model_chunks,
    zero_grads: Callable[[], None],
    num_microbatches: int,
) -> None:
    """Add the offset to the group and take the gradient of the divergence on it from a pass of its own; leaves
    the gradients zeroed, the data iterators where they were and the offset in the weights for the policy pass."""
    if not _ready:
        _collect(model_chunks)
    _write(adapted=True)
    offsets = [iterator.offset for iterator in data_iterator]
    with score_centering_head_kl.divergence_phase():
        run_pass(num_microbatches)
    assert score_centering_head_kl.phase_uses() > 0, f"{ENV_VAR} needs the score-centering loss"
    with torch.no_grad():
        for entry in _state:
            entry["gradient"] = _grad_of(entry["param"])[entry["rows"]].float().clone()
    zero_grads()
    for iterator, offset in zip(data_iterator, offsets, strict=True):
        iterator.offset = offset


@torch.no_grad()
def finish(rollout_id: int) -> None:
    """After the policy pass: the sampler's values back in the weights and the offset's step; with
    MILES_SAMPLER_ADAPTER_FREEZE no policy gradient for the group."""
    gradient2, offset2, policy2 = 0.0, 0.0, 0.0
    for entry in _state:
        grad = _grad_of(entry["param"])
        policy2 += float(grad[entry["rows"]].float().square().sum())
        if _FREEZE:
            grad[entry["rows"]] = 0
        gradient = entry.pop("gradient")
        gradient2 += float(gradient.square().sum())
        entry["offset"].add_(gradient, alpha=-_LR)
        offset2 += float(entry["offset"].square().sum())
    _write(adapted=False)
    logger.info(
        "sampler adapter rollout=%d divergence_grad_norm=%.4e offset_norm=%.4e group_policy_grad_norm=%.4e"
        " (this rank)",
        rollout_id,
        gradient2**0.5,
        offset2**0.5,
        policy2**0.5,
    )
