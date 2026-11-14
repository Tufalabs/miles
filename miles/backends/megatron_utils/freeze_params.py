"""Freeze groups of parameters by zeroing their gradients after the backward pass (experiment, LLMT-83).

MILES_FREEZE_PARAMS is a comma-separated list of entries:

- a substring: a parameter whose name contains it gets its gradient zeroed before the optimizer sees it;
- `qkv_rows:<letters of q, k, v>`: the rows of every attention layer's fused `linear_qkv.weight` that produce
  the named parts get their gradient zeroed (Megatron lays the rows out per query group as q heads, k, v).

With plain SGD (no momentum, no weight decay) that keeps the frozen values at their start. The logged gradient
norm is then the norm over what is left free.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_FREEZE_PARAMS"
_ROWS_PREFIX = "qkv_rows:"
_ENTRIES = tuple(p for p in os.environ.get(ENV_VAR, "").split(",") if p)
_PATTERNS = tuple(p for p in _ENTRIES if not p.startswith(_ROWS_PREFIX))
_QKV_PARTS = "".join(p[len(_ROWS_PREFIX) :] for p in _ENTRIES if p.startswith(_ROWS_PREFIX))
_logged = False


def _grad_of(param: torch.Tensor) -> torch.Tensor | None:
    grad = getattr(param, "main_grad", None)  # config-access-exempt: DDP attaches main_grad at runtime
    return param.grad if grad is None else grad


def zero_qkv_rows(attention: torch.nn.Module, grad: torch.Tensor, parts: str) -> int:
    """Zero the rows of a `linear_qkv.weight` gradient that produce the parts in `parts`; the number of rows."""
    head_dim = attention.hidden_size_per_attention_head
    groups = attention.num_query_groups_per_partition
    q_rows = attention.num_attention_heads_per_partition // groups * head_dim
    group_rows = q_rows + 2 * head_dim
    assert grad.shape[0] == groups * group_rows, f"unexpected linear_qkv layout: {tuple(grad.shape)}"
    spans = {"q": (0, q_rows), "k": (q_rows, q_rows + head_dim), "v": (q_rows + head_dim, group_rows)}
    by_group = grad.view(groups, group_rows, -1)
    zeroed = 0
    for part in parts:
        start, end = spans[part]
        by_group[:, start:end].zero_()
        zeroed += groups * (end - start)
    return zeroed


@torch.no_grad()
def maybe_zero_frozen_grads(model_chunks) -> None:
    """Zero the gradients MILES_FREEZE_PARAMS names."""
    global _logged
    if not _ENTRIES:
        return
    frozen, free, frozen_elements, free_elements, qkv_layers, qkv_rows = 0, 0, 0, 0, 0, 0
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            grad = _grad_of(param)
            if grad is None:
                continue
            if any(pattern in name for pattern in _PATTERNS):
                grad.zero_()
                frozen += 1
                frozen_elements += grad.numel()
            else:
                free += 1
                free_elements += grad.numel()
        if not _QKV_PARTS:
            continue
        for module in chunk.modules():
            linear_qkv = getattr(module, "linear_qkv", None)
            if linear_qkv is None or not hasattr(module, "num_query_groups_per_partition"):
                continue
            grad = _grad_of(linear_qkv.weight)
            if grad is None:
                continue
            bias = getattr(linear_qkv, "bias", None)  # Transformer Engine keeps an empty tensor when there is none
            assert bias is None or bias.numel() == 0, "qkv_rows does not handle a QKV bias"
            qkv_layers += 1
            qkv_rows += zero_qkv_rows(module, grad, _QKV_PARTS)
    if not _logged:
        _logged = True
        logger.info(
            "frozen parameters: patterns=%s frozen tensors=%d (%.3e elements) free tensors=%d (%.3e elements) qkv parts=%r: %d rows in %d layers",
            _PATTERNS,
            frozen,
            frozen_elements,
            free,
            free_elements,
            _QKV_PARTS,
            qkv_rows,
            qkv_layers,
        )
        assert frozen > 0 or qkv_rows > 0, f"{ENV_VAR}={_ENTRIES} matches no parameter"
