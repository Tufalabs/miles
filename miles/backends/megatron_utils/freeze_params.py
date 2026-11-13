"""Freeze groups of parameters by zeroing their gradients after the backward pass (experiment, LLMT-83).

MILES_FREEZE_PARAMS is a comma-separated list of substrings; a parameter whose name contains one of them gets
its gradient zeroed before the optimizer sees it. With plain SGD (no momentum, no weight decay) that keeps the
parameter at its starting value. The logged gradient norm is then the norm over the parameters left free.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_FREEZE_PARAMS"
_PATTERNS = tuple(p for p in os.environ.get(ENV_VAR, "").split(",") if p)
_logged = False


@torch.no_grad()
def maybe_zero_frozen_grads(model_chunks) -> None:
    """Zero the gradients of the parameters MILES_FREEZE_PARAMS names."""
    global _logged
    if not _PATTERNS:
        return
    frozen, free, frozen_elements, free_elements = 0, 0, 0, 0
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            grad = getattr(param, "main_grad", None)  # config-access-exempt: DDP attaches main_grad at runtime
            if grad is None:
                grad = param.grad
            if grad is None:
                continue
            if any(pattern in name for pattern in _PATTERNS):
                grad.zero_()
                frozen += 1
                frozen_elements += grad.numel()
            else:
                free += 1
                free_elements += grad.numel()
    if not _logged:
        _logged = True
        logger.info(
            "frozen parameters: patterns=%s frozen tensors=%d (%.3e elements) free tensors=%d (%.3e elements)",
            _PATTERNS,
            frozen,
            frozen_elements,
            free,
            free_elements,
        )
        assert frozen > 0, f"{ENV_VAR}={_PATTERNS} matches no parameter"
