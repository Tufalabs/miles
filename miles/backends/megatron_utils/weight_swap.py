"""Put the base model's weights back into groups of parameters of a loaded checkpoint (experiment, LLMT-83).

MILES_WEIGHT_SWAP is a comma-separated list of entries, as MILES_FREEZE_PARAMS (freeze_params.py):

- a substring: a parameter whose name contains it takes the base model's values;
- `qkv_rows:<letters of q, k, v>`: the rows of every attention layer's fused `linear_qkv.weight` that produce
  the named parts take the base model's values.

The base model is the run's `--ref-load` checkpoint. Everything else keeps the values of the checkpoint the run
loaded. MILES_WEIGHT_SWAP_INVERT=1 inverts the choice: the named groups keep the checkpoint's values and
everything else takes the base model's. The swap happens once, after the checkpoint is loaded and before
anything reads the weights; it is for probes at learning rate 0 that ask where in the weights a property of a
checkpoint sits.
"""

import logging
import os

import torch

from miles.utils.argparse_utils import inplace_modify_args

from .checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_WEIGHT_SWAP"
INVERT_ENV_VAR = "MILES_WEIGHT_SWAP_INVERT"
_ROWS_PREFIX = "qkv_rows:"
_ENTRIES = tuple(p for p in os.environ.get(ENV_VAR, "").split(",") if p)
_PATTERNS = tuple(p for p in _ENTRIES if not p.startswith(_ROWS_PREFIX))
_QKV_PARTS = "".join(p[len(_ROWS_PREFIX) :] for p in _ENTRIES if p.startswith(_ROWS_PREFIX))
_INVERT = os.environ.get(INVERT_ENV_VAR, "") == "1"


def enabled() -> bool:
    return bool(_ENTRIES)


def _qkv_row_mask(attention: torch.nn.Module, weight: torch.Tensor, parts: str) -> torch.Tensor:
    """True for the rows of a `linear_qkv.weight` that produce the parts in `parts` (layout: freeze_params.py)."""
    head_dim = attention.hidden_size_per_attention_head
    groups = attention.num_query_groups_per_partition
    q_rows = attention.num_attention_heads_per_partition // groups * head_dim
    group_rows = q_rows + 2 * head_dim
    assert weight.shape[0] == groups * group_rows, f"unexpected linear_qkv layout: {tuple(weight.shape)}"
    spans = {"q": (0, q_rows), "k": (q_rows, q_rows + head_dim), "v": (q_rows + head_dim, group_rows)}
    mask = torch.zeros(groups, group_rows, dtype=torch.bool, device=weight.device)
    for part in parts:
        start, end = spans[part]
        mask[:, start:end] = True
    return mask.reshape(-1)


def _base_masks(model_chunks) -> dict[int, torch.Tensor | bool]:
    """For every parameter (by id): True when it takes the base values, False when it keeps the checkpoint's,
    or a mask over the rows of a `linear_qkv.weight`."""
    masks: dict[int, torch.Tensor | bool] = {}
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            masks[id(param)] = any(pattern in name for pattern in _PATTERNS) != _INVERT
        if not _QKV_PARTS:
            continue
        for module in chunk.modules():
            linear_qkv = getattr(module, "linear_qkv", None)
            if linear_qkv is None or not hasattr(module, "num_query_groups_per_partition"):
                continue
            weight = linear_qkv.weight
            if masks[id(weight)] != _INVERT:  # the whole tensor is named by a substring
                continue
            rows = _qkv_row_mask(module, weight, _QKV_PARTS)
            masks[id(weight)] = ~rows if _INVERT else rows
    return masks


@torch.no_grad()
def swap_in_base_weights(args, model_chunks) -> None:
    """Replace the groups MILES_WEIGHT_SWAP names by the values of the `--ref-load` checkpoint."""
    assert args.ref_load, f"{ENV_VAR} needs --ref-load, the base model's checkpoint"
    masks = _base_masks(model_chunks)
    params = [(name, param) for chunk in model_chunks for name, param in chunk.named_parameters()]
    loaded = [param.detach().clone() for _, param in params]
    with args.backend.mutable(), inplace_modify_args(args.backend, dict(load=args.ref_load, no_load_optim=True, no_load_rng=True, finetune=True, ckpt_step=None)):
        load_checkpoint(model_chunks, None, None, checkpointing_context={}, args=args, skip_load_to_model_and_opt=False)
    # elements and squared norms: [from the base, kept from the checkpoint], and the change the base part undoes
    elements, moved2, base2 = [0, 0], 0.0, 0.0
    for (_name, param), checkpoint_value in zip(params, loaded, strict=True):
        mask = masks[id(param)]
        if mask is True:
            elements[0] += param.numel()
            moved2 += float((checkpoint_value.float() - param.float()).square().sum())
            base2 += float(param.float().square().sum())
        elif mask is False:
            elements[1] += param.numel()
            param.copy_(checkpoint_value)
        else:
            row_elements = param[0].numel()
            elements[0] += int(mask.sum()) * row_elements
            elements[1] += int((~mask).sum()) * row_elements
            moved2 += float((checkpoint_value[mask].float() - param[mask].float()).square().sum())
            base2 += float(param[mask].float().square().sum())
            param[~mask] = checkpoint_value[~mask]
    logger.info(
        "weight swap: entries=%s invert=%d base elements=%.4e checkpoint elements=%.4e |checkpoint - base| / |base| over the base part=%.4e (this rank)",
        _ENTRIES,
        int(_INVERT),
        elements[0],
        elements[1],
        (moved2 / max(base2, 1e-30)) ** 0.5,
    )
    assert elements[0] > 0 and elements[1] > 0, f"{ENV_VAR}={_ENTRIES} names nothing, or everything"
