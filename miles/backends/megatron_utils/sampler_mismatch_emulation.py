"""Emulate a sampler's weight and activation quantization in the trainer, rollout by rollout (LLMT-83 probe).

MILES_MISMATCH_EMULATION=<rollout>:<part>[+<part>...],...  parts:
  w_fp8   the attention (linear_qkv, linear_proj) and routed-expert (linear_fc1, linear_fc2) weights through
          SGLang's blockwise FP8 as miles' weight push makes it: blocks of 128 x 128, scale = amax / 448 (amax
          floored at 1e-4), values cast to E4M3. The scale is a power of two (UE8M0, rounded up) where the push
          would requantize for DeepGEMM (the attention matrices when SGLang's
          should_deepgemm_weight_requant_ue8m0 says so), else fp32. The fused q/k/v and gate/up rows keep their
          128-row blocks, since every head (128 rows) and the 768-row gate and up halves are whole blocks.
  w_int8, w_int4  the same matrices rounded per row to a symmetric 8- or 4-bit grid
          (processors/quantizer_fake_int.round_rows, what MILES_SAMPLER_FAKE_INT_WEIGHTS sends a bf16 engine).
  a_int8  the inputs of linear_proj and of the experts' linear_fc1 and linear_fc2 rounded per token to 8 bits
          (sampler_fake_int.round_last_dim), straight-through. The q/k/v projection's input is normed inside the
          fused layer-norm linear and is not rounded here.
Weights: before each train step the targeted parameters are set to the quantized originals of the step's part,
or back to the originals on a step without one. The originals are copied on the GPU at the first use; that holds
only while the policy does not train (learning rate 0), which is what this is for. Rollouts not listed get no
part. K and V emulation is MILES_KV_SUBSTITUTE's (emul_nvfp4, emul_int4g64). With
MILES_SAMPLER_ADAPTER_FORWARD_ONLY=1 no optimizer step writes the weights back, so they are put in place only
when the step's weight part changes.
"""

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_MISMATCH_EMULATION"
_KEEP = os.environ.get("MILES_SAMPLER_ADAPTER_FORWARD_ONLY", "") == "1"
PARTS = ("w_fp8", "w_int8", "w_int4", "a_int8")
_WEIGHT = re.compile(r"(self_attention\.linear_(qkv|proj)\.weight|mlp\.experts\.linear_fc[12]\.weight\d*)$")
_ACT_MODULE = re.compile(r"(self_attention\.linear_proj|mlp\.experts\.linear_fc[12])$")

_schedule: dict[int, frozenset[str]] = {}
_params: list[tuple[str, torch.nn.Parameter]] = []
_originals: list[torch.Tensor] = []
_state = {"weights": None, "act": False, "ue8m0_attention": None}


def enabled() -> bool:
    return bool(_schedule)


def _round_rows(weight: torch.Tensor, bits: int) -> torch.Tensor:
    top = 2 ** (bits - 1) - 1
    values = weight.float()
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / top
    return (torch.round(values / scale).clamp_(-top, top) * scale).to(weight.dtype)


def _fp8_blockwise(weight: torch.Tensor, ue8m0: bool) -> torch.Tensor:
    rows, cols = weight.shape
    assert rows % 128 == 0 and cols % 128 == 0, f"blockwise FP8 needs multiples of 128, got {tuple(weight.shape)}"
    blocks = weight.float().reshape(rows // 128, 128, cols // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4) / 448.0
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return quantized.reshape(rows, cols).to(weight.dtype)


def _ue8m0_attention() -> bool:
    if _state["ue8m0_attention"] is None:
        try:
            from miles.backends.megatron_utils.sglang import should_deepgemm_weight_requant_ue8m0
        except Exception:  # sglang not importable here
            should_deepgemm_weight_requant_ue8m0 = None
        _state["ue8m0_attention"] = bool(
            should_deepgemm_weight_requant_ue8m0
            and should_deepgemm_weight_requant_ue8m0(weight_block_size=[128, 128], grouped_moe=False)
        )
        logger.info("mismatch emulation: FP8 attention scales %s, expert scales fp32", "UE8M0" if _state["ue8m0_attention"] else "fp32")
    return _state["ue8m0_attention"]


def _quantize(name: str, weight: torch.Tensor, part: str) -> torch.Tensor:
    if part == "w_fp8":
        return _fp8_blockwise(weight, ue8m0=".experts." not in name and _ue8m0_attention())
    return _round_rows(weight, 8 if part == "w_int8" else 4)


@torch.no_grad()
def before_train_step(rollout_id: int) -> None:
    """Put the step's weights in place (quantized, or the originals) before any pass of the step."""
    if not _schedule:
        return
    parts = _schedule.get(rollout_id, frozenset())
    weights = next((part for part in parts if part.startswith("w_")), None)
    _state["act"] = "a_int8" in parts
    if weights is None and _state["weights"] is None:
        return
    if _KEEP and weights == _state["weights"]:
        return
    if not _originals:
        _originals.extend(param.detach().clone() for _, param in _params)
    for (name, param), original in zip(_params, _originals, strict=True):
        param.data.copy_(_quantize(name, original, weights) if weights else original)
    if weights != _state["weights"]:
        logger.info("mismatch emulation rollout=%d: weights %s, activations %s", rollout_id, weights or "original", "int8" if _state["act"] else "as is")
    _state["weights"] = weights


def _act_hook(module, args):
    if not _state["act"] or not args or not torch.is_tensor(args[0]):
        return None
    x = args[0]
    top = 127
    values = x.float()
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / top
    rounded = (torch.round(values / scale).clamp_(-top, top) * scale).to(x.dtype)
    return (x + (rounded - x).detach(), *args[1:])


def maybe_install(model_chunks) -> None:
    spec = os.environ.get(ENV_VAR, "").strip()
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        rollout, parts = entry.split(":")
        chosen = frozenset(filter(None, parts.split("+")))
        assert chosen <= set(PARTS), f"{ENV_VAR}: {entry}: parts are {PARTS}"
        assert sum(part.startswith("w_") for part in chosen) <= 1, f"{ENV_VAR}: {entry}: one weight part at most"
        _schedule[int(rollout)] = chosen
    if not _schedule:
        return
    hooks = 0
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if _WEIGHT.search(name):
                assert param.dim() == 2, f"{name}: expected a matrix, got {tuple(param.shape)}"
                _params.append((name, param))
        for name, module in chunk.named_modules():
            if _ACT_MODULE.search(name):
                module.register_forward_pre_hook(_act_hook)
                hooks += 1
    assert _params, f"{ENV_VAR} is set but no attention or expert matrix was found"
    logger.info("mismatch emulation installed: %d matrices, %d activation hooks, schedule %s", len(_params), hooks, {k: sorted(v) for k, v in _schedule.items()})
