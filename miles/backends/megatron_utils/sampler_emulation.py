"""Make the Megatron forward compute what a quantized SGLang attention computes (experiment, LLMT-83).

SGLang with `--kv-cache-dtype nvfp4` on SM100 (trtllm_mha, native FP4) quantizes K (after k_norm and RoPE) and V
to NVFP4 for every position, casts the query to FP8 E4M3 at scale 1, and writes the attention output as FP8
E4M3 at scale 1. This module wraps each layer's core attention to do the same in the trainer, with
straight-through gradients, so that the trainer's distribution matches the sampler's.

Enabled by the environment variable MILES_SAMPLER_EMULATION, a comma-separated list of:
  kv_nvfp4   K and V through NVFP4: blocks of 16 along head_dim, an E4M3 scale of amax / 6 per block, values
             rounded to E2M1; a block whose scale rounds to zero is zeroed, as in FlashInfer's kernel
  q_fp8      the query cast to FP8 E4M3 and back
  out_fp8    the attention output cast to FP8 E4M3 and back
CUDA only (Triton kernel).
"""

import logging
import os

import torch

from miles.kernels.quant.fake_quant import COMPRESSED_KV_BLOCK_SIZE, fake_quant_compressed_kv

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_SAMPLER_EMULATION"
PARTS = ("kv_nvfp4", "q_fp8", "out_fp8")
# FlashInfer rounds amax / 6 to E4M3 with no floor: below half of the smallest subnormal (2**-9) it is zero.
_ZERO_BLOCK_AMAX = 6.0 * 2.0**-10


class _Fp8StraightThrough(torch.autograd.Function):
    """The value cast to FP8 E4M3 and back, exactly; the gradient passes unchanged."""

    @staticmethod
    def forward(ctx, x):
        return x.to(torch.float8_e4m3fn).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _fp8_straight_through(x: torch.Tensor) -> torch.Tensor:
    return _Fp8StraightThrough.apply(x)


def _nvfp4_kv(x: torch.Tensor) -> torch.Tensor:
    quantized = fake_quant_compressed_kv(x)
    blocks = x.detach().reshape(*x.shape[:-1], -1, COMPRESSED_KV_BLOCK_SIZE)
    keep = (blocks.abs().amax(dim=-1, keepdim=True) > _ZERO_BLOCK_AMAX).expand_as(blocks).reshape(x.shape)
    return quantized * keep.to(quantized.dtype)


def _wrap_core_attention(core_attention: torch.nn.Module, parts: frozenset[str]) -> None:
    original_forward = core_attention.forward

    def forward(query, key, value, *args, **kwargs):
        if "kv_nvfp4" in parts:
            key, value = _nvfp4_kv(key), _nvfp4_kv(value)
        if "q_fp8" in parts:
            query = _fp8_straight_through(query)
        output = original_forward(query, key, value, *args, **kwargs)
        if "out_fp8" in parts:
            output = _fp8_straight_through(output)
        return output

    core_attention.forward = forward


def maybe_install_sampler_emulation(model_chunks) -> None:
    """Wrap every layer's core attention when MILES_SAMPLER_EMULATION is set."""
    spec = os.environ.get(ENV_VAR, "").strip()
    if not spec:
        return
    parts = frozenset(part.strip() for part in spec.split(","))
    unknown = parts - set(PARTS)
    assert not unknown, f"{ENV_VAR}: unknown parts {sorted(unknown)}, expected a subset of {PARTS}"
    wrapped = 0
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if name.endswith("self_attention") and hasattr(module, "core_attention"):
                _wrap_core_attention(module.core_attention, parts)
                wrapped += 1
    assert wrapped > 0, f"{ENV_VAR} is set but no self_attention.core_attention module was found"
    logger.info("sampler emulation installed: parts=%s layers=%d", sorted(parts), wrapped)
