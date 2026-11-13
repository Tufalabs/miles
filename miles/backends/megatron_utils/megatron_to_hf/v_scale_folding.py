"""Send the sampler V projections scaled up and output projections scaled down by the same factor (LLMT-83).

Attention is linear in V, so `v_proj * c` with `o_proj / c` computes the same function. It changes the range in
which a quantized KV cache and an FP8 attention output see V: SGLang's NVFP4 cache stores one FP8 E4M3 scale
(amax / 6) per block of 16 values and runs Qwen3-MoE with a global scale of 1, so small V values put the block
scales in E4M3's subnormal range or at zero.

Enabled by the environment variable MILES_SAMPLER_V_SCALE_LOG2: one integer per layer, comma separated; layer i
is scaled by 2 ** k_i, which is exact in floating point. The trainer's weights are not changed.
"""

import os
import re

import torch

ENV_VAR = "MILES_SAMPLER_V_SCALE_LOG2"
_NAME = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(v_proj\.weight|v_proj\.bias|o_proj\.weight)$")


def _read_log2_scales() -> tuple[int, ...] | None:
    spec = os.environ.get(ENV_VAR, "").strip()
    return tuple(int(k) for k in spec.split(",")) if spec else None


_LOG2_SCALES = _read_log2_scales()


def fold_v_scale(named_tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    """Scale each layer's v_proj by 2**k and its o_proj by 2**-k; a no-op when the variable is unset."""
    if _LOG2_SCALES is None:
        return named_tensors
    out = []
    for name, tensor in named_tensors:
        match = _NAME.match(name)
        if match is not None:
            k = _LOG2_SCALES[int(match.group(1))]
            tensor = tensor * 2.0 ** (k if match.group(2).startswith("v_proj") else -k)
        out.append((name, tensor))
    return out
