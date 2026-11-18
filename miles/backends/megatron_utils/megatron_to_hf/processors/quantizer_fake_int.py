"""A harsher sampler without a new serving format: the weights sent to an unquantized (bf16) rollout engine are
rounded to an integer grid, or get a fixed noise, on their way out of the trainer. The trainer's own weights are
untouched, so the two sides disagree by exactly this change (LLMT-83).

MILES_SAMPLER_FAKE_INT_WEIGHTS=<bits> rounds every row of the attention projections (q, k, v, o) and of the
routed and shared experts' matrices to a symmetric grid of 2 ** bits - 1 levels, scale = the row's largest
magnitude / (2 ** (bits - 1) - 1). These are the matrices the FP8 path quantizes; the router, the norms, the
embeddings and the output head keep their values. The rounded values are stored in the weights' own dtype: in
bf16 a level times the scale is exact to 2 ** -9 of the value, a quarter of a grid step at 8 bits and less
than a tenth of one from 6 bits down.

MILES_SAMPLER_WEIGHT_NOISE=<s> adds to the same matrices a Gaussian offset of standard deviation s times the
matrix's root mean square, drawn from the weight's name, so that it is the same offset at every sync.

Both apply only when the rollout checkpoint has no quantization config, and in that order when both are set.
Everything that goes through this converter gets the change, the weights saved by --save-hf included."""

import os
import re
import zlib

import torch

BITS_ENV_VAR = "MILES_SAMPLER_FAKE_INT_WEIGHTS"
NOISE_ENV_VAR = "MILES_SAMPLER_WEIGHT_NOISE"
_BITS = int(os.environ.get(BITS_ENV_VAR, "") or 0)
_NOISE = float(os.environ.get(NOISE_ENV_VAR, "") or 0.0)
assert _BITS == 0 or 2 <= _BITS <= 8, f"{BITS_ENV_VAR}={_BITS}: 2 to 8 bits, or unset"
assert _NOISE >= 0, f"{NOISE_ENV_VAR}={_NOISE}: a relative standard deviation"
_MATRIX = re.compile(
    r"\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(experts\.\d+|shared_expert)\.(gate_proj|up_proj|down_proj))"
    r"\.weight$"
)
_reported = set()


def enabled() -> bool:
    return _BITS > 0 or _NOISE > 0


def round_rows(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Each row of a matrix on a symmetric integer grid of 2 ** bits - 1 levels, in the matrix's dtype."""
    top = 2 ** (bits - 1) - 1
    values = weight.float()
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / top
    return (torch.round(values / scale).clamp_(-top, top) * scale).to(weight.dtype)


def add_noise(name: str, weight: torch.Tensor, relative: float) -> torch.Tensor:
    """The matrix plus a Gaussian offset that depends only on its name, shape and `relative`."""
    generator = torch.Generator(device=weight.device).manual_seed(zlib.crc32(name.encode()))
    noise = torch.randn(weight.shape, generator=generator, dtype=torch.float32, device=weight.device)
    values = weight.float()
    return (values + noise * (relative * values.square().mean().sqrt())).to(weight.dtype)


def change_params(converted_named_params):
    changed = []
    for name, weight in converted_named_params:
        if weight.dim() == 2 and _MATRIX.search(name):
            before = weight
            if _BITS:
                weight = round_rows(weight, _BITS)
            if _NOISE:
                weight = add_noise(name, weight, _NOISE)
            kind = _MATRIX.search(name).group(1).split(".")[-1]
            if kind not in _reported:
                # once per kind of matrix and process: the size of the change that the sampler gets
                _reported.add(kind)
                error = (weight.float() - before.float()).square().mean().sqrt()
                size = before.float().square().mean().sqrt()
                print(
                    f"sampler weight change: {name} bits={_BITS} noise={_NOISE} relative_rms_change="
                    f"{float(error / size):.4f} levels_in_first_row={int(weight[0].float().unique().numel())}",
                    flush=True,
                )
        changed.append((name, weight))
    return changed


def change_stacked_params(converted_named_params):
    """change_params for tensors that hold the same matrix of several experts stacked along dim 0, each under the
    first expert's name. Rounding works on rows, so the stacked tensor is rounded in one call. None with noise on:
    the offset is drawn per matrix from its name."""
    if _NOISE:
        return None
    changed = []
    for name, weight in converted_named_params:
        if weight.dim() == 3 and _MATRIX.search(name):
            weight = round_rows(weight, _BITS)
        changed.append((name, weight))
    return changed
