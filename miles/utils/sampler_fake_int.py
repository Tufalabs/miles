"""Integer activations and an integer KV cache in an unquantized (bf16) SGLang rollout engine (LLMT-83).

With MILES_SAMPLER_FAKE_INT_WEIGHTS (processors/quantizer_fake_int.py, the weights on their way to the engine)
this emulates a sampler with integer weights, activations and KV cache, the hardest setting of the score
centering paper (INT8 W/A, INT4 KV), without a new serving format. The functions here are called from SGLang's
Qwen3-MoE model through SGLang's source patcher (DUMPER_SOURCE_PATCHER_CONFIG; the patch file is
sglang_fake_int.yaml in rave's artifacts folder of LLMT-83). Without that config SGLang's code is untouched.

MILES_SAMPLER_FAKE_INT_ACT=<bits>  the inputs of the matrices that the weight rounding covers are rounded to a
    symmetric integer grid of 2 ** bits - 1 levels, per token (scale = the largest magnitude over the last
    dimension / (2 ** (bits - 1) - 1)): the input of the q/k/v projection, of the attention output projection, of
    the experts (their gate and up projections) and of the experts' down projection. The router reads the
    unrounded input, as it keeps its weights.
MILES_SAMPLER_FAKE_INT_KV=<bits>   the key and value of every token are rounded the same way per head, after the
    q/k norm and RoPE and before the KV cache and the attention kernel, so that everything attention reads, the
    current token's key and value included, is on the grid. The engine keeps a bf16 KV cache; its fused write of
    K and V in the RoPE kernel is switched off by the patch.

The rounding is done in float32 and stored in the tensor's dtype: bf16 keeps a value to 2 ** -8 of itself, which
is up to half a grid step at the top of an 8-bit grid (measured 0.42) and under a tenth of one from 6 bits down.
Nothing here reads a tensor's values on the host, so the calls can be captured in a CUDA graph.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ACT_ENV_VAR = "MILES_SAMPLER_FAKE_INT_ACT"
KV_ENV_VAR = "MILES_SAMPLER_FAKE_INT_KV"
ACT_BITS = int(os.environ.get(ACT_ENV_VAR, "") or 0)
KV_BITS = int(os.environ.get(KV_ENV_VAR, "") or 0)
assert ACT_BITS == 0 or 2 <= ACT_BITS <= 8, f"{ACT_ENV_VAR}={ACT_BITS}: 2 to 8 bits, or unset"
assert KV_BITS == 0 or 2 <= KV_BITS <= 8, f"{KV_ENV_VAR}={KV_BITS}: 2 to 8 bits, or unset"
_seen: set[str] = set()
logger.warning(
    "sampler fake int: activations %d bits, KV cache %d bits (0: off), pid %d", ACT_BITS, KV_BITS, os.getpid()
)


def round_last_dim(x: torch.Tensor, bits: int) -> torch.Tensor:
    """x on a symmetric integer grid of 2 ** bits - 1 levels with one scale per vector along the last dimension."""
    top = 2 ** (bits - 1) - 1
    values = x.float()
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / top
    return (torch.round(values / scale).clamp_(-top, top) * scale).to(x.dtype)


def _note(site: str, x: torch.Tensor) -> None:
    if site not in _seen:
        _seen.add(site)
        logger.warning("sampler fake int: first call at %s, shape %s, %s", site, tuple(x.shape), x.dtype)


def act(x: torch.Tensor, site: str) -> torch.Tensor:
    """The input of a quantized matrix, [..., features], rounded per token."""
    if not ACT_BITS or x.numel() == 0:
        return x
    _note(site, x)
    return round_last_dim(x, ACT_BITS)


def kv(attention, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """k and v of one layer, each [tokens, heads * head_dim], rounded per token and head."""
    if not KV_BITS or k.numel() == 0:
        return k, v
    _note("kv", k)
    head_dim = attention.head_dim
    return (
        round_last_dim(k.reshape(-1, head_dim), KV_BITS).reshape(k.shape),
        round_last_dim(v.reshape(-1, head_dim), KV_BITS).reshape(v.shape),
    )
