"""Make the Megatron forward compute what a quantized SGLang attention computes (experiment, LLMT-83).

SGLang with `--kv-cache-dtype nvfp4` on SM100 (trtllm_mha, native FP4) quantizes K (after k_norm and RoPE) and V
to NVFP4 for every position, casts the query to FP8 E4M3 at scale 1, and writes the attention output as FP8
E4M3 at scale 1. This module wraps each layer's core attention to do the same in the trainer, with
straight-through gradients, so that the trainer's distribution matches the sampler's.

Enabled by the environment variable MILES_SAMPLER_EMULATION, a comma-separated list of:
  kv_nvfp4   K and V through NVFP4: blocks of 16 along head_dim, an E4M3 scale of amax / 6 per block, values
             rounded to E2M1; a block whose scale rounds to zero is zeroed, as in FlashInfer's kernel. In plain
             PyTorch, with the scale rounded to nearest E4M3
  k_nvfp4, v_nvfp4  only K, only V
  k_nvfp4_rotated   K through NVFP4 in a Hadamard-rotated basis of head_dim (what a rotating sampler would store)
  k_nvfp4_equalized K through NVFP4 after dividing each coordinate by its RMS (measured on the first batch, per
                    layer, shared within a RoPE pair), multiplied back after
  kv_nvfp4_kernel   the same through miles' Triton kernel (its rounding of the scale is not verified)
  q_fp8      the query cast to FP8 E4M3 and back
  out_fp8    the attention output cast to FP8 E4M3 and back
MILES_SAMPLER_EMULATION_STATS=1 logs, once per layer, where K, V, the query and the attention output sit
against the formats' ranges (it can be set without any part, to measure only).
MILES_SAMPLER_EMULATION_SCORE_STATS=1 logs, once per layer, what K's NVFP4 rounding does to the attention
scores and patterns of the first sequences of the first batch (it can be set without any part).
CUDA only (Triton kernel).
"""

import logging
import os

import torch

from miles.backends.megatron_utils.megatron_to_hf.v_scale_folding import _LOG2_SCALES as _V_LOG2_SCALES
from miles.kernels.quant.fake_quant import COMPRESSED_KV_BLOCK_SIZE, fake_quant_compressed_kv

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_SAMPLER_EMULATION"
STATS_ENV_VAR = "MILES_SAMPLER_EMULATION_STATS"
SCORE_STATS_ENV_VAR = "MILES_SAMPLER_EMULATION_SCORE_STATS"
_SCORE_STATS_SEQUENCES = 8
_E4M3_MIN_NORMAL = 2.0**-6
_E4M3_MIN_SUBNORMAL = 2.0**-9
PARTS = (
    "kv_nvfp4",
    "k_nvfp4",
    "k_nvfp4_rotated",
    "k_nvfp4_equalized",
    "v_nvfp4",
    "kv_nvfp4_kernel",
    "q_fp8",
    "out_fp8",
)
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


def _nvfp4_round_trip(x: torch.Tensor) -> torch.Tensor:
    """FlashInfer's `nvfp4_kv_quantize` at a global scale of 1, and the read back, in plain PyTorch.

    Per block of 16 along the last dimension: the scale is amax / 6 rounded to nearest E4M3 (zero when it
    rounds to zero, and then the block reads back as zeros); the values over that rounded scale are rounded to
    nearest-even E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6), saturating at 6.
    """
    blocks = x.float().reshape(-1, COMPRESSED_KV_BLOCK_SIZE)
    scale = (blocks.abs().amax(dim=-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn).float()
    nonzero = scale > 0
    scaled = (blocks / torch.where(nonzero, scale, torch.ones_like(scale))).clamp(-6.0, 6.0)
    magnitude = scaled.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    rounded = torch.round(magnitude / step) * step * torch.sign(scaled)
    return torch.where(nonzero, rounded * scale, torch.zeros_like(rounded)).reshape(x.shape).to(x.dtype)


class _Nvfp4StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _nvfp4_round_trip(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _hadamard(n: int, device, dtype) -> torch.Tensor:
    """The orthonormal Sylvester Hadamard matrix of size n (a power of two)."""
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    assert h.shape[0] == n
    return (h / n**0.5).to(dtype)


def _nvfp4_rotated(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 in a rotated basis of head_dim: rotate by a Hadamard matrix with fixed random signs, quantize,
    rotate back. What a sampler would store if it rotated K (and the query) before the cache."""
    n = x.shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(0)
    signs = (torch.randint(0, 2, (n,), generator=generator) * 2 - 1).to(device=x.device, dtype=torch.float32)
    rotation = signs[:, None] * _hadamard(n, x.device, torch.float32)
    rotated = (x.float() @ rotation).to(x.dtype)
    return (_Nvfp4StraightThrough.apply(rotated).float() @ rotation.T).to(x.dtype)


def _pair_rms(x: torch.Tensor) -> torch.Tensor:
    """The RMS of each head_dim coordinate over tokens and heads, shared by the two coordinates of a RoPE pair
    (i and i + n/2), so that a per-coordinate scale built from it commutes with RoPE."""
    n = x.shape[-1]
    mean_square = x.detach().float().reshape(-1, n).pow(2).mean(dim=0)
    pair = 0.5 * (mean_square[: n // 2] + mean_square[n // 2 :])
    return torch.cat([pair, pair]).sqrt().clamp_min(1e-6)


def _nvfp4_equalized(x: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
    """NVFP4 on K divided by its per-coordinate RMS, multiplied back. What the sampler would store if the RMS
    were folded into k_norm's gain (and its inverse into q_norm's), which leaves q.k unchanged."""
    scaled = (x.float() / rms).to(x.dtype)
    return (_Nvfp4StraightThrough.apply(scaled).float() * rms).to(x.dtype)


def _nvfp4_kv(x: torch.Tensor, kernel: bool) -> torch.Tensor:
    if not kernel:
        return _Nvfp4StraightThrough.apply(x)
    quantized = fake_quant_compressed_kv(x)
    blocks = x.detach().reshape(*x.shape[:-1], -1, COMPRESSED_KV_BLOCK_SIZE)
    keep = (blocks.abs().amax(dim=-1, keepdim=True) > _ZERO_BLOCK_AMAX).expand_as(blocks).reshape(x.shape)
    return quantized * keep.to(quantized.dtype)


def _block_scale_stats(x: torch.Tensor) -> str:
    """Where the NVFP4 block scales (amax / 6) of a K or V tensor fall in E4M3."""
    blocks = x.detach().float().reshape(-1, COMPRESSED_KV_BLOCK_SIZE)
    scale = blocks.abs().amax(dim=-1) / 6.0
    quantiles = torch.quantile(scale[:: max(1, scale.numel() // 200_000)], torch.tensor([0.01, 0.1, 0.5, 0.9], device=scale.device))
    return f"scale q01={quantiles[0]:.2e} q10={quantiles[1]:.2e} q50={quantiles[2]:.2e} q90={quantiles[3]:.2e} zeroed={(scale < _E4M3_MIN_SUBNORMAL / 2).float().mean():.4f} subnormal={(scale < _E4M3_MIN_NORMAL).float().mean():.4f}"


def _element_stats(x: torch.Tensor) -> str:
    """Where the elements of a tensor that is cast to FP8 E4M3 at scale 1 fall."""
    mag = x.detach().float().abs().flatten()
    quantiles = torch.quantile(mag[:: max(1, mag.numel() // 200_000)], torch.tensor([0.1, 0.5, 0.9], device=mag.device))
    return f"abs q10={quantiles[0]:.2e} q50={quantiles[1]:.2e} q90={quantiles[2]:.2e} max={mag.max():.2e} to_zero={(mag < _E4M3_MIN_SUBNORMAL / 2).float().mean():.4f} subnormal={(mag < _E4M3_MIN_NORMAL).float().mean():.4f} above_448={(mag > 448).float().mean():.6f}"


@torch.no_grad()
def _score_stats(query: torch.Tensor, key: torch.Tensor, packed_seq_params) -> str:
    """What K's NVFP4 rounding does to the attention scores q.k / sqrt(head_dim) and to the attention patterns.

    Over the first sequences of a packed batch (query [tokens, heads, head_dim], key [tokens, groups,
    head_dim]) and every head, with a causal mask:
      rel_k_err   |K rounded - K| / |K| (RMS over elements)
      score_err   RMS of the score error over query-key pairs
      score_std   the spread of a query's scores over its keys (mean over queries)
      tv, tv_p99  total variation between the exact and the rounded attention pattern of a query: mean, 99th
                  percentile over queries
      kl          KL(exact pattern || rounded pattern), mean over queries
      entropy, p_max  of the exact pattern, mean over queries
    """
    bounds = packed_seq_params.cu_seqlens_host
    heads, groups, head_dim = query.shape[1], key.shape[1], query.shape[2]
    rounded_key = _nvfp4_round_trip(key)
    sums = dict.fromkeys(("score_err2", "pairs", "score_std", "tv", "kl", "entropy", "p_max", "queries"), 0.0)
    tvs = []
    sequences = min(_SCORE_STATS_SEQUENCES, len(bounds) - 1)
    for start, end in zip(bounds[:sequences], bounds[1 : sequences + 1], strict=True):
        length = end - start
        q = query[start:end].float().transpose(0, 1)  # [heads, length, head_dim]
        causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        patterns = []
        for k in (key, rounded_key):
            k = k[start:end].float().transpose(0, 1).repeat_interleave(heads // groups, dim=0)
            scores = q @ k.transpose(1, 2) / head_dim**0.5  # [heads, query, key]
            patterns.append((scores, scores.masked_fill(~causal, float("-inf")).log_softmax(dim=-1)))
        (scores, log_p), (rounded_scores, log_p_rounded) = patterns
        p = log_p.exp()
        count = causal.sum() * heads
        mean = (scores * causal).sum(-1) / causal.sum(-1)
        variance = (((scores - mean[..., None]) ** 2) * causal).sum(-1) / causal.sum(-1)
        tv = 0.5 * (p - log_p_rounded.exp()).abs().sum(-1)
        tvs.append(tv.flatten())
        sums["score_err2"] += (((rounded_scores - scores) ** 2) * causal).sum().item()
        sums["pairs"] += count.item()
        sums["score_std"] += variance.sqrt().sum().item()
        sums["tv"] += tv.sum().item()
        sums["kl"] += (p * (log_p - log_p_rounded).masked_fill(~causal, 0.0)).sum().item()
        sums["entropy"] += -(p * log_p.masked_fill(~causal, 0.0)).sum().item()
        sums["p_max"] += p.amax(-1).sum().item()
        sums["queries"] += heads * length
    tv_all = torch.cat(tvs)
    tv_p99 = torch.quantile(tv_all[:: max(1, tv_all.numel() // 1_000_000)], 0.99).item()
    rel_k_err = ((rounded_key.float() - key.float()).pow(2).mean() / key.float().pow(2).mean()).sqrt().item()
    n = sums["queries"]
    return f"sequences={sequences} queries={int(n)} rel_k_err={rel_k_err:.4e} score_err={(sums['score_err2'] / sums['pairs']) ** 0.5:.4e} score_std={sums['score_std'] / n:.4e} tv={sums['tv'] / n:.4e} tv_p99={tv_p99:.4e} kl={sums['kl'] / n:.4e} entropy={sums['entropy'] / n:.4e} p_max={sums['p_max'] / n:.4e}"


def _wrap_core_attention(
    core_attention: torch.nn.Module,
    parts: frozenset[str],
    stats_name: str | None,
    v_scale: float,
    score_stats_name: str | None = None,
) -> None:
    """`v_scale`: the factor the sampler's V carries (MILES_SAMPLER_V_SCALE_LOG2); V is quantized at that scale."""
    original_forward = core_attention.forward
    pending_stats = [stats_name] if stats_name is not None else []
    pending_output_stats = list(pending_stats)
    pending_score_stats = [score_stats_name] if score_stats_name is not None else []
    k_rms: list[torch.Tensor] = []

    def forward(query, key, value, *args, **kwargs):
        if pending_score_stats:
            name = pending_score_stats.pop()
            packed = kwargs.get("packed_seq_params")
            assert packed is not None and query.dim() == 3, "score stats need packed sequences (qkv_format thd)"
            logger.info("sampler emulation score stats %s: %s", name, _score_stats(query, key, packed))
        if pending_stats:
            name = pending_stats.pop()
            logger.info("sampler emulation stats %s K: %s", name, _block_scale_stats(key))
            logger.info("sampler emulation stats %s V: %s", name, _block_scale_stats(value))
            logger.info("sampler emulation stats %s Q: %s", name, _element_stats(query))
        kernel = "kv_nvfp4_kernel" in parts
        if "k_nvfp4_equalized" in parts:
            if not k_rms:
                k_rms.append(_pair_rms(key))
                ratio = k_rms[0].max() / k_rms[0].min()
                logger.info("sampler emulation K rms per coordinate: min=%.3e median=%.3e max=%.3e max/min=%.1f", k_rms[0].min(), k_rms[0].median(), k_rms[0].max(), ratio)
            key = _nvfp4_equalized(key, k_rms[0])
        elif "k_nvfp4_rotated" in parts:
            key = _nvfp4_rotated(key)
        elif kernel or "kv_nvfp4" in parts or "k_nvfp4" in parts:
            key = _nvfp4_kv(key, kernel)
        if kernel or "kv_nvfp4" in parts or "v_nvfp4" in parts:
            value = _nvfp4_kv(value * v_scale, kernel) * (1.0 / v_scale)
        if "q_fp8" in parts:
            query = _fp8_straight_through(query)
        output = original_forward(query, key, value, *args, **kwargs)
        if pending_output_stats:
            logger.info("sampler emulation stats %s OUT: %s", pending_output_stats.pop(), _element_stats(output))
        if "out_fp8" in parts:
            output = _fp8_straight_through(output)
        return output

    core_attention.forward = forward


def maybe_install_sampler_emulation(model_chunks) -> None:
    """Wrap every layer's core attention when MILES_SAMPLER_EMULATION is set."""
    spec = os.environ.get(ENV_VAR, "").strip()
    stats = os.environ.get(STATS_ENV_VAR, "") == "1"
    score_stats = os.environ.get(SCORE_STATS_ENV_VAR, "") == "1"
    if not spec and not stats and not score_stats:
        return
    parts = frozenset(part.strip() for part in spec.split(",") if part.strip())
    unknown = parts - set(PARTS)
    assert not unknown, f"{ENV_VAR}: unknown parts {sorted(unknown)}, expected a subset of {PARTS}"
    wrapped = 0
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if name.endswith("self_attention") and hasattr(module, "core_attention"):
                layer = int(name.split(".layers.")[1].split(".")[0])
                v_scale = 2.0 ** _V_LOG2_SCALES[layer] if _V_LOG2_SCALES is not None else 1.0
                _wrap_core_attention(module.core_attention, parts, name if stats else None, v_scale, name if score_stats else None)
                wrapped += 1
    assert wrapped > 0, f"{ENV_VAR} is set but no self_attention.core_attention module was found"
    logger.info("sampler emulation installed: parts=%s layers=%d", sorted(parts), wrapped)
