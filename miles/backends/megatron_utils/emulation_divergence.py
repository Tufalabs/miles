"""Where the plain forward and the sampler-emulating forward part, layer by layer (experiment, LLMT-83).

With MILES_SAMPLER_EMULATION_DIVERGENCE=1, in a run whose training step makes a rounded pass and then a plain
pass over each micro-batch (MILES_GAIN_PENALTY or MILES_GAIN_PROJECTION), the first micro-batch of each rank is
compared between the two passes and one line per layer is logged:

  resid, attn, mlp   the residual stream after the layer, the attention sublayer's output and the MoE's output:
                     `rel` = |rounded - plain| / |plain| over the first tokens of the batch, `tok_p50`/`tok_p99`
                     the same per token, `diff_rms` and `rms` the per-token norms of the difference and of the
                     plain value
  router             the share of tokens whose set of chosen experts differs, and the mean number of experts
                     that differ
  pattern            for the first sequences and every head, with the upstream differences included: `score_diff`
                     (RMS difference of the attention scores q.k / sqrt(head_dim) over causal pairs), `score_std`,
                     `tv` (total variation between the two attention patterns of a query, mean), `entropy` of
                     the plain pattern, and `head_tv` / `head_entropy`, the same per head

Together with MILES_SAMPLER_EMULATION_LAYERS (round K in a few layers only) this shows how a difference injected
at one depth grows or dies on its way through the later layers.
"""

import logging
import os

import torch

from miles.backends.training_utils.loss.hub import gain_penalty

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_SAMPLER_EMULATION_DIVERGENCE"
_TOKENS = 4096  # tokens of the micro-batch that are compared
_SEQUENCES = 4  # sequences whose attention patterns are compared
_stored: dict[str, torch.Tensor] = {}
_state = {"armed": True, "last_layer": -1}


def enabled() -> bool:
    return os.environ.get(ENV_VAR, "") == "1"


def _first(output):
    return output[0] if isinstance(output, tuple | list) else output


def _compare(name: str, kind: str, rounded: torch.Tensor, plain: torch.Tensor) -> None:
    r, p = rounded.float().reshape(-1, rounded.shape[-1]), plain.float().reshape(-1, plain.shape[-1])
    difference = (r - p).norm(dim=-1)
    size = p.norm(dim=-1)
    quantiles = torch.quantile(difference / size.clamp_min(1e-12), torch.tensor([0.5, 0.99], device=size.device))
    logger.info(
        "sampler emulation divergence %s %s: rel=%.4e tok_p50=%.4e tok_p99=%.4e diff_rms=%.4e rms=%.4e",
        name,
        kind,
        difference.norm() / size.norm().clamp_min(1e-12),
        quantiles[0],
        quantiles[1],
        difference.pow(2).mean().sqrt(),
        size.pow(2).mean().sqrt(),
    )


def _hook(name: str, kind: str, layer: int):
    def hook(module, inputs, output):
        if not _state["armed"] or not torch.is_grad_enabled():
            return
        key = f"{name}/{kind}"
        value = (output[1] if kind == "router" else _first(output))[:_TOKENS].detach()
        if gain_penalty.rounding_active():
            _stored[key] = value
            return
        rounded = _stored.pop(key, None)
        if rounded is None:
            return
        if kind == "router":
            changed = rounded != value
            logger.info(
                "sampler emulation divergence %s router: tokens_changed=%.4e experts_changed=%.4e",
                name,
                changed.any(dim=-1).float().mean(),
                changed.float().sum(dim=-1).mean() / 2,
            )
        else:
            _compare(name, kind, rounded, value)
        if kind == "resid" and layer == _state["last_layer"]:
            _state["armed"] = False
            _stored.clear()

    return hook


@torch.no_grad()
def _patterns(query: torch.Tensor, key: torch.Tensor, bounds) -> list[torch.Tensor]:
    """Scores q.k / sqrt(head_dim) of the first sequences, [heads, queries, keys] per sequence, as a list."""
    heads, groups, head_dim = query.shape[1], key.shape[1], query.shape[2]
    out = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        q = query[start:end].float().transpose(0, 1)
        k = key[start:end].float().transpose(0, 1).repeat_interleave(heads // groups, dim=0)
        out.append(q @ k.transpose(1, 2) / head_dim**0.5)
    return out


def compare_patterns(name: str, query: torch.Tensor, key: torch.Tensor, packed_seq_params) -> None:
    """Called by the emulation wrapper with the query and the key this pass attends with (the rounded key in
    a rounded pass of a layer that rounds). Stores them in the rounded pass, compares in the plain pass."""
    if not _state["armed"] or not torch.is_grad_enabled() or packed_seq_params is None:
        return
    bounds = packed_seq_params.cu_seqlens_host
    bounds = bounds[: min(_SEQUENCES, len(bounds) - 1) + 1]
    slot = f"{name}/pattern"
    if gain_penalty.rounding_active():
        _stored[slot] = (query[: bounds[-1]].detach().clone(), key[: bounds[-1]].detach().clone())
        return
    stored = _stored.pop(slot, None)
    if stored is not None:
        _log_patterns(name, stored, query.detach(), key.detach(), bounds)


@torch.no_grad()
def _log_patterns(name: str, stored: tuple[torch.Tensor, torch.Tensor], query, key, bounds) -> None:
    sums = torch.zeros(4, dtype=torch.float64)  # squared score difference, pairs, score std, queries
    heads = query.shape[1]
    head_tv, head_entropy = torch.zeros(heads, dtype=torch.float64), torch.zeros(heads, dtype=torch.float64)
    for rounded_scores, scores in zip(_patterns(*stored, bounds), _patterns(query, key, bounds), strict=True):
        length = scores.shape[-1]
        causal = torch.ones(length, length, dtype=torch.bool, device=scores.device).tril()
        log_p = scores.masked_fill(~causal, float("-inf")).log_softmax(dim=-1)
        p = log_p.exp()
        p_rounded = rounded_scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        mean = (scores * causal).sum(-1) / causal.sum(-1)
        variance = (((scores - mean[..., None]) ** 2) * causal).sum(-1) / causal.sum(-1)
        sums[0] += (((rounded_scores - scores) ** 2) * causal).sum().item()
        sums[1] += causal.sum().item() * heads
        sums[2] += variance.sqrt().sum().item()
        sums[3] += heads * length
        head_tv += (0.5 * (p - p_rounded).abs().sum(-1)).sum(-1).double().cpu()
        head_entropy += (-(p * log_p.masked_fill(~causal, 0.0)).sum(-1)).sum(-1).double().cpu()
    queries_per_head = sums[3] / heads
    logger.info(
        "sampler emulation divergence %s pattern: score_diff=%.4e score_std=%.4e tv=%.4e entropy=%.4e head_tv=%s head_entropy=%s",
        name,
        (sums[0] / sums[1]) ** 0.5,
        sums[2] / sums[3],
        head_tv.sum() / sums[3],
        head_entropy.sum() / sums[3],
        ",".join(f"{value:.4f}" for value in (head_tv / queries_per_head).tolist()),
        ",".join(f"{value:.3f}" for value in (head_entropy / queries_per_head).tolist()),
    )


def maybe_install(model_chunks) -> None:
    """Hook every layer's output, attention, MoE and router when MILES_SAMPLER_EMULATION_DIVERGENCE=1."""
    if not enabled():
        return
    assert gain_penalty.configured(), f"{ENV_VAR} needs the two passes of MILES_GAIN_PENALTY or MILES_GAIN_PROJECTION"
    layers = 0
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if ".layers." not in name:
                continue
            tail = name.split(".layers.")[1]
            layer = int(tail.split(".")[0])
            kind = {"": "resid", "self_attention": "attn", "mlp": "mlp", "mlp.router": "router"}.get(tail.partition(".")[2])
            if kind is None:
                continue
            module.register_forward_hook(_hook(name.split(".layers.")[0] + f".layers.{layer}", kind, layer))
            if kind == "resid":
                layers += 1
                _state["last_layer"] = max(_state["last_layer"], layer)
    assert layers > 0, f"{ENV_VAR}: no transformer layer found"
    logger.info("sampler emulation divergence installed: %d layers", layers)
