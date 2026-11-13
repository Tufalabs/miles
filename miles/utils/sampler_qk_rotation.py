"""Rotate q and k of an SGLang attention layer by a random orthogonal matrix before the KV cache (LLMT-83).

Called from SGLang's Qwen3-MoE attention through SGLang's source patcher (DUMPER_SOURCE_PATCHER_CONFIG; the
patch file is in rave's artifacts folder of LLMT-83). A rotation shared by q and k leaves q.k unchanged, so the
attention computes the same function; what changes is the basis in which a quantized KV cache rounds K.

MILES_SAMPLER_QK_ROTATION:
  fixed      one rotation per layer for the whole run
  per_sync   redrawn every time the model loads weights (each weight update)
The rotation is a Hadamard matrix with random signs on its rows, per layer, shared by the heads. It is kept in
one buffer per layer and device that is rewritten in place, so a captured CUDA graph sees the new values.
"""

import logging
import os
import zlib

import torch

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_SAMPLER_QK_ROTATION"
_MODE = os.environ.get(ENV_VAR, "")
assert _MODE in ("", "fixed", "per_sync"), f"{ENV_VAR}: {_MODE!r}"
_epoch = 0
_buffers: dict[tuple[int, int, str], torch.Tensor] = {}


def _hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    assert h.shape[0] == n, f"head_dim {n} is not a power of two"
    return h / n**0.5


def _draw(layer_id: int, head_dim: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(zlib.crc32(f"{layer_id}:{_epoch}".encode()))
    signs = (torch.randint(0, 2, (head_dim,), generator=generator) * 2 - 1).float()
    return signs[:, None] * _hadamard(head_dim)


def bump_epoch() -> None:
    """New rotations for every layer (per_sync only). Called when the model loads weights."""
    global _epoch
    if _MODE != "per_sync":
        return
    _epoch += 1
    for (layer_id, head_dim, _), buffer in _buffers.items():
        buffer.copy_(_draw(layer_id, head_dim))


def rotate_qk(attention, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q and k of one layer, each [tokens, heads * head_dim], rotated within every head."""
    if not _MODE:
        return q, k
    head_dim = attention.head_dim
    key = (attention.attn.layer_id, head_dim, str(q.device))
    rotation = _buffers.get(key)
    if rotation is None:
        rotation = _draw(attention.attn.layer_id, head_dim).to(q.device)
        _buffers[key] = rotation
        if key[0] == 0:
            logger.warning("sampler q/k rotation active: mode=%s head_dim=%d device=%s", _MODE, head_dim, q.device)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(-1, head_dim).float() @ rotation).to(x.dtype).reshape(x.shape)

    return rotate(q), rotate(k)
