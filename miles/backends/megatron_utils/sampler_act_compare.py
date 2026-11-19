"""Compare the trainer's per-layer activations of generated tokens with SGLang's (LLMT-83 mismatch probe).

MILES_ACT_COMPARE_DIR=<dir>         SGLang's dumps (sampler_act_dump.py: files named by the SHA-1 of a sequence's
                                    token ids, with the generated tokens' positions, emb, attn and moe).
MILES_ACT_COMPARE_ROLLOUTS=<id>,... the rollouts whose training forward is compared.
MILES_ACT_COMPARE_LAYERS=<l>,...    the layers reported (default 0, 4, ..., 44 and 47).
At the same positions as the dump, three tensors per layer: attn (the attention output, before the residual add),
moe (the MoE output) and resid, the residual stream after the layer rebuilt on both sides the same way as the
embedding plus the sum of attn and moe over the layers up to it. Accumulated over the dumped sequences of a
rollout: the sum of ||trainer - sglang||^2, of ||sglang||^2, of the per-token cosine and the token count, logged
by each rank after each micro-batch as the rollout's running totals ("act compare rollout=<id> {json}"); the last
line per rank and rollout holds that rank's totals. Each (micro-batch, layer, tensor) counts once, so the
forward that activation recomputation repeats in the backward is not counted twice.
"""

import hashlib
import json
import logging
import os
import re

import numpy as np
import torch

from miles.backends.training_utils.metrics import debug_dump

logger = logging.getLogger(__name__)

DIR = os.environ.get("MILES_ACT_COMPARE_DIR", "")
ROLLOUTS = {int(r) for r in os.environ.get("MILES_ACT_COMPARE_ROLLOUTS", "").split(",") if r.strip()}
LAYERS = [int(r) for r in (os.environ.get("MILES_ACT_COMPARE_LAYERS", "") or ",".join(map(str, [*range(0, 48, 4), 47]))).split(",")]

_files: dict[str, str] = {}
_batch: dict = {}
_totals: dict[int, dict] = {}
_micro = [0]


def enabled() -> bool:
    return bool(DIR and ROLLOUTS)


def set_batch(unconcat_tokens) -> None:
    _batch.clear()
    _micro[0] += 1
    rollout = debug_dump._training_rollout_id
    if not enabled() or rollout not in ROLLOUTS:
        return
    if not _files:
        _files.update({name[:-3]: os.path.join(DIR, name) for name in os.listdir(DIR) if name.endswith(".pt")})
        logger.info("act compare: %d dumped sequences in %s", len(_files), DIR)
    lengths = [int(t.numel()) for t in unconcat_tokens]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    matched = []
    for start, tokens in zip(starts, unconcat_tokens, strict=True):
        name = hashlib.sha1(tokens.detach().cpu().numpy().astype(np.int32).tobytes()).hexdigest()
        if name in _files:
            record = torch.load(_files[name], map_location="cpu", weights_only=False)
            assert record["tokens"] == tokens.tolist(), f"{_files[name]}: tokens differ"
            matched.append((start, record))
    if not matched:
        return
    device = torch.cuda.current_device()
    index = torch.cat([torch.tensor(r["positions"], dtype=torch.long) + s for s, r in matched]).to(device)
    _batch.update(
        rollout=rollout,
        micro=_micro[0],
        index=index,
        emb=torch.cat([r["emb"] for _, r in matched]).to(device).float(),
        attn=torch.cat([r["attn"] for _, r in matched], dim=1),
        moe=torch.cat([r["moe"] for _, r in matched], dim=1),
        done=set(),
        sequences=len(matched),
    )
    _batch["resid_s"] = _batch["emb"].clone()
    _batch["resid_t"] = None


def _rows(output) -> torch.Tensor:
    x = output[0] if isinstance(output, tuple) else output
    return x.reshape(-1, x.shape[-1])[_batch["index"]].detach().float()


def _add(layer: int, kind: str, trainer: torch.Tensor, sglang: torch.Tensor) -> None:
    if layer not in LAYERS:
        return
    totals = _totals.setdefault(_batch["rollout"], {})
    entry = totals.setdefault(f"{kind}{layer}", [0.0, 0.0, 0.0, 0])
    entry[0] += float((trainer - sglang).pow(2).sum())
    entry[1] += float(sglang.pow(2).sum())
    entry[2] += float(torch.nn.functional.cosine_similarity(trainer, sglang, dim=-1).sum())
    entry[3] += trainer.shape[0]


def _hook(layer: int, kind: str):
    def hook(module, args, output):
        if not _batch or (layer, kind) in _batch["done"]:
            return
        _batch["done"].add((layer, kind))
        trainer = _rows(output)
        if kind == "emb":
            _batch["resid_t"] = trainer
            return
        sglang = _batch[kind][layer].to(trainer.device).float()
        _add(layer, kind, trainer, sglang)
        _batch["resid_t"] = _batch["resid_t"] + trainer
        _batch["resid_s"] = _batch["resid_s"] + sglang
        if kind == "moe":
            _add(layer, "resid", _batch["resid_t"], _batch["resid_s"])
            if layer == max(LAYERS):
                logger.info("act compare rollout=%d sequences=%d %s", _batch["rollout"], _batch["sequences"], json.dumps(_totals[_batch["rollout"]]))

    return hook


def maybe_install(model_chunks) -> None:
    if not enabled():
        return
    hooks = 0
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if name.endswith("embedding") and module.__class__.__name__.endswith("Embedding") and ".layers." not in name:
                module.register_forward_hook(_hook(-1, "emb"))
                hooks += 1
            match = re.search(r"(?:^|\.)layers\.(\d+)\.(self_attention|mlp)$", name)
            if match:
                module.register_forward_hook(_hook(int(match.group(1)), "attn" if match.group(2) == "self_attention" else "moe"))
                hooks += 1
    logger.info("act compare installed: %d hooks, rollouts %s, layers %s, dumps in %s", hooks, sorted(ROLLOUTS), LAYERS, DIR)
