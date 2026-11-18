"""Engine-side loader for weight updates that carry routed experts stacked along dim 0.

SGLang's own loader takes one tensor per expert matrix and per scale, and its cost follows the tensor count: a
Qwen3-30B FP8 update is about 37,000 tensors and 4 s of host work per engine for 0.03 s of copies. With
--update-weight-stacked-experts the trainer sends the experts of a layer matrix as one tensor [experts, rows, cols]
under the name of its first expert with the id replaced by "<first>-<last>"
(model.layers.3.mlp.experts.0-127.gate_proj.weight), and this loader writes it with one copy.

It runs inside the SGLang scheduler process, registered with --custom-weight-loader and named as the load format
of the update request, so it imports nothing of miles. The request carries one flat byte tensor; the tensors' names,
dtypes, shapes and byte ranges travel in the name of that tensor (BUCKET_NAME_PREFIX + base64 JSON), because the
request format of a custom loader is a list of (name, tensor).

What a stacked tensor is written to is decided by SGLang's code, not here: the model's load_weights resolves the
first and the last expert's name to a parameter, a shard and an expert slot (its calls to the MoE layer's
weight_loader are recorded, not executed), and the copy itself is the layer's own _load_w13 / _load_w2 on the
slots' slice. Layers, quantization methods and tensors outside the cases handled below go to load_weights one expert
at a time, as before.

The first time a stacked name arrives (the first update of an engine), the bucket is also loaded through
load_weights, and the two results are compared on every parameter the bucket touches, with the payload's bytes
inverted so that an unwritten or misplaced byte cannot go unnoticed. A difference turns the stacked path off for the
process and leaves the weights as load_weights wrote them. MILES_STACKED_EXPERT_LOADER_CHECK=0 skips the comparison.
"""

import base64
import json
import logging
import os
import re
import time

import torch

logger = logging.getLogger(__name__)

LOAD_FORMAT = "miles.backends.sglang_utils.stacked_expert_loader.load_bucket"
BUCKET_NAME_PREFIX = "miles_stacked_bucket."
_STACKED_NAME = re.compile(r"\.experts\.(\d+)-(\d+)\.")
_CHECK = os.environ.get("MILES_STACKED_EXPERT_LOADER_CHECK", "1") != "0"

# per model: stacked name -> _Target, or None when the name has to go through load_weights
_targets: dict[int, dict[str, "_Target | None"]] = {}
# per model: stacked names whose stacked load was compared with load_weights
_checked: dict[int, set[str]] = {}
_disabled = False
_stats = {"calls": 0, "stacked": 0, "unstacked": 0, "plain": 0, "seconds": 0.0}


class _Target:
    """Where the experts [first, last] of a stacked tensor go: slots [slot, slot + count) of `param`."""

    def __init__(self, param, layer, param_name, shard_id, slot, count):
        self.param = param
        self.layer = layer
        self.param_name = param_name
        self.shard_id = shard_id
        self.slot = slot
        self.count = count


def encode_bucket_name(metadata) -> str:
    """The name under which a flattened bucket travels: its tensors' (name, dtype, shape, start, end)."""
    entries = [
        [meta.name, str(meta.dtype).removeprefix("torch."), list(meta.shape), meta.start_idx, meta.end_idx]
        for meta in metadata
    ]
    return BUCKET_NAME_PREFIX + base64.urlsafe_b64encode(json.dumps(entries).encode()).decode()


def decode_bucket(name: str, flat: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    entries = json.loads(base64.urlsafe_b64decode(name.removeprefix(BUCKET_NAME_PREFIX)))
    return [
        (tensor_name, flat[start:end].view(getattr(torch, dtype)).reshape(shape))
        for tensor_name, dtype, shape, start, end in entries
    ]


def load_bucket(model, named_tensors) -> None:
    """SGLang custom weight loader (load_format = LOAD_FORMAT)."""
    started = time.perf_counter()
    tensors: list[tuple[str, torch.Tensor]] = []
    for name, tensor in named_tensors:
        if name.startswith(BUCKET_NAME_PREFIX):
            tensors.extend(decode_bucket(name, tensor))
        else:
            tensors.append((name, tensor))

    plain = [(name, tensor) for name, tensor in tensors if _STACKED_NAME.search(name) is None]
    stacked = [(name, tensor) for name, tensor in tensors if _STACKED_NAME.search(name) is not None]
    if plain:
        model.load_weights(plain)
    if stacked:
        _load_stacked(model, stacked)

    _stats["calls"] += 1
    _stats["plain"] += len(plain)
    _stats["seconds"] += time.perf_counter() - started
    if _stats["calls"] in (1, 10, 100) or _stats["calls"] % 1000 == 0:
        logger.info(
            "stacked expert loader: %d requests, %d plain tensors, %d stacked tensors written with one copy each, "
            "%d stacked tensors through load_weights, %.2f s in all",
            *(_stats[key] for key in ("calls", "plain", "stacked", "unstacked", "seconds")),
        )


def _unstack(named_tensors):
    """The per-expert (name, tensor) list that SGLang's load_weights takes."""
    unstacked = []
    for name, tensor in named_tensors:
        match = _STACKED_NAME.search(name)
        first, last = int(match.group(1)), int(match.group(2))
        assert tensor.shape[0] == last - first + 1, f"{name}: {tuple(tensor.shape)}"
        for offset in range(last - first + 1):
            unstacked.append((_expert_name(name, match, first + offset), tensor[offset]))
    return unstacked


def _expert_name(name: str, match: re.Match, expert_id: int) -> str:
    return f"{name[: match.start()]}.experts.{expert_id}.{name[match.end() :]}"


def _load_stacked(model, named_tensors) -> None:
    global _disabled
    targets = _targets.setdefault(id(model), {})
    unknown = [name for name, _tensor in named_tensors if name not in targets]
    if unknown and not _disabled:
        try:
            targets.update(_resolve_targets(model, [(name, t) for name, t in named_tensors if name in unknown]))
        except Exception:
            logger.exception("stacked expert loader: cannot resolve targets; loading through load_weights from now on")
            _disabled = True

    fast = [] if _disabled else [(name, tensor) for name, tensor in named_tensors if targets.get(name) is not None]
    fast_names = {name for name, _tensor in fast}
    slow = [(name, tensor) for name, tensor in named_tensors if name not in fast_names]

    checked = _checked.setdefault(id(model), set())
    if fast and _CHECK and any(name not in checked for name, _tensor in fast):
        if _stacked_load_matches_load_weights(model, fast, targets):
            checked.update(fast_names)
        else:
            _disabled = True
        # the comparison leaves the weights as load_weights wrote them
        fast_done = True
    else:
        fast_done = False

    if fast and not fast_done:
        for name, tensor in fast:
            _write_stacked(targets[name], tensor)
        _stats["stacked"] += len(fast)
    if slow:
        model.load_weights(_unstack(slow))
        _stats["unstacked"] += len(slow)


def _resolve_targets(model, named_tensors) -> dict:
    """Ask the model's load_weights where the first and the last expert of each stacked tensor go, by recording its
    calls to the MoE layers' weight_loader instead of executing them."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    loaders = []
    for _param_name, param in model.named_parameters():
        loader = getattr(param, "weight_loader", None)
        if isinstance(getattr(loader, "__self__", None), FusedMoE):
            loaders.append((param, loader))

    probes = []
    probe_of = {}
    for name, tensor in named_tensors:
        match = _STACKED_NAME.search(name)
        for expert_id in (int(match.group(1)), int(match.group(2))):
            probe = tensor[0]
            probes.append((_expert_name(name, match, expert_id), probe))
            probe_of[id(probe)] = (name, expert_id)
            probe_of[(name, expert_id)] = probe  # keeps the probe alive, so its id stays its own

    records: dict[tuple[str, int], tuple] = {}

    def recorder_for(param, loader):
        def record(recorded_param, loaded_weight, weight_name, shard_id, expert_id):
            key = probe_of.get(id(loaded_weight))
            assert key is not None and recorded_param is param, f"unexpected load of {weight_name}"
            assert key not in records, f"{key} loaded twice"
            records[key] = (param, loader.__self__, weight_name, shard_id, expert_id)

        return record

    try:
        for param, loader in loaders:
            param.weight_loader = recorder_for(param, loader)
        model.load_weights(probes)
    finally:
        for param, loader in loaders:
            param.weight_loader = loader

    targets = {}
    for name, tensor in named_tensors:
        match = _STACKED_NAME.search(name)
        first, last = int(match.group(1)), int(match.group(2))
        targets[name] = _target_of(
            name, tensor, records.get((name, first)), records.get((name, last)), last - first + 1
        )
    return targets


def _target_of(name, tensor, first_record, last_record, count):
    """The target of a stacked tensor, or None when it is outside what _write_stacked reproduces."""
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe

    if first_record is None or last_record is None or tensor.shape[0] != count:
        return None
    param, layer, param_name, shard_id, first_id = first_record
    if last_record[0] is not param or last_record[2] != param_name or last_record[3] != shard_id:
        return None
    last_id = last_record[4]

    # FusedMoE.weight_loader
    if first_id is None or (layer.uses_expert_location and moe.get_global_expert_location_metadata() is not None):
        return None
    if getattr(param, "_sglang_require_global_experts", False):
        return None
    slot = layer._map_global_expert_id_to_local_expert_id(first_id)
    last_slot = layer._map_global_expert_id_to_local_expert_id(last_id)
    if slot < 0 or last_slot != slot + count - 1:
        return None

    # FusedMoE._weight_loader_impl: only the plain weight and block-scale cases
    method = _layout_method(layer)
    if type(method) not in (moe.Fp8MoEMethod, moe.UnquantizedFusedMoEMethod):
        return None
    if getattr(param, "is_gguf_weight", False) or getattr(param, "is_gguf_weight_type", False):
        return None
    if layer._has_fused_shared or shard_id not in ("w1", "w2", "w3"):
        return None
    if any(key in param_name for key in ("input_scale", "g_idx", "zero", "offset", "weight_shape", "bias")):
        return None
    if "scale" in param_name:
        block_scales = (moe.FusedMoeWeightScaleSupported.GROUP.value, moe.FusedMoeWeightScaleSupported.BLOCK.value)
        if getattr(param, "quant_method", None) not in block_scales:
            return None
    elif "weight" not in param_name:
        return None
    return _Target(param, layer, param_name, shard_id, slot, count)


def _layout_method(layer):
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe

    method = moe._weight_layout_method(layer)
    if layer.scheme is not None:
        method = layer.scheme
    if method.__class__.__name__ == "KTEPWrapperMethod":
        method = method.gpu_method
    return method


def _write_stacked(target: _Target, stacked: torch.Tensor) -> None:
    """FusedMoE._weight_loader_impl for `count` consecutive experts at once: the same steps, with the expert slots'
    slice in place of one slot and every dim shifted by one."""
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe

    layer, param, shard_id = target.layer, target.param, target.shard_id
    method = _layout_method(layer)
    if isinstance(method, moe.UnquantizedFusedMoEMethod):
        method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            layer=layer, param=param, weight_name=target.param_name
        )
    elif isinstance(method, moe.Fp8MoEMethod) and layer.use_flashinfer_trtllm_moe:
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import clear_mxfp8_shuffle_index_cache

        clear_mxfp8_shuffle_index_cache()

    if layer.use_flashinfer_trtllm_moe:
        shard_id = {"w1": "w3", "w3": "w1", "w2": "w2"}[shard_id]
    shard_dim = {"w1": 0, "w2": 1, "w3": 0}[shard_id]
    if getattr(param, "is_transposed", False) or layer.use_triton_kernels:
        shard_dim = int(not shard_dim)

    layer._load_model_weight_or_group_weight_scale(
        shard_id=shard_id,
        shard_dim=shard_dim + 1,
        loaded_weight=stacked,
        expert_data=param.data[target.slot : target.slot + target.count],
    )


def _stacked_load_matches_load_weights(model, named_tensors, targets) -> bool:
    """Load the bucket both ways with inverted bytes and compare every parameter it touches; then load the real
    bytes through load_weights. Inverted bytes differ from the real ones everywhere, so the two results are equal
    only if both ways write the same bytes to the same places."""
    started = time.perf_counter()
    unstacked = _unstack(named_tensors)
    inverted = [(name, _inverted(tensor)) for name, tensor in named_tensors]
    params = list({id(targets[name].param): targets[name].param for name, _tensor in named_tensors}.values())

    model.load_weights(unstacked)
    model.load_weights(_unstack(inverted))
    expected = [param.data.clone() for param in params]
    model.load_weights(unstacked)
    mismatched = []
    try:
        for name, tensor in inverted:
            _write_stacked(targets[name], tensor)
        for param, reference in zip(params, expected, strict=True):
            same = param.data.shape == reference.shape and torch.equal(
                param.data.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
            )
            if not same:
                mismatched.append(next(t.param_name for t in targets.values() if t is not None and t.param is param))
    except Exception:
        logger.exception("stacked expert loader: the stacked load failed")
        mismatched.append("<exception>")
    finally:
        del expected
        model.load_weights(unstacked)

    if mismatched:
        logger.error(
            "stacked expert loader: the stacked load differs from load_weights on %d of %d parameters (%s); "
            "loading through load_weights from now on",
            len(mismatched),
            len(params),
            ", ".join(mismatched[:4]),
        )
        return False
    logger.info(
        "stacked expert loader: %d stacked tensors into %d parameters equal load_weights byte for byte (%.2f s)",
        len(named_tensors),
        len(params),
        time.perf_counter() - started,
    )
    return True


def _inverted(tensor: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_not(tensor.contiguous().view(torch.uint8)).view(tensor.dtype).view(tensor.shape)
