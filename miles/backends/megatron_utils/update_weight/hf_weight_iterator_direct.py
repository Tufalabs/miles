import itertools
import logging
import os
import re
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.megatron_utils.lora.native import export_lora_hf_named
from miles.backends.megatron_utils.megatron_to_hf import convert_stacked_experts_to_hf, convert_to_hf
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.megatron_utils.sglang import monkey_patch_torch_reductions
from miles.backends.megatron_utils.update_weight.expert_gather import ExpertGather
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    _iter_mm_tower_units,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.test_utils.fault_injector.controller import reach_fault_hook
from miles.utils.test_utils.fault_injector.models import FaultHookName
from miles.utils.types import ParamInfo

logger = logging.getLogger(__name__)

# MILES_WEIGHT_SYNC_STACK_EXPERTS=0 converts and quantizes every routed expert matrix by itself, as before.
_STACK_EXPERTS = os.environ.get("MILES_WEIGHT_SYNC_STACK_EXPERTS", "1") != "0"
_GROUPED_EXPERT_WEIGHT = re.compile(r"^(.*\.weight)(\d+)$")
_HF_EXPERT_ID = re.compile(r"\.experts\.(\d+)\.")


@dataclass(frozen=True)
class _ExpertBatch:
    param_infos: Sequence[ParamInfo]
    gathers: tuple[ExpertGather, ...]
    # The stacked wire format has other units than the per-expert one, and a gather keeps one layout for life.
    stacked_gathers: tuple[ExpertGather, ...] = ()


@dataclass(frozen=True)
class _StackedExpertPlan:
    """How one matrix of the experts of a batch (ascending expert id) leaves the stacked conversion."""

    # per expert, the HF names of its unit as the per-expert conversion gives them
    unit_names: tuple[tuple[str, ...], ...]
    # (first, stop, names): experts [first, stop) of the stack go out as one unit of tensors [stop - first, ...]
    # under names whose expert id is "<first id>-<last id>"; a run of one expert keeps its own names and shape.
    # None when the HF names do not carry the expert id in the expected place.
    stacked_units: tuple[tuple[int, int, tuple[str, ...]], ...] | None


class HfWeightIteratorDirect(MegatronHfWeightIteratorBase):
    # Lower bound: TP/ETP/EP are always gathered; PP follows the requirement.
    forced_placement = WeightUpdatePlacement(gather_pp=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parallel = get_parallel_state()
        self._convert_experts_before_gather = parallel.etp.size == 1
        # Set by the weight updater when the receiver takes tensors of several experts stacked along dim 0 under
        # names "...experts.<first>-<last>...." (see send_stacked_experts on the transfer protocol).
        self.send_stacked_experts = False
        # One conversion and one quantization call for all the experts a batch holds of a layer matrix.
        self.stack_experts = _STACK_EXPERTS
        self._stacked_expert_plans: dict[tuple[str, ...], _StackedExpertPlan | None] = {}
        non_expert_infos, expert_infos = _get_megatron_local_param_infos(
            self.args, self.model, gather_pp=self.placement.gather_pp
        )
        ep_size = parallel.ep.size
        self._non_expert_batches = _pack_param_infos_by_size(self.args, non_expert_infos)
        self._expert_batches = []
        if not self._convert_experts_before_gather:
            self._expert_batches = [
                _ExpertBatch(param_infos=infos, gathers=())
                for infos in _pack_param_infos_by_size(self.args, expert_infos, size_multiplier=ep_size)
            ]
        elif expert_infos:
            edp = parallel.edp
            assert edp is not None, "Expert data parallel state is required for MoE weight updates"
            owner_infos = _partition_expert_infos(
                expert_infos, num_local_experts=self.args.backend.num_experts // ep_size, edp_size=edp.size
            )
            # At ETP1, TP x DP x CP is exactly EP x EDP. Borrow Megatron's
            # combined group to avoid repacking between EP and EDP gathers.
            combined = parallel.tp_dp_cp
            assert (
                combined is not None and combined.size == ep_size * edp.size
            ), "The combined TP/DP/CP group must cover all EP/EDP ranks"
            groups = ([parallel.pp] if self.placement.gather_pp else []) + [combined]
            # Pack each owner's share first so a round can quantize on every EDP
            # replica, even when the complete local expert set spans many batches.
            owner_batches = [
                _pack_param_infos_by_size(self.args, infos, size_multiplier=ep_size * edp.size)
                for infos in owner_infos
            ]
            self._expert_batches = [
                _ExpertBatch(
                    param_infos=batches[edp.rank],
                    gathers=tuple(ExpertGather(group=group.group) for group in groups if group.size > 1),
                    stacked_gathers=tuple(ExpertGather(group=group.group) for group in groups if group.size > 1),
                )
                for batches in itertools.zip_longest(*owner_batches, fillvalue=())
            ]

    def _iter_hf_param_units(self, weights, *, materialize):
        rank = dist.get_rank()

        pbar = tqdm(
            total=len(self._non_expert_batches) + len(self._expert_batches),
            disable=rank != 0,
            desc="Update weights",
        )
        for param_infos in self._non_expert_batches:
            named_params = _materialize_non_expert_batch(
                self.args, param_infos, weights, gather_pp=self.placement.gather_pp
            )
            if materialize:
                yield from self._convert_to_hf_param_units(named_params)
            del named_params
            pbar.update(1)
        for batch in self._expert_batches:
            if self._convert_experts_before_gather:
                units = self._convert_and_gather_expert_batch(batch, weights)
                if materialize:
                    yield from units
                del units
            else:
                # ETP shards must form complete experts before conversion/quantization.
                named_params = _gather_megatron_expert_batch(
                    self.args, batch.param_infos, weights, gather_pp=self.placement.gather_pp
                )
                if materialize:
                    yield from self._convert_to_hf_param_units(named_params)
                del named_params
            pbar.update(1)
        pbar.close()
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _convert_and_gather_expert_batch(self, batch: _ExpertBatch, weights):
        """Convert once per expert across EP/EDP, then gather HF weights and scales."""
        device = torch.device("cuda", torch.cuda.current_device())
        rank = dist.get_rank()
        # Sender placement is independent of ownership: non-senders also
        # quantize their assigned experts, once across all expert-DP replicas.
        local_params = (
            (info.name, weights[info.name].detach().to(device=device, non_blocking=True))
            for info in batch.param_infos
            if info.src_rank == rank
        )
        if self.stack_experts:
            units = self._convert_expert_units_stacked(list(local_params))
        else:
            units = list(self._convert_to_hf_param_units(local_params))
        for gather in batch.stacked_gathers if self.send_stacked_experts else batch.gathers:
            units = gather(units, device=device)
        return units

    def _convert_expert_units_stacked(self, named_params: list[tuple[str, torch.Tensor]]):
        """The units of _convert_to_hf_param_units, computed with one conversion and one quantization call for all
        the experts a batch holds of a layer matrix. With send_stacked_experts off the result is the per-expert
        one, name for name and byte for byte; with it on, each run of consecutive experts is one unit."""
        groups: dict[str, list[int]] = {}
        for index, (name, _param) in enumerate(named_params):
            match = _GROUPED_EXPERT_WEIGHT.match(name)
            groups.setdefault(match.group(1) if match else name, []).append(index)

        units_of_param: list = [None] * len(named_params)
        units_of_group: list = []
        for indices in groups.values():
            # ascending expert id, so that consecutive experts are consecutive in the stack
            indices = sorted(
                indices, key=lambda index: int(_GROUPED_EXPERT_WEIGHT.match(named_params[index][0]).group(2))
            )
            members = [named_params[index] for index in indices]
            stacked_units, member_units = self._convert_expert_group(members)
            units_of_group.append(stacked_units)
            for index, unit in zip(indices, member_units, strict=True):
                units_of_param[index] = unit
        if self.send_stacked_experts:
            return [unit for units in units_of_group for unit in units]
        return units_of_param

    def _convert_expert_group(self, members: list[tuple[str, torch.Tensor]]):
        """(units in the stacked wire format, one unit per expert) of one layer matrix's experts. The second is
        lazy: only the format in use is materialized."""
        names = tuple(name for name, _param in members)
        known = names in self._stacked_expert_plans
        plan = self._stacked_expert_plans.get(names)
        stackable = (
            len(members) > 1
            and (plan is not None or not known)
            and all(_GROUPED_EXPERT_WEIGHT.match(name) for name in names)
            and all(param.dim() == 2 and param.shape == members[0][1].shape for _name, param in members)
        )
        if not stackable:
            units = list(self._convert_to_hf_param_units(members))
            return units, units

        try:
            stacked = convert_stacked_experts_to_hf(
                self.args,
                self.model_name,
                names[0],
                torch.stack([param for _name, param in members]),
                self.quantization_config,
            )
        except Exception:
            if known:
                raise
            logger.warning("No stacked conversion for %s; converting its experts one by one", names[0], exc_info=True)
            stacked = None
        if stacked is None:
            assert not known, f"The stacked conversion of {names[0]} stopped working"
            self._stacked_expert_plans[names] = None
            units = list(self._convert_to_hf_param_units(members))
            return units, units

        if plan is None:
            # First sight of this group: the per-expert conversion gives the names, and the stacked result has to
            # equal it in every byte before anything is sent.
            reference = list(self._convert_to_hf_param_units(members))
            plan = _build_stacked_expert_plan(names, reference, stacked)
            self._stacked_expert_plans[names] = plan

        tensors = [tensor for _name, tensor in stacked]
        if not self.send_stacked_experts or plan.stacked_units is None:
            units = [
                [(name, tensor[expert]) for name, tensor in zip(unit_names, tensors, strict=True)]
                for expert, unit_names in enumerate(plan.unit_names)
            ]
            return units, units
        stacked_units = [
            [
                (name, tensor[first] if stop - first == 1 else tensor[first:stop])
                for name, tensor in zip(unit_names, tensors, strict=True)
            ]
            for first, stop, unit_names in plan.stacked_units
        ]
        return stacked_units, [None] * len(members)

    def _export_pp_local_lora(self, adapter):
        assert adapter is None, "multi-LoRA export requires --megatron-to-hf-mode bridge"
        return export_lora_hf_named(self.model)

    def _convert_to_hf_param_units(self, named_params: Sequence[tuple[str, torch.Tensor]]):
        for name, param in named_params:
            yield list(
                convert_to_hf(
                    self.args, self.model_name, name, param, self.quantization_config, self.packed_weight_basenames
                )
            )


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
    )


def _build_stacked_expert_plan(
    megatron_names: tuple[str, ...],
    reference: list[list[tuple[str, torch.Tensor]]],
    stacked: list[tuple[str, torch.Tensor]],
) -> _StackedExpertPlan:
    """Check the stacked conversion of a group against its per-expert conversion, byte for byte, and derive the
    names of both wire formats from the per-expert names."""
    for expert, unit in enumerate(reference):
        assert len(unit) == len(
            stacked
        ), f"{megatron_names[expert]}: {len(unit)} tensors per expert, {len(stacked)} from the stacked conversion"
    for position, (first_name, tensor) in enumerate(stacked):
        assert reference[0][position][0] == first_name, f"{reference[0][position][0]} != {first_name}"
        expected = torch.stack([unit[position][1] for unit in reference])
        if not _same_bytes(tensor, expected):
            raise RuntimeError(
                f"Stacked conversion of {first_name} ({len(reference)} experts from {megatron_names[0]}) differs from "
                f"the per-expert conversion: {tuple(tensor.shape)} {tensor.dtype} vs {tuple(expected.shape)} "
                f"{expected.dtype}. Set MILES_WEIGHT_SYNC_STACK_EXPERTS=0 to convert per expert."
            )

    unit_names = tuple(tuple(name for name, _tensor in unit) for unit in reference)
    # The expert id of each unit, read from its HF names; all tensors of a unit must agree on it.
    expert_ids = []
    for names in unit_names:
        ids = {tuple(_HF_EXPERT_ID.findall(name)) for name in names}
        if len(ids) != 1 or len(next(iter(ids))) != 1:
            return _StackedExpertPlan(unit_names=unit_names, stacked_units=None)
        expert_ids.append(int(next(iter(ids))[0]))

    stacked_units = []
    first = 0
    for stop in range(1, len(expert_ids) + 1):
        if stop < len(expert_ids) and expert_ids[stop] == expert_ids[stop - 1] + 1:
            continue
        if stop - first == 1:
            names = unit_names[first]
        else:
            span = f".experts.{expert_ids[first]}-{expert_ids[stop - 1]}."
            names = tuple(_HF_EXPERT_ID.sub(span, name) for name in unit_names[first])
            # every expert of the run must have the first one's names up to the expert id
            for expert in range(first, stop):
                if tuple(_HF_EXPERT_ID.sub(span, name) for name in unit_names[expert]) != names:
                    return _StackedExpertPlan(unit_names=unit_names, stacked_units=None)
        stacked_units.append((first, stop, names))
        first = stop
    return _StackedExpertPlan(unit_names=unit_names, stacked_units=tuple(stacked_units))


def _load_or_allocate_params(param_infos: Sequence[ParamInfo], megatron_local_weights) -> list[torch.Tensor]:
    """Owners load from the weight source; other ranks allocate receive buffers."""
    params = []
    for info in param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    torch.cuda.synchronize()
    return params


def _broadcast_across_pp(param_infos: Sequence[ParamInfo], params: Sequence[torch.Tensor]) -> None:
    pp = get_parallel_state().pp
    if pp.size == 1:
        return
    pp_ranks = dist.get_process_group_ranks(pp.group)
    handles = []
    for info, param in zip(param_infos, params, strict=True):
        if info.src_rank in pp_ranks:
            handles.append(dist.broadcast(param, src=info.src_rank, group=pp.group, async_op=True))
    for handle in handles:
        handle.wait()


def _set_tp_attrs(param_infos: Sequence[ParamInfo], params: Sequence[torch.Tensor]) -> None:
    for info, param in zip(param_infos, params, strict=True):
        for key, value in info.attrs.items():
            setattr(param, key, value)


def _materialize_non_expert_batch(
    args: Namespace,
    param_infos: Sequence[ParamInfo],
    megatron_local_weights,
    *,
    gather_pp: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Load -> PP broadcast (when gather_pp) -> TP all_gather."""
    monkey_patch_torch_reductions()
    params = _load_or_allocate_params(param_infos, megatron_local_weights)
    if gather_pp:
        _broadcast_across_pp(param_infos, params)
    _set_tp_attrs(param_infos, params)
    gathered = all_gather_params_async(args, list(zip(param_infos, params, strict=True)))
    return [(info.name, param) for info, param in zip(param_infos, gathered, strict=True)]


def _gather_megatron_expert_batch(
    args: Namespace,
    param_infos: Sequence[ParamInfo],
    megatron_local_weights,
    *,
    gather_pp: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Load -> PP broadcast (when gather_pp) -> ETP all_gather -> EP all_gather.

    Expert metadata is EP-local; the full expert set is materialized by a
    symmetric EP all_gather with a name exchange.
    """
    monkey_patch_torch_reductions()
    params = _load_or_allocate_params(param_infos, megatron_local_weights)
    if gather_pp:
        _broadcast_across_pp(param_infos, params)
    _set_tp_attrs(param_infos, params)
    etp_gathered = all_gather_params_async(args, list(zip(param_infos, params, strict=True)))

    ep = get_parallel_state().ep
    if ep.size == 1:
        return [(info.name, param) for info, param in zip(param_infos, etp_gathered, strict=True)]

    names = [info.name for info in param_infos]
    all_names: list = [None] * ep.size
    dist.all_gather_object(all_names, names, group=ep.group)
    for ep_names in all_names:
        assert len(ep_names) == len(
            names
        ), f"EP-asymmetric expert batch: {len(names)} params locally vs {len(ep_names)} on a peer rank"

    all_gathered: list[list[tuple[str, torch.Tensor]]] = [[] for _ in range(ep.size)]
    handles = []
    for i, param in enumerate(etp_gathered):
        buffers = [torch.empty_like(param, device=torch.cuda.current_device()) for _ in range(ep.size)]
        reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER)
        handles.append(dist.all_gather(buffers, param, group=ep.group, async_op=True))
        for ep_rank, ep_names in enumerate(all_names):
            all_gathered[ep_rank].append((ep_names[i], buffers[ep_rank]))
    for handle in handles:
        handle.wait()

    return [named for per_rank in all_gathered for named in per_rank]


def _pack_param_infos_by_size(
    args: Namespace, param_infos: list[ParamInfo], *, size_multiplier: int = 1
) -> list[list[ParamInfo]]:
    """Greedy size packing into gather batches ≤ update_weight_buffer_size."""
    batches: list[list[ParamInfo]] = [[]]
    buffer_size = 0
    for info in param_infos:
        size = _get_param_full_size(info) * size_multiplier
        if buffer_size + size > args.update_weight_buffer_size and batches[-1]:
            batches.append([])
            buffer_size = 0
        batches[-1].append(info)
        buffer_size += size
    return [batch for batch in batches if batch]


def _partition_expert_infos(
    param_infos: Sequence[ParamInfo], *, num_local_experts: int, edp_size: int
) -> list[list[ParamInfo]]:
    """Assign contiguous local-expert ranges to EDP replicas, keeping FC1/FC2 together."""
    count, remainder = divmod(num_local_experts, edp_size)
    owners = [owner for owner in range(edp_size) for _ in range(count + (owner < remainder))]
    partitions: list[list[ParamInfo]] = [[] for _ in range(edp_size)]
    for info in param_infos:
        # The direct converter uses TE grouped weights with global expert IDs.
        match = re.search(r"\.weight(\d+)$", info.name)
        if match is None:
            raise ValueError(f"Cannot identify the routed expert in {info.name!r}")
        local_expert = int(match.group(1)) % num_local_experts
        partitions[owners[local_expert]].append(info)
    return partitions


def _get_param_full_size(info: ParamInfo) -> int:
    if is_routed_expert_param(info.name):
        tp_size = get_parallel_state().etp.size
    else:
        tp_size = get_parallel_state().tp.size
    return info.size * tp_size


def _get_megatron_local_param_infos(
    args: Namespace, model: Sequence[torch.nn.Module], *, gather_pp: bool
) -> tuple[list[ParamInfo], list[ParamInfo]]:
    """Collect param metadata, exchanged across PP when gather_pp.

    Returns (non_expert_infos, expert_infos); expert infos stay EP-local.
    """
    pp_size = get_parallel_state().pp.size

    from ..lora.utils import adapter_named_parameters

    adapter_ids = {id(param) for params in adapter_named_parameters(model) for param in params.values()}
    param_infos: dict[str, ParamInfo] = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):
        if id(param) in adapter_ids:
            continue
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(
                    param, "tensor_model_parallel", False
                ),  # config-access-exempt: tensor_model_parallel is optional backend-attached tensor metadata
                "partition_dim": getattr(
                    param, "partition_dim", -1
                ),  # config-access-exempt: partition_dim is optional backend-attached tensor metadata
                "partition_stride": getattr(
                    param, "partition_stride", 1
                ),  # config-access-exempt: partition_stride is optional backend-attached tensor metadata
                "parallel_mode": getattr(
                    param, "parallel_mode", None
                ),  # config-access-exempt: parallel_mode is optional backend-attached tensor metadata
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if gather_pp and pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=get_parallel_state().pp.group
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    # Duplicates across PP only exist for MTP virtual-PP layers.
                    assert args.backend.mtp_num_layers is not None
                    if param_infos[name].src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    infos = sorted(param_infos.values(), key=lambda info: info.name)
    non_expert_infos = [info for info in infos if not is_routed_expert_param(info.name)]
    expert_infos = [info for info in infos if is_routed_expert_param(info.name)]

    if gather_pp:
        _check_param_infos_consistent(non_expert_infos)

    return non_expert_infos, expert_infos


def _check_param_infos_consistent(param_infos: list[ParamInfo]) -> None:
    """Every rank must hold identical non-expert metadata once PP is gathered."""
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(obj=param_infos, object_list=all_param_info_list, group=get_gloo_group())
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"


def _gather_with_stride(
    param_partitions: list[torch.Tensor], partition_dim: int, partition_stride: int
) -> torch.Tensor:
    """Gather partitions respecting partition_stride (strided/interleaved TP sharding)."""
    if partition_stride == 1:
        return torch.cat(param_partitions, dim=partition_dim)
    # Interleaved (strided) partitioning, e.g. linear_fc1.weight under GLU/SwiGLU
    chunks_per_rank = [p.chunk(partition_stride, dim=partition_dim) for p in param_partitions]
    interleaved = [chunks_per_rank[r][s] for s in range(partition_stride) for r in range(len(param_partitions))]
    return torch.cat(interleaved, dim=partition_dim)


def is_routed_expert_param(name: str) -> bool:
    """Whether a Megatron param name belongs to the routed (expert-parallel) experts.

    Routed experts live under ".experts.", but shared experts may nest an inner
    ModuleList (e.g. Inkling's "mlp.shared_experts.experts.N.") whose params are
    regular-TP sharded and EP-replicated, so they must not match.
    """
    return ".experts." in name and ".shared_experts." not in name


def _is_unmarked_grouped_expert_weight(name: str, param: torch.nn.Parameter) -> bool:
    """TEGroupedLinear never marks its per-expert weight0..weightN, so Megatron fills in
    the defaults (tensor_model_parallel=False, partition_dim=-1) and the tensor claims to
    be unsharded. It is expert-TP sharded whenever etp > 1, so the gather must still run.
    """
    return (
        is_routed_expert_param(name)
        and ("linear_fc1.weight" in name or "linear_fc2.weight" in name)
        and not param.tensor_model_parallel
        and get_parallel_state().etp.size > 1
    )


def _check_and_fix_partition(args: Namespace, name: str, partition_stride: int, partition_dim: int) -> tuple[int, int]:
    """Validate partition_stride values for known parameter patterns.

    After Megatron-LM PR #2708, linear_fc1 correctly reports partition_stride=2
    (GLU/SwiGLU interleaved [gate, up]), so assert partition_stride==2 is removed.
    But TEGroupedLinear still does not set partition_stride/partition_dim correctly for grouped moe gemm
    """
    if "linear_fc1.weight" in name and (args.backend.swiglu or args.backend.quick_geglu):
        partition_stride = 2
        if partition_dim < 0:
            partition_dim = 0
    elif "linear_fc2.weight" in name:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
        if partition_dim <= 0:
            partition_dim = 1
    else:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
    return partition_stride, partition_dim


def all_gather_params_async(
    args: Namespace,
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    """
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group (skip expert_bias/non-TP/duplicated).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None, None))
            handles.append(None)
        elif getattr(
            param, "parallel_mode", None
        ) == "duplicated" or (  # config-access-exempt: parallel_mode is optional backend-attached tensor metadata
            not param.tensor_model_parallel and not _is_unmarked_grouped_expert_weight(info.name, param)
        ):
            gather_tasks.append((info, param.data, None, None, None, None))
            handles.append(None)
        else:
            # Start async all_gather
            if is_routed_expert_param(info.name):
                tp_size = get_parallel_state().etp.size
                tp_group = get_parallel_state().etp.group
            else:
                tp_size = get_parallel_state().tp.size
                tp_group = get_parallel_state().tp.group

            if tp_size <= 1:
                gather_tasks.append((info, param.data, None, None, None, None))
                handles.append(None)
                continue

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER)
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim, param.partition_stride))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    for handle in handles:
        if handle is not None:
            handle.wait()

    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim, partition_stride in gather_tasks:
        if handle is None:
            # No all_gather needed
            param = direct_param
        else:
            partition_stride, partition_dim = _check_and_fix_partition(
                args, info.name, partition_stride, partition_dim
            )
            param = _gather_with_stride(param_partitions, partition_dim, partition_stride)

        gathered_params.append(param)

    return gathered_params
