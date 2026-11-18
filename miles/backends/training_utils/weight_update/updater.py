"""Backend-neutral weight-update driver.

The updater owns the lifecycle: it builds the HF weight iterator against the
protocol's required placement, runs the engine session frame, streams base
buckets (senders transmit, other ranks join the gathers), and orchestrates
LoRA adapter pushes.
"""

import logging
import os
import random
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.sglang_utils.arguments import sglang_attention_dp_enabled
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager
from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.receiver_pool import ReceiverPools
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater
from miles.backends.training_utils.weight_update.session import (
    begin_weight_update,
    end_weight_update,
    maybe_pause_engines,
    maybe_resume_engines,
    register_lora_adapter,
    set_weight_version,
    weight_update_selector,
)
from miles.backends.training_utils.weight_update.utils import record_lora_checksums
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora.utils import LORA_ADAPTER_NAME
from miles.utils.timer import timer

if TYPE_CHECKING:
    from miles.ray.rollout.inference_controller import UpdatableEngines

logger = logging.getLogger(__name__)

# MILES_WEIGHT_SYNC_REFERENCE_CHECK=1: the first update sends the weights twice, first the way they were sent before
# stacked experts existed (one tensor per expert, converted and quantized one by one), then the configured way with
# the engines' parameters overwritten by random values in between, and requires the SHA-256 of every engine
# parameter to be the same after both. Costs two updates and two hashes of all weights, once.
_REFERENCE_CHECK = os.environ.get("MILES_WEIGHT_SYNC_REFERENCE_CHECK", "0") == "1"


class WeightUpdater:
    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        *,
        weights_getter: Callable[[], Mapping[str, torch.Tensor] | None],
        model_name: str,
        quantization_config: dict | None,
        iterator_factory: Callable,
        parallel_state: ParallelState,
        is_lora: bool,
        lora_sync_config: dict | None = None,
    ) -> None:
        self.args = args
        self.parallel_state = parallel_state
        self.protocol = get_weight_transfer_protocol(args)
        self.conn_status = ConnStatusManager()
        assert (
            not is_lora or self.protocol.supports_lora
        ), f"LoRA weight sync is not supported for {args.update_weight_transfer_mode!r} weight transfer."
        self._build_iterator = lambda: iterator_factory(
            args,
            model,
            required_placement=self.protocol.required_placement,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self._hf_weight_iterator = self._build_iterator()
        self._reference_checked = False
        self.weights_getter = weights_getter
        self.is_lora = is_lora
        if is_lora:
            assert lora_sync_config is not None
        self._lora_sync_config = lora_sync_config
        self._registered_adapters: set[str] = set()
        self._receiver_pools = ReceiverPools()

    def receiver_pool(self, pool_id: str):
        return self._receiver_pools.activate(self, pool_id)

    def mark_trainer_stale(self) -> None:
        self._receiver_pools.mark_trainer_stale(self)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
    ) -> None:
        self.protocol.connect(
            rollout_engines=rollout_engines,
            engine_gpu_counts=engine_gpu_counts,
            engine_gpu_offsets=engine_gpu_offsets,
            engine_cell_ids=engine_cell_ids,
            parallel_state=self.parallel_state,
            placement=self._hf_weight_iterator.placement,
            selector=weight_update_selector(
                speculative_algorithm=self.args.sglang.base_args["speculative_algorithm"],
                enable_mtp_training=self.args.enable_mtp_training,
            ),
        )
        assert self.protocol.is_sender is not None, "connect() must set is_sender"
        self._hf_weight_iterator.send_stacked_experts = self.protocol.send_stacked_experts
        self._registered_adapters.clear()

    def reconnect_if_needed(self, info: "UpdatableEngines") -> bool:
        if not self.conn_status.needs_reconnect(info.snapshot_cell_id_to_hashes):
            return False
        self.reconnect(info)
        return True

    def reconnect(self, info: "UpdatableEngines") -> None:
        self.connect_rollout_engines(
            info.rollout_engines,
            engine_gpu_counts=info.engine_gpu_counts,
            engine_gpu_offsets=info.engine_gpu_offsets,
            engine_cell_ids=info.engine_cell_ids,
        )
        self.conn_status.mark_reconnected(info.snapshot_cell_id_to_hashes)
        dist.barrier(group=get_gloo_group())

    def verify_engine_version(self, rollout_engines: Sequence[SGLangApiClient], *, weight_version: int) -> None:
        if not rollout_engines:
            return
        engine = random.choice(list(rollout_engines))
        engine_version = async_utils.run(engine.get_weight_version())
        if str(engine_version) != str(weight_version):
            raise RuntimeError(f"Weight version mismatch! Engine: {engine_version}, Updater: {weight_version}")

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear the protocol's metrics; the actor drains them onto the step log."""
        return self.protocol.pop_metrics()

    @torch.no_grad()
    def update_weights(self, weight_version: int) -> bool:
        """Run one weight sync: session frame + base-bucket stream + adapter pushes for LoRA."""
        protocol = self.protocol
        if not protocol.begin_sync(weight_version, self._iter_base_buckets):
            return False

        sync_base = not self.is_lora or protocol.needs_base_resync_for_lora
        adapters = self._get_updated_adapters()

        driver = dist.get_rank() == 0
        cell_updaters = list(protocol.cell_updaters_of_cell_id.values())
        if protocol.use_weight_update_session and driver:
            maybe_pause_engines(self.args, cell_updaters)
            self._register_new_lora_adapters(protocol.rollout_engines, adapters)
            begin_weight_update(
                cell_updaters,
                weight_update_selector(
                    speculative_algorithm=self.args.sglang.base_args["speculative_algorithm"],
                    enable_mtp_training=self.args.enable_mtp_training,
                ),
                sync_base=sync_base,
            )
        _mark_cells_errored_on_any_rank(cell_updaters)
        dist.barrier(group=get_gloo_group())

        checksums = {name: {} for name, _ in adapters} if self.is_lora and self.args.check_lora_weight_equal else None
        if checksums is not None:
            assert (
                self._hf_weight_iterator.placement.gather_pp
            ), "the LoRA checksum manifest is recorded on one rank, which must hold the full adapter"
        reference_check = _REFERENCE_CHECK and not self._reference_checked and sync_base and not adapters
        with timer("update_weights_implementation"):
            if reference_check:
                self._reference_checked = True
                reference_checksums = self._send_reference_weights(driver)
            self._send_weights(self._hf_weight_iterator, sync_base, adapters, checksums, driver)
            if reference_check and driver:
                _require_same_checksums(reference_checksums, self._engine_parameter_checksums())

        with timer("finalize_and_resume_engines"):
            protocol.finalize(weight_version)
            _mark_cells_errored_on_any_rank(cell_updaters)
            if protocol.use_weight_update_session and driver:
                end_weight_update(cell_updaters, expected_lora_checksums=checksums)
                set_weight_version(cell_updaters, weight_version)
                maybe_resume_engines(self.args, cell_updaters)
            _mark_cells_errored_on_any_rank(cell_updaters)
            dist.barrier(group=get_gloo_group())
        protocol.after_engines_resumed()
        return True

    def _send_weights(self, iterator, sync_base: bool, adapters, checksums, driver: bool) -> None:
        protocol = self.protocol
        pbar = tqdm(desc=f"[{protocol.group_name}] Update weights", total=0) if protocol.is_sender else None
        started = time.perf_counter()
        num_buckets = num_tensors = num_bytes = 0
        send_seconds = 0.0
        for bucket in iterator.iter_hf_weights(
            self.weights_getter(),
            include_base=sync_base,
            adapters=adapters,
            materialize=protocol.is_sender,
        ):
            if protocol.is_sender:
                if driver and checksums is not None:
                    record_lora_checksums(bucket, checksums)
                send_started = time.perf_counter()
                protocol.send_bucket(bucket)
                send_seconds += time.perf_counter() - send_started
                num_buckets += 1
                num_tensors += len(bucket)
                num_bytes += sum(tensor.nbytes for _name, tensor in bucket)
                pbar.update(1)
        protocol.after_base_weights()
        dist.barrier(group=get_gloo_group())
        if driver and num_buckets:
            # send = serialize, hand over and wait for the receiver to load; the rest is reading, converting,
            # quantizing and gathering on the trainer.
            total = time.perf_counter() - started
            logger.info(
                f"weight update stream: {num_buckets} buckets, {num_tensors} tensors, {num_bytes / 2**30:.2f} GiB, "
                f"{total:.3f} s = {total - send_seconds:.3f} s produce + {send_seconds:.3f} s send"
            )

    def _send_reference_weights(self, driver: bool) -> list | None:
        """MILES_WEIGHT_SYNC_REFERENCE_CHECK: send the weights one tensor per expert, each converted and quantized by
        itself, hash what the engines hold, and overwrite their parameters with random values, so that the update
        that follows has to write every byte again. Collective: every rank sends."""
        protocol = self.protocol
        reference_iterator = self._build_iterator()
        reference_iterator.stack_experts = False
        send_stacked_experts, protocol.send_stacked_experts = protocol.send_stacked_experts, False
        try:
            self._send_weights(reference_iterator, True, [], None, driver)
        finally:
            protocol.send_stacked_experts = send_stacked_experts
        reference_checksums = None
        if driver:
            reference_checksums = self._engine_parameter_checksums()
            selector = weight_update_selector(
                speculative_algorithm=self.args.sglang.base_args["speculative_algorithm"],
                enable_mtp_training=self.args.enable_mtp_training,
            )
            self._check_weights_on_engines({"action": "reset_tensors", "selector": selector})
        dist.barrier(group=get_gloo_group())
        return reference_checksums

    def _engine_parameter_checksums(self) -> list:
        """Per engine and rank, {parameter name: SHA-256 of its bytes}."""
        started = time.perf_counter()
        responses = self._check_weights_on_engines({"action": "raw_checksum", "selector": "all"})
        checksums = [[rank["checksums"] for rank in response["ranks"]] for response in responses]
        logger.info(
            f"weight sync reference check: SHA-256 of {sum(len(c) for engine in checksums for c in engine)} parameters "
            f"on {len(checksums)} engines in {time.perf_counter() - started:.1f} s; per engine "
            f"{[response['per_engine_checksum'][:16] for response in responses]}"
        )
        return checksums

    def _check_weights_on_engines(self, payload: dict) -> list:
        futures = [
            async_utils.submit(engine._make_request("weights_checker", payload, timeout=3600.0))
            for engine in self.protocol.rollout_engines
        ]
        return async_utils.wait_futures(futures)

    def _iter_base_buckets(self, *, materialize: bool):
        return self._hf_weight_iterator.iter_hf_weights(self.weights_getter(), materialize=materialize)

    def _get_updated_adapters(self) -> list[tuple[str, object]]:
        """``(lora_name, adapter_or_None)`` pairs for this sync; the push set is
        identical on every rank so the iterator's collectives align."""
        if not self.is_lora:
            return []
        return [(LORA_ADAPTER_NAME, None)]

    def _register_new_lora_adapters(self, rollout_engines, adapters: list[tuple[str, object]]) -> None:
        """Register adapters the current engine set has not seen, with their
        per-adapter config; eager so the engine validates rank before any bytes move."""
        for lora_name, adapter in adapters:
            if lora_name in self._registered_adapters:
                continue
            config = self._lora_sync_config
            if adapter is not None:
                config = config | {"r": adapter.rank, "lora_alpha": adapter.alpha}
            register_lora_adapter(
                rollout_engines,
                lora_name=lora_name,
                lora_config=config,
                pinned=sglang_attention_dp_enabled(
                    attn_dp_size=self.args.sglang.base_args["attn_dp_size"],
                    enable_dp_attention=self.args.sglang.base_args["enable_dp_attention"],
                    dp_size=self.args.sglang.base_args["dp_size"],
                ),
            )
            self._registered_adapters.add(lora_name)


def _require_same_checksums(reference: list, actual: list) -> None:
    different = [
        (engine, rank, name)
        for engine, (reference_ranks, actual_ranks) in enumerate(zip(reference, actual, strict=True))
        for rank, (expected, got) in enumerate(zip(reference_ranks, actual_ranks, strict=True))
        for name in sorted(expected.keys() | got.keys())
        if expected.get(name) != got.get(name)
    ]
    count = sum(len(checksums) for ranks in reference for checksums in ranks)
    if different:
        raise RuntimeError(
            f"weight sync reference check: {len(different)} of {count} engine parameters differ from the per-expert "
            f"update, e.g. {different[:8]}"
        )
    logger.info(
        f"weight sync reference check: all {count} engine parameters have the same SHA-256 after the configured "
        "update as after the per-expert update"
    )


def _mark_cells_errored_on_any_rank(cell_updaters: Sequence[_RolloutCellUpdater]) -> set[str]:
    group = get_gloo_group()
    errored_cell_ids_per_rank: list[list[str] | None] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(errored_cell_ids_per_rank, [u.cell_id for u in cell_updaters if u.is_errored], group=group)

    errored_cell_ids = {cell_id for cell_ids in errored_cell_ids_per_rank for cell_id in cell_ids}
    for cell_updater in cell_updaters:
        if cell_updater.cell_id in errored_cell_ids and not cell_updater.is_errored:
            cell_updater.mark_errored(
                RuntimeError(f"another trainer rank failed to update cell {cell_updater.cell_id}")
            )
    return errored_cell_ids
