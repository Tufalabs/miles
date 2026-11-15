from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.debug_utils.dump_schedule import is_dump_rollout
from miles.utils.types import RolloutBatch

_POLICY_LOSS_DUMP_COUNTER = 0
# The global rollout id whose training is running, for the loss calls that cannot see it; None outside
# training_rollout_scope (every call is then dumped, without an id).
_training_rollout_id: int | None = None


@contextmanager
def training_rollout_scope(rollout_id: int) -> Iterator[None]:
    """Mark the loss calls made inside this block as part of this rollout's training."""
    global _training_rollout_id
    _training_rollout_id = rollout_id
    try:
        yield
    finally:
        _training_rollout_id = None


def maybe_dump_policy_loss_debug(
    *,
    args: Namespace,
    batch: RolloutBatch,
    train_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor] | None,
    advantages: list[torch.Tensor],
    local_loss_masks: list[torch.Tensor],
    ppo_kl: torch.Tensor,
    pg_loss: torch.Tensor,
) -> None:
    dump_dir = args.dump_details
    rollout_id = _training_rollout_id
    if dump_dir is None or (
        rollout_id is not None and not is_dump_rollout(rollout_id, interval=args.dump_details_interval)
    ):
        return
    # TP peers compute the loss on identical data; CP ranks hold distinct token slices
    if get_parallel_state().tp.rank != 0:
        return

    global _POLICY_LOSS_DUMP_COUNTER
    counter = _POLICY_LOSS_DUMP_COUNTER
    _POLICY_LOSS_DUMP_COUNTER += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    path = Path(dump_dir) / "policy_loss_debug" / f"rank_{rank}_call_{counter}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu()

    samples = []
    for index, train_lp in enumerate(train_log_probs):
        sample = {
            "index": index,
            "total_length": batch["total_lengths"][index],
            "response_length": batch["response_lengths"][index],
            "train_log_probs": to_cpu(train_lp),
            "old_log_probs": to_cpu(old_log_probs[index]),
            "advantages": to_cpu(advantages[index]),
            "local_loss_mask": to_cpu(local_loss_masks[index]),
        }
        if rollout_log_probs is not None:
            sample["rollout_log_probs"] = to_cpu(rollout_log_probs[index])
            if train_lp.shape == rollout_log_probs[index].shape:
                sample["train_rollout_abs_diff"] = to_cpu((train_lp - rollout_log_probs[index]).abs())
        samples.append(sample)

    torch.save(
        {
            "rank": rank,
            "rollout_id": rollout_id,
            "call": counter,
            "samples": samples,
            "ppo_kl": to_cpu(ppo_kl),
            "pg_loss": to_cpu(pg_loss),
            "finite": {
                "ppo_kl": torch.isfinite(ppo_kl).all().item(),
                "pg_loss": torch.isfinite(pg_loss).all().item(),
                "train_log_probs": all(torch.isfinite(t).all().item() for t in train_log_probs),
                "old_log_probs": all(torch.isfinite(t).all().item() for t in old_log_probs),
                "advantages": all(torch.isfinite(t).all().item() for t in advantages),
            },
        },
        path,
    )


def maybe_dump_score_centering_debug(args: Namespace, tensors: dict[str, torch.Tensor]) -> None:
    """Per-token tensors of the score-centering loss on the dump steps, flat over this rank's response tokens
    (experiment, LLMT-83): the trainer's and the sampler's log-prob of the sampled token, the advantage, the
    active mask, and the candidate ids with both log-probs, to `<dump_details>/score_centering_debug/`."""
    dump_dir = args.dump_details
    rollout_id = _training_rollout_id
    if dump_dir is None or (rollout_id is not None and not is_dump_rollout(rollout_id, interval=args.dump_details_interval)):
        return
    if get_parallel_state().tp.rank != 0:
        return
    global _POLICY_LOSS_DUMP_COUNTER
    counter = _POLICY_LOSS_DUMP_COUNTER
    _POLICY_LOSS_DUMP_COUNTER += 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    path = Path(dump_dir) / "score_centering_debug" / f"rank_{rank}_call_{counter}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, value in tensors.items():
        value = value.detach()
        if value.dtype == torch.bool:
            saved[name] = value.cpu()
        elif value.dtype in (torch.int32, torch.int64):
            saved[name] = value.to(torch.int32).cpu()
        else:
            saved[name] = value.to(torch.float16).cpu()
    torch.save({"rank": rank, "rollout_id": rollout_id, "call": counter, **saved}, path)
