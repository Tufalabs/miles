import gc
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


_frozen_long_lived_objects = False


def clear_memory(clear_host_memory: bool = False, freeze_long_lived_objects: bool = False):
    """freeze_long_lived_objects: after this call's collection, take every object alive now out of later
    collections (gc.freeze). A process that calls this every step otherwise walks its whole heap each time: the
    model, the optimizer and the parallel state are tens of millions of objects and none of them is garbage.
    Objects created afterwards are collected as before."""
    global _frozen_long_lived_objects
    torch.accelerator.synchronize()
    gc.collect()
    if freeze_long_lived_objects and not _frozen_long_lived_objects:
        gc.freeze()
        _frozen_long_lived_objects = True
    torch.accelerator.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.accelerator.current_device_index()
    free, total = torch.accelerator.get_memory_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.accelerator.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.accelerator.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
