import torch
import torch.distributed as dist

_GB = 1024**3

# The allocator's lifetime counters at the previous call, to report their change per step. They start at 0,
# so the first step also counts retries and OOMs from before it (model setup).
_previous_counters = {"num_alloc_retries": 0, "num_ooms": 0}


def step_memory_metrics(*, reduce_over_ranks: bool) -> dict[str, float]:
    """GPU memory of the training step since the previous call, then resets the peaks.

    Peaks are the largest over the ranks; ``alloc_retries`` (the allocator freed its cache and retried an
    allocation, a sign of memory pressure) and ``ooms`` are summed over them. Every rank must call it when
    ``reduce_over_ranks``; otherwise it reports this rank's values. Nothing without CUDA (CPU tests).
    """
    if not torch.cuda.is_available():
        return {}
    stats = torch.cuda.memory_stats()
    device = torch.cuda.current_device()
    peaks = torch.tensor(
        [stats.get("allocated_bytes.all.peak", 0), stats.get("reserved_bytes.all.peak", 0)],
        dtype=torch.float64,
        device=device,
    )
    counts = torch.tensor(
        [stats.get(key, 0) - previous for key, previous in _previous_counters.items()],
        dtype=torch.float64,
        device=device,
    )
    _previous_counters.update({key: stats.get(key, 0) for key in _previous_counters})
    torch.cuda.reset_peak_memory_stats()
    if reduce_over_ranks:
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    peak_allocated, peak_reserved = peaks.tolist()
    alloc_retries, ooms = counts.tolist()
    return {
        "perf/peak_allocated_gb": peak_allocated / _GB,
        "perf/peak_reserved_gb": peak_reserved / _GB,
        "perf/alloc_retries": alloc_retries,
        "perf/ooms": ooms,
    }
