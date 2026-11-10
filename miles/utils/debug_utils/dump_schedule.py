def is_dump_rollout(rollout_id: int, *, interval: int) -> bool:
    """Whether the debug dumps of this training rollout are written (--dump-details-interval).

    Keyed on the global rollout id, so a resumed attempt keeps the schedule of the first one.
    """
    return rollout_id % interval == 0
