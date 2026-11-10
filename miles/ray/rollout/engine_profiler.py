"""SGLang torch-profiler windows at listed global rollout ids (--sglang-profile-rollout-ids).

Each window runs as a task beside the rollout, never in its way: it waits until the engine is
generating (a trace started while the engine is idle records the scheduler's idle loop, which with
stacks on is hundreds of MB), then asks the engine for --sglang-profile-num-steps forward steps.
The engine stops by itself after that many steps; ``finish`` stops a window the rollout ended
before it filled, so it can't run on into the idle time after the rollout.
"""

import asyncio
import logging
import time

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient

logger = logging.getLogger(__name__)

_POLL_SECONDS = 1.0
_DELAY_AFTER_GENERATION_STARTS_SECONDS = 5.0
_WAIT_FOR_GENERATION_SECONDS = 1800.0
# Advances on every decode step (SGLang's realtime token counter); the gauge below is the fallback for
# engines that don't export it, though it is only refreshed every --decode-log-interval steps.
_PROGRESS_COUNTER = "sglang:realtime_tokens_total"
_RUNNING_GAUGE = "sglang:num_running_reqs"


class RolloutEngineProfiler:
    def __init__(self, args):
        self._args = args
        self._tasks: set[asyncio.Task] = set()
        self._started: list[SGLangApiClient] = []

    def profiles(self, rollout_id: int) -> bool:
        return rollout_id in self._args.sglang_profile_rollout_ids

    def start(self, rollout_id: int, engines: list[SGLangApiClient]) -> None:
        """Start this rollout's window on the first engine; ``engines`` ordered by GPU offset."""
        if not engines:
            return
        output_dir = f"{self._args.sglang_profile_dir}/rollout_{rollout_id}/engine_0"
        task = asyncio.create_task(self._profile(engines[0], output_dir=output_dir))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def finish(self) -> None:
        """End every window at the end of a rollout: cancel the waiting ones, stop the running ones."""
        for task in list(self._tasks):
            task.cancel()
        started, self._started = self._started, []
        for engine in started:
            try:
                await engine.stop_profile()
                logger.info(f"Stopped the SGLang profile of {engine.server_url} at the end of the rollout")
            except Exception as e:  # the engine already stopped it after its num_steps: the usual case
                logger.info(f"SGLang profile of {engine.server_url} had already stopped ({type(e).__name__})")

    async def _profile(self, engine: SGLangApiClient, *, output_dir: str) -> None:
        args = self._args
        try:
            waited = await _wait_until_generating(engine)
            await asyncio.sleep(_DELAY_AFTER_GENERATION_STARTS_SECONDS)
            logger.info(
                f"Starting an SGLang profile of {args.sglang_profile_num_steps} steps on {engine.server_url} "
                f"({waited:.0f} s after the rollout began), to {output_dir}"
            )
            self._started.append(engine)
            await engine.start_profile(
                output_dir=output_dir,
                num_steps=args.sglang_profile_num_steps,
                activities=["CPU", "GPU"],
                with_stack=args.sglang_profile_with_stack,
                record_shapes=args.sglang_profile_record_shapes,
            )
        except asyncio.CancelledError:
            logger.info(f"The rollout ended before {engine.server_url} was profiled")
        except Exception:
            logger.exception(f"SGLang profile of {engine.server_url} failed; the run goes on")


async def _wait_until_generating(engine: SGLangApiClient) -> float:
    """Seconds until the engine's metrics show it decoding."""
    start = time.monotonic()
    previous = None
    while time.monotonic() - start < _WAIT_FOR_GENERATION_SECONDS:
        values = parse_metrics(await engine.get_metrics())
        if _PROGRESS_COUNTER in values:
            if previous is not None and values[_PROGRESS_COUNTER] > previous:
                return time.monotonic() - start
            previous = values[_PROGRESS_COUNTER]
        elif values.get(_RUNNING_GAUGE, 0) > 0:
            return time.monotonic() - start
        await asyncio.sleep(_POLL_SECONDS)
    raise TimeoutError(f"{engine.server_url} did not start generating within {_WAIT_FOR_GENERATION_SECONDS:.0f} s")


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text format to {metric name: value summed over its label sets}."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "}" in line:
            head, _, rest = line.rpartition("}")
            name = head.split("{", 1)[0]
        else:
            name, _, rest = line.partition(" ")
        try:
            value = float(rest.split()[0])
        except (IndexError, ValueError):
            continue
        values[name] = values.get(name, 0.0) + value
    return values
