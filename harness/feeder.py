"""When requests are sent.

Three modes, all engine-agnostic.

- Fixed rate (latency mode): input i arrives at start + i * period and is sent
  at once, whether or not earlier requests have returned. The world does not
  wait for the model. The deadline is the next arrival. Requests queue inside
  the engine, so none is ever dropped.
- Mailbox: the same schedule, but frames go through a depth-1 mailbox and the
  model takes the newest one when it is free. A frame that is replaced before
  it is taken is recorded as `overwritten`. This is the deadline-miss harness.
- Closed loop (throughput mode): the next request goes out the moment the
  previous one returns. This keeps the engine busy and measures how many
  requests per second it can finish.

The first two take a `send_one(req_id, t_arrival, deadline)` coroutine; the
mailbox mode takes `send_frame(frame)`. Both return the request's record.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from harness.mailbox import Frame, Mailbox
from harness.records import RequestRecord

SendOne = Callable[[int, float, float | None], Awaitable[RequestRecord]]
SendFrame = Callable[[Frame], Awaitable[RequestRecord]]


def arrival_times(start: float, period_s: float, n_requests: int) -> list[float]:
    """Arrival time of each input on a fixed-rate schedule."""
    return [start + i * period_s for i in range(n_requests)]


# Linux lets a sleep wake up late by up to 0.1% of its length: 30 ms on a 30 s
# sleep. Sleeping in two steps keeps the final, short step accurate.
FINAL_SLEEP_S = 0.05


async def sleep_until(target: float) -> None:
    """Sleep until a monotonic time, waking within about a millisecond of it."""
    remaining = target - time.monotonic()
    if remaining > 2 * FINAL_SLEEP_S:
        await asyncio.sleep(remaining - FINAL_SLEEP_S)
    await asyncio.sleep(max(0.0, target - time.monotonic()))


async def run_fixed_rate(
    send_one: SendOne, period_s: float, n_requests: int, start_delay_s: float = 1.0
) -> list[RequestRecord]:
    start = time.monotonic() + start_delay_s
    in_flight = []
    for req_id, t_arrival in enumerate(arrival_times(start, period_s, n_requests)):
        await sleep_until(t_arrival)
        deadline = t_arrival + period_s
        in_flight.append(asyncio.create_task(send_one(req_id, t_arrival, deadline)))
    return list(await asyncio.gather(*in_flight))


async def run_closed_loop(send_one: SendOne, duration_s: float) -> list[RequestRecord]:
    """One request in flight at a time, matching the server's single slot."""
    records = []
    stop_at = time.monotonic() + duration_s
    while time.monotonic() < stop_at:
        records.append(await send_one(len(records), time.monotonic(), None))
    return records


@dataclass
class EmitterFault:
    """A tick that fired late by more than the allowed fraction of the period."""

    frame_id: int
    late_s: float


@dataclass
class MailboxRun:
    records: list[RequestRecord] = field(default_factory=list)
    emitter_faults: list[EmitterFault] = field(default_factory=list)
    frames_emitted: int = 0


async def run_mailbox(
    send_frame: SendFrame,
    period_s: float,
    n_frames: int,
    tile_ids: list[str],
    overwritten_fields: dict,
    start_delay_s: float = 1.0,
    max_late_fraction: float = 0.1,
    stop_at: float | None = None,
    run: MailboxRun | None = None,
) -> MailboxRun:
    """Emit frames on a fixed schedule through a depth-1 mailbox to one consumer.

    Frame i carries tile_ids[i % len(tile_ids)]. `overwritten_fields` are the
    run's static record fields (engine, model, quant, power_mode) for frames
    that never reach the model. `stop_at` (monotonic) ends emission early.
    Pass a `run` to have it filled in place, so an interrupted cell keeps the
    records it already has.
    """
    mailbox = Mailbox()
    if run is None:
        run = MailboxRun()
    start = time.monotonic() + start_delay_s

    async def emit() -> None:
        for frame_id, t_arrival in enumerate(arrival_times(start, period_s, n_frames)):
            if stop_at is not None and t_arrival >= stop_at:
                break
            await sleep_until(t_arrival)
            t_obs = time.monotonic()
            if t_obs - t_arrival > max_late_fraction * period_s:
                run.emitter_faults.append(EmitterFault(frame_id, t_obs - t_arrival))
            frame = Frame(frame_id, tile_ids[frame_id % len(tile_ids)], t_obs, t_obs + period_s)
            run.frames_emitted += 1
            replaced = mailbox.put(frame)
            if replaced is not None:
                run.records.append(
                    RequestRecord.overwritten(
                        replaced.frame_id,
                        replaced.t_obs,
                        replaced.deadline,
                        replaced.tile_id,
                        **overwritten_fields,
                    )
                )
        mailbox.close()

    async def consume() -> None:
        while (frame := await mailbox.take()) is not None:
            run.records.append(await send_frame(frame))

    await asyncio.gather(emit(), consume())
    run.records.sort(key=lambda r: r.req_id)
    return run
