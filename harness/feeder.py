"""When requests are sent.

Two modes, both engine-agnostic. They take a `send_one(req_id, t_arrival, deadline)`
coroutine that sends one request and returns its record.

- Fixed rate (latency mode): input i arrives at start + i * period and is sent
  at once, whether or not earlier requests have returned. The world does not
  wait for the model. The deadline is the next arrival.
- Closed loop (throughput mode): the next request goes out the moment the
  previous one returns. This keeps the engine busy and measures how many
  requests per second it can finish.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

from harness.records import RequestRecord

SendOne = Callable[[int, float, float | None], Awaitable[RequestRecord]]


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
