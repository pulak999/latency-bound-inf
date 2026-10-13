import asyncio
import time

from harness.feeder import arrival_times, run_closed_loop, run_fixed_rate, sleep_until


def test_arrival_times_are_evenly_spaced():
    assert arrival_times(start=100.0, period_s=30.0, n_requests=3) == [100.0, 130.0, 160.0]


def test_fixed_rate_sends_on_schedule_even_when_replies_are_slow():
    period = 0.05
    sends = []

    async def slow_send(req_id, t_arrival, deadline):
        sends.append((req_id, t_arrival, time.monotonic(), deadline))
        await asyncio.sleep(period * 3)  # each reply takes three periods
        return req_id

    results = asyncio.run(run_fixed_rate(slow_send, period, n_requests=5, start_delay_s=0.0))

    assert results == [0, 1, 2, 3, 4]
    for req_id, t_arrival, t_sent, deadline in sends:
        assert abs(t_sent - t_arrival) < 0.02, f"request {req_id} waited for an earlier reply"
        assert deadline == t_arrival + period


def test_closed_loop_keeps_one_request_in_flight():
    in_flight = 0
    max_in_flight = 0

    async def send(req_id, t_arrival, deadline):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        assert deadline is None
        return req_id

    results = asyncio.run(run_closed_loop(send, duration_s=0.1))
    assert max_in_flight == 1
    assert results == list(range(len(results))) and len(results) >= 5


def test_sleep_until_wakes_on_time_after_a_long_wait():
    # A plain 3 s sleep can wake up to 3 ms late on Linux; sleep_until should not.
    async def overshoot_ms():
        target = time.monotonic() + 3.0
        await sleep_until(target)
        return (time.monotonic() - target) * 1000

    assert asyncio.run(overshoot_ms()) < 1.0
