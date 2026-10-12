import math

import pytest

from harness.device import PowerSample
from harness.records import RequestRecord
from harness.report import energy_joules, percentiles_ms, summarize, wilson_interval


def make_record(req_id, t_arrival, latency_s, deadline=None, error=None):
    t_last = None if error else t_arrival + latency_s
    return RequestRecord(
        req_id=req_id,
        t_arrival=t_arrival,
        t_send=t_arrival,
        t_first_token=t_last,
        t_last_token=t_last,
        deadline=deadline,
        n_prompt_tokens=288,
        n_output_tokens=4,
        output_text="Field",
        engine_prompt_ms=None,
        engine_decode_ms=None,
        engine="test",
        model="test",
        quant="test",
        power_mode="120W",
        error=error,
    )


def test_wilson_interval_known_values():
    # Reference values from the closed-form Wilson score interval at z = 1.96.
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)


def test_wilson_interval_no_trials_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_percentiles_ms_on_a_known_tail():
    # 99 requests at 1 s and one at 2 s.
    stats = percentiles_ms([1.0] * 99 + [2.0])
    assert stats["p50"] == 1000.0
    assert stats["max"] == 2000.0
    assert 1000.0 < stats["p99"] <= 2000.0


def test_energy_of_constant_power_is_power_times_time():
    samples = [PowerSample(t=i * 0.01, gpu_w=20.0, board_w=40.0) for i in range(201)]
    assert energy_joules(samples, 0.5, 1.5, "gpu_w") == pytest.approx(20.0)
    assert energy_joules(samples, 0.5, 1.5, "board_w") == pytest.approx(40.0)


def test_energy_with_no_samples_in_window_is_nan():
    samples = [PowerSample(t=0.0, gpu_w=20.0, board_w=40.0)]
    assert math.isnan(energy_joules(samples, 5.0, 6.0, "gpu_w"))


def test_summarize_counts_late_and_failed_requests_as_misses():
    records = [
        make_record(0, t_arrival=0.0, latency_s=0.5, deadline=1.0),  # on time
        make_record(1, t_arrival=1.0, latency_s=1.5, deadline=2.0),  # late
        make_record(2, t_arrival=2.0, latency_s=0.0, deadline=3.0, error="timeout"),  # failed
    ]
    power = [PowerSample(t=i * 0.01, gpu_w=10.0, board_w=30.0) for i in range(400)]
    summary = summarize(records, power)

    assert summary["n_errors"] == 1
    assert summary["deadline_misses"]["missed"] == 2
    assert summary["deadline_misses"]["of"] == 3
    assert summary["latency_ms"]["max"] == pytest.approx(1500.0)
    # Two answered requests between first send (0.0) and last answer (2.5).
    assert summary["throughput_rps"] == pytest.approx(2 / 2.5, abs=1e-4)
    assert summary["energy_per_request_j"]["gpu"] == pytest.approx(10.0, abs=0.1)


def test_summarize_throughput_mode_has_no_deadline_section():
    records = [make_record(i, t_arrival=float(i), latency_s=1.0) for i in range(5)]
    power = [PowerSample(t=i * 0.01, gpu_w=10.0, board_w=30.0) for i in range(700)]
    summary = summarize(records, power)
    assert "deadline_misses" not in summary
    assert summary["throughput_rps"] == pytest.approx(1.0)
