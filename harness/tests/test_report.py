import math

import pytest

from harness.device import DeviceSample
from harness.records import RequestRecord
from harness.report import (
    DeviceTrace,
    percentiles_ms,
    stages_ms,
    summarize,
    tail_by_stage,
    wilson_interval,
)


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
    trace = DeviceTrace([DeviceSample(t=i * 0.01, gpu_w=20.0, board_w=40.0) for i in range(201)])
    assert trace.energy_j("gpu", 0.5, 1.5) == pytest.approx(20.0)
    assert trace.energy_j("board", 0.5, 1.5) == pytest.approx(40.0)


def test_energy_with_no_samples_in_window_is_nan():
    trace = DeviceTrace([DeviceSample(t=0.0, gpu_w=20.0, board_w=40.0)])
    assert math.isnan(trace.energy_j("gpu", 5.0, 6.0))


def test_summarize_counts_late_and_failed_requests_as_misses():
    records = [
        make_record(0, t_arrival=0.0, latency_s=0.5, deadline=1.0),  # on time
        make_record(1, t_arrival=1.0, latency_s=1.5, deadline=2.0),  # late
        make_record(2, t_arrival=2.0, latency_s=0.0, deadline=3.0, error="timeout"),  # failed
    ]
    power = [DeviceSample(t=i * 0.01, gpu_w=10.0, board_w=30.0) for i in range(400)]
    summary = summarize(records, DeviceTrace(power))

    assert summary["n_errors"] == 1
    assert summary["deadline_misses"]["missed"] == 2
    assert summary["deadline_misses"]["of"] == 3
    assert summary["latency_ms"]["max"] == pytest.approx(1500.0)
    # Two answered requests between first send (0.0) and last answer (2.5).
    assert summary["throughput_rps"] == pytest.approx(2 / 2.5, abs=1e-4)
    assert summary["energy_per_request_j"]["gpu"] == pytest.approx(10.0, abs=0.1)


def test_summarize_throughput_mode_has_no_deadline_section():
    records = [make_record(i, t_arrival=float(i), latency_s=1.0) for i in range(5)]
    power = [DeviceSample(t=i * 0.01, gpu_w=10.0, board_w=30.0) for i in range(700)]
    summary = summarize(records, DeviceTrace(power))
    assert "deadline_misses" not in summary
    assert summary["throughput_rps"] == pytest.approx(1.0)


def test_summarize_robot_records_without_tokens():
    # A policy returns its whole chunk at once: no first token, stage timings instead.
    records = []
    for i in range(4):
        record = make_record(i, t_arrival=float(i), latency_s=0.2, deadline=float(i) + 1.0)
        record.t_first_token = None
        record.engine_stages_ms = {"preprocess": 10.0, "inference": 150.0, "postprocess": 5.0}
        records.append(record)
    power = [DeviceSample(t=i * 0.01, gpu_w=10.0, board_w=30.0) for i in range(500)]
    summary = summarize(records, DeviceTrace(power))

    assert "time_to_first_token_ms" not in summary
    assert summary["engine_ms"]["p50"] == 165.0
    assert summary["stages_ms"]["outside_engine"]["p50"] == pytest.approx(35.0)
    assert summary["stages_ms"]["inference"]["p50"] == 150.0
    assert summary["deadline_misses"]["missed"] == 0


def test_stages_add_up_to_the_latency():
    record = make_record(0, t_arrival=10.0, latency_s=0.25)
    record.t_send = 10.01  # waited 10 ms to be sent
    record.engine_stages_ms = {"preprocess": 20.0, "inference": 200.0, "postprocess": 5.0}
    stages = stages_ms(record)
    assert stages["wait"] == pytest.approx(10.0)
    assert stages["outside_engine"] == pytest.approx(15.0)
    assert sum(stages.values()) == pytest.approx(record.latency_s * 1000)


def test_tail_is_attributed_to_the_stage_that_grew():
    # 99 requests at 200 ms of inference; one where inference took 300 ms.
    records = []
    for i in range(100):
        inference_ms = 300.0 if i == 42 else 200.0
        record = make_record(i, t_arrival=float(i), latency_s=(inference_ms + 10.0) / 1000)
        record.engine_stages_ms = {"preprocess": 10.0, "inference": inference_ms}
        records.append(record)
    tail = tail_by_stage(records)
    assert tail["slowest_1pct_mean_minus_median"] == pytest.approx(100.0)
    assert tail["by_stage"]["inference"] == pytest.approx(100.0)
    assert tail["by_stage"]["preprocess"] == pytest.approx(0.0)


def test_gpu_clock_lookups():
    # Clock low until t = 1 s, then high.
    trace = DeviceTrace(
        [
            DeviceSample(t=i * 0.01, gpu_w=10.0, board_w=30.0, gpu_mhz=315.0 if i < 100 else 1386.0)
            for i in range(300)
        ]
    )
    assert trace.gpu_clock_at(0.995) == 315.0
    assert trace.gpu_clock_at(1.0) == 1386.0
    assert trace.median_gpu_clock(0.5, 2.5) == 1386.0


def test_runs_without_clock_samples_have_no_clock_section():
    records = [make_record(i, t_arrival=float(i), latency_s=1.0) for i in range(5)]
    old_samples = [DeviceSample(t=i * 0.01, gpu_w=10.0, board_w=30.0) for i in range(700)]
    assert "gpu_clock_mhz" not in summarize(records, DeviceTrace(old_samples))
