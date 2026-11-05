import json

from harness.sweep import SWEEPS, Step, load_steps


def test_steps_build_the_run_command_line():
    assert Step(duration_s=200).mode_arguments() == ["throughput", "--duration", "200"]
    assert Step(period_s=0.5, n_requests=1000).mode_arguments() == [
        "latency",
        "--period",
        "0.5",
        "--n-requests",
        "1000",
    ]


def test_every_sweep_has_a_back_to_back_reference():
    for _, steps in SWEEPS.values():
        assert any(step.period_s is None for step in steps)


def test_load_steps_puts_back_to_back_first_then_by_period(tmp_path):
    def fake_run(name, period_s):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({"period_s": period_s}))
        summary = {
            "n_requests": 10,
            "latency_ms": {"p50": 1.0, "p99": 2.0},
            "throughput_rps": 1.0,
            "tail_ms": {"by_stage": {}},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))

    fake_run("b-latency", 3.0)
    fake_run("a-latency", 0.2)
    fake_run("c-throughput", None)
    assert [row["period_s"] for row in load_steps(tmp_path)] == [None, 0.2, 3.0]
