from harness.runner import mode_arguments


def test_latency_mode_defaults_to_the_stack_period():
    args = mode_arguments("test", default_period_s=1.0).parse_args(["latency"])
    assert args.mode == "latency"
    assert args.period_s == 1.0
    assert args.n_requests == 100


def test_throughput_mode_takes_a_duration():
    args = mode_arguments("test", default_period_s=30.0).parse_args(
        ["throughput", "--duration", "60"]
    )
    assert args.mode == "throughput"
    assert args.duration_s == 60.0
