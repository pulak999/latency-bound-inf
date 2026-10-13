from harness.records import RequestRecord, read_records, write_records


def test_records_round_trip_through_jsonl(tmp_path):
    original = [
        RequestRecord(
            req_id=0,
            t_arrival=1.0,
            t_send=1.001,
            t_first_token=2.0,
            t_last_token=2.1,
            deadline=31.0,
            n_prompt_tokens=288,
            n_output_tokens=4,
            output_text="Field",
            engine_prompt_ms=1400.0,
            engine_decode_ms=75.0,
            engine="llama.cpp@358a420",
            model="gemma-3-4b-it",
            quant="QAT Q4_0",
            power_mode="120W",
            co_tenants=[{"pid": 1, "name": "other", "used_mib": "100"}],
        )
    ]
    path = tmp_path / "records.jsonl"
    write_records(path, original)
    assert read_records(path) == original


def test_latency_and_missed_follow_the_deadline():
    common = dict(
        req_id=0,
        t_arrival=10.0,
        t_send=10.0,
        t_first_token=None,
        n_prompt_tokens=None,
        n_output_tokens=None,
        output_text="",
        engine_prompt_ms=None,
        engine_decode_ms=None,
        engine="e",
        model="m",
        quant="q",
        power_mode="p",
    )
    on_time = RequestRecord(**common, t_last_token=12.0, deadline=40.0)
    late = RequestRecord(**common, t_last_token=41.0, deadline=40.0)
    never = RequestRecord(**common, t_last_token=None, deadline=40.0)
    no_deadline = RequestRecord(**common, t_last_token=12.0, deadline=None)

    assert on_time.latency_s == 2.0 and on_time.missed is False
    assert late.missed is True
    assert never.latency_s is None and never.missed is True
    assert no_deadline.missed is None


def test_records_written_before_the_robot_fields_still_load(tmp_path):
    # A line in the original Stack A format, without n_actions or engine_stages_ms.
    old_line = (
        '{"req_id": 0, "t_arrival": 1.0, "t_send": 1.0, "t_first_token": 2.0, '
        '"t_last_token": 2.1, "deadline": 31.0, "n_prompt_tokens": 288, "n_output_tokens": 4, '
        '"output_text": "Field", "engine_prompt_ms": 1400.0, "engine_decode_ms": 75.0, '
        '"engine": "llama.cpp@358a420", "model": "gemma-3-4b-it", "quant": "QAT Q4_0", '
        '"power_mode": "120W", "co_tenants": [], "error": null, '
        '"latency_s": 1.1, "missed": false}\n'
    )
    path = tmp_path / "records.jsonl"
    path.write_text(old_line)
    (record,) = read_records(path)
    assert record.n_actions is None
    assert record.engine_total_ms == 1475.0


def test_engine_total_prefers_named_stages():
    record = RequestRecord(
        req_id=0,
        t_arrival=0.0,
        t_send=0.0,
        t_first_token=None,
        t_last_token=0.2,
        deadline=None,
        n_prompt_tokens=None,
        n_output_tokens=None,
        output_text="",
        engine_prompt_ms=None,
        engine_decode_ms=None,
        engine="lerobot",
        model="m",
        quant="bfloat16",
        power_mode="120W",
        n_actions=50,
        engine_stages_ms={"preprocess": 5.0, "inference": 180.0, "postprocess": 1.0},
    )
    assert record.engine_total_ms == 186.0
