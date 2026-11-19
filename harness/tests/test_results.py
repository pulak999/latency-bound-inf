import json

import pytest

from harness import results
from harness.collect import collect
from harness.device import PowerSample, read_power_csv
from harness.records import RequestRecord, read_records, write_records

STATIC = {"engine": "llama.cpp@abc1234", "model": "m", "quant": "q", "power_mode": "120W"}


def make_cell(cell_dir, n_background_lines=2):
    """A raw cell directory as the controller writes it, with awkward values on purpose."""
    cell_dir.mkdir()
    records = [
        RequestRecord(
            req_id=0,
            t_arrival=100.000000001,
            t_send=100.1,
            t_first_token=101.23456789012345,
            t_last_token=101.23456789012345,
            deadline=106.0,
            n_prompt_tokens=301,
            n_output_tokens=1,
            output_text='keep, "maybe"\nno',  # commas, quotes and a newline survive CSV
            engine_prompt_ms=1392.28,
            engine_decode_ms=0.001,
            co_tenants=[{"pid": 7, "name": "/bin/x", "used_mib": "100"}],
            tile_id="00007",
            t_scheduled=99.9999,
            **STATIC,
        ),
        RequestRecord.overwritten(1, t_arrival=106.0, deadline=112.0, tile_id="00008", **STATIC),
        RequestRecord(
            req_id=2,
            t_arrival=112.0,
            t_send=112.0,
            t_first_token=None,
            t_last_token=113.0,
            deadline=118.0,
            n_prompt_tokens=None,
            n_output_tokens=None,
            output_text="",
            engine_prompt_ms=None,
            engine_decode_ms=None,
            engine_stages_ms={"inference": 12.5},
            n_actions=50,
            **STATIC,
        ),
    ]
    write_records(cell_dir / "records.jsonl", records)
    (cell_dir / "power.csv").write_text(
        "t,gpu_w,board_w,gpu_mhz\n100.0,2.5,22.0,315\n100.01,nan,nan,\n"
    )
    (cell_dir / "gpu_util.csv").write_text("t,gpu_util_pct\n100.0,0\n100.1,[N/A]\n")
    lines = [
        {
            "i": i,
            "t_due": None if i else 99.0,
            "t_send": 99.0 + i,
            "t_done": 99.5 + i,
            "engine_prompt_ms": 400.0,
            "error": None,
        }
        for i in range(n_background_lines)
    ]
    (cell_dir / "background-1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    manifest = {
        "cell_id": "n2-bursty",
        "cadence_s": 6.0,
        "tenancy": {"tenants": 2, "load": "bursty", "sharing": "timeslice"},
    }
    (cell_dir / "manifest.json").write_text(json.dumps(manifest))
    return records


def test_collect_then_load_gives_back_exactly_what_was_recorded(tmp_path):
    cell_dir = tmp_path / "n2-bursty-20261119-120000"
    original = make_cell(cell_dir)
    out = tmp_path / "results" / "2026-11-19-test"
    collect([cell_dir], out)

    loaded = results.load(out)
    name = cell_dir.name
    assert loaded.records(name) == read_records(cell_dir / "records.jsonl") == original
    power = loaded.power(name)
    assert power[0] == PowerSample(t=100.0, gpu_w=2.5, board_w=22.0, gpu_mhz=315.0)
    assert len(power) == len(read_power_csv(cell_dir / "power.csv")) == 2
    assert loaded.gpu_util(name) == [(100.0, 0.0)]  # unreadable samples are skipped on load
    (neighbour,) = loaded.neighbour_requests(name).values()
    assert [r["t_send"] for r in neighbour] == [99.0, 100.0]
    assert neighbour[0]["t_scheduled"] == 99.0 and neighbour[1]["t_scheduled"] is None
    cell = loaded.cell(name)
    assert (cell["tenants"], cell["load"], cell["cadence_s"]) == (2, "bursty", 6.0)
    assert loaded.manifest(name)["cell_id"] == "n2-bursty"


def test_records_file_stores_no_derived_values(tmp_path):
    cell_dir = tmp_path / "cell"
    make_cell(cell_dir)
    first = json.loads((cell_dir / "records.jsonl").read_text().splitlines()[0])
    assert not {"latency_s", "missed", "feeder_late_s"} & set(first)


def test_collect_refuses_unknown_background_fields(tmp_path):
    cell_dir = tmp_path / "cell"
    make_cell(cell_dir)
    with (cell_dir / "background-1.jsonl").open("a") as f:
        f.write(json.dumps({"i": 9, "t_send": 1.0, "t_done": 2.0, "new_field": 1}) + "\n")
    with pytest.raises(ValueError, match="new_field"):
        collect([cell_dir], tmp_path / "out")


def test_collect_checks_every_input_before_writing(tmp_path):
    cell_dir = tmp_path / "cell"
    make_cell(cell_dir)
    stray = tmp_path / "run.log"
    stray.write_text("")
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="run.log"):
        collect([cell_dir, stray], out)
    assert not out.exists()
