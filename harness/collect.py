"""Copy raw cell directories into a results folder, losing nothing that was measured.

    uv run python -m harness.collect RAW_CELL_DIR... -o results/<YYYY-MM-DD>-<question>

Reads each cell's manifest.json, records.jsonl, background-*.jsonl, power.csv
and gpu_util.csv and writes the tables described in harness/results.py. Nothing
is computed: no latency, no miss flag, no percentile. Values are copied as
recorded (floats round-trip exactly), and an unknown field is an error rather
than being dropped.

Rewrites the folder's tables and manifests/ from the cells given, so pass every
cell of the experiment each time. README.md and figures/ are left alone.
Engine and server logs and tegrastats.log stay in the raw directory (`raw_dir`).
"""

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from harness import results
from harness.records import read_records

# A background tenant's line (harness/background.py) -> the requests.csv column it fills.
BACKGROUND_FIELDS = {
    "i": "req_id",
    "t_due": "t_scheduled",
    "t_send": "t_send",
    "t_done": "t_last_token",
    "engine_prompt_ms": "engine_prompt_ms",
    "error": "error",
}


def cell_row(cell_dir: Path, manifest: dict, static: dict) -> dict:
    """One cells.csv row. `static` is the engine, model, quant and power mode of the records."""
    tenancy = manifest.get("tenancy", {})
    return {
        "cell": cell_dir.name,
        "cell_id": manifest.get("cell_id"),
        "raw_dir": str(cell_dir.resolve()),
        "wall_start": manifest.get("wall_start"),
        "wall_end": manifest.get("wall_end"),
        "harness_commit": manifest.get("harness_commit"),
        "config_path": manifest.get("config_path"),
        "config_sha256": manifest.get("config_sha256"),
        "engine_kind": manifest.get("engine_kind"),
        "model_file_sha256": manifest.get("model_file_sha256"),
        "cadence_s": manifest.get("cadence_s"),
        "frames_requested": manifest.get("frames_requested"),
        "warmup_frames": manifest.get("warmup_frames"),
        "tenants": tenancy.get("tenants", 1),
        "load": tenancy.get("load"),
        "sharing": tenancy.get("sharing"),
        "sms_per_tenant": tenancy.get("sms_per_tenant"),
        "placement": tenancy.get("placement"),
        "background_period_s": tenancy.get("background_period_s"),
        "burst_size": tenancy.get("burst_size"),
        "aborted": manifest.get("aborted"),
        **static,
    }


def measured_rows(cell: str, cell_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """requests.csv and co_tenants.csv rows of the measured tenant, and its per-cell fields."""
    records = read_records(cell_dir / "records.jsonl")
    statics = {tuple(getattr(r, k) for k in results.PER_CELL_RECORD_FIELDS) for r in records}
    if len(statics) > 1:
        raise ValueError(f"{cell}: engine/model/quant/power_mode change between records")
    static = (
        dict(zip(results.PER_CELL_RECORD_FIELDS, statics.pop(), strict=True)) if records else {}
    )

    requests, co_tenants = [], []
    for record in records:
        fields = asdict(record)
        for key in results.PER_CELL_RECORD_FIELDS:
            del fields[key]
        for co_tenant in fields.pop("co_tenants"):
            co_tenants.append({"cell": cell, "req_id": record.req_id, **co_tenant})
        if fields["engine_stages_ms"] is not None:
            fields["engine_stages_ms"] = json.dumps(fields["engine_stages_ms"])
        requests.append({"cell": cell, "tenant": results.MEASURED_TENANT, **fields})
    return requests, co_tenants, static


def background_rows(cell: str, cell_dir: Path) -> list[dict]:
    """requests.csv rows of the background tenants; background-N.jsonl is tenant N."""
    rows = []
    for log in sorted(cell_dir.glob("background-*.jsonl")):
        tenant = int(log.stem.removeprefix("background-"))
        for line in log.read_text().splitlines():
            request = json.loads(line)
            unknown = set(request) - set(BACKGROUND_FIELDS)
            if unknown:
                raise ValueError(f"{log}: fields this collector does not know: {sorted(unknown)}")
            row = {BACKGROUND_FIELDS[k]: v for k, v in request.items()}
            row["outcome"] = "ok" if row.get("error") is None else "error"
            rows.append({"cell": cell, "tenant": tenant, **row})
    return rows


def sample_rows(cell: str, path: Path, columns: list[str]) -> list[dict]:
    """A sampler CSV, copied as text with the cell prepended. Missing columns stay empty."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        unknown = set(reader.fieldnames or []) - set(columns)
        if unknown:
            raise ValueError(f"{path}: columns this collector does not know: {sorted(unknown)}")
        return [{"cell": cell, **row} for row in reader]


def collect(cell_dirs: list[Path], out: Path) -> None:
    names = [d.name for d in cell_dirs]
    if len(set(names)) != len(names):
        raise ValueError("two cell directories have the same name")
    not_cells = [str(d) for d in cell_dirs if not (d / "manifest.json").exists()]
    if not_cells:  # checked before anything in `out` is touched
        raise FileNotFoundError(f"not cell directories (no manifest.json): {not_cells}")
    cells, requests, co_tenants, power, gpu_util = [], [], [], [], []
    manifests = out / results.MANIFESTS
    if manifests.exists():
        shutil.rmtree(manifests)
    manifests.mkdir(parents=True)

    for cell_dir in cell_dirs:
        cell = cell_dir.name
        manifest = json.loads((cell_dir / "manifest.json").read_text())
        shutil.copyfile(cell_dir / "manifest.json", manifests / f"{cell}.json")

        measured, seen, static = measured_rows(cell, cell_dir)
        cells.append(cell_row(cell_dir, manifest, static))
        requests += measured + background_rows(cell, cell_dir)
        co_tenants += seen
        power += sample_rows(cell, cell_dir / "power.csv", results.POWER_COLUMNS)
        gpu_util += sample_rows(cell, cell_dir / "gpu_util.csv", results.GPU_UTIL_COLUMNS)

    results.write_csv(out / results.CELLS, results.CELL_COLUMNS, cells)
    results.write_csv(out / results.REQUESTS, results.REQUEST_COLUMNS, requests)
    results.write_csv(out / results.CO_TENANTS, results.CO_TENANT_COLUMNS, co_tenants)
    results.write_csv(out / results.POWER, results.POWER_COLUMNS, power)
    results.write_csv(out / results.GPU_UTIL, results.GPU_UTIL_COLUMNS, gpu_util)
    (out / results.FIGURES).mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cell_dirs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True, help="results folder")
    args = parser.parse_args()
    collect(args.cell_dirs, args.out)
    print(f"wrote {len(args.cell_dirs)} cells to {args.out}")


if __name__ == "__main__":
    main()
