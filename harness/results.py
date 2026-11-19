"""The results folder: raw measurements of one experiment, as tidy CSV tables.

A results folder holds what was measured, never what was computed from it.
p50, p99, miss rates, energy and every other statistic are computed when a
figure or table is made, from these files, so any new statistic can be computed
later from the same data. `harness.collect` writes a folder from raw cell
directories; `load` reads one back.

    results/<YYYY-MM-DD>-<question>/
      README.md        the question, the answer, figures, caveats, how to rebuild
      cells.csv        one row per cell: what was run (settings and provenance)
      requests.csv     one row per request, the measured tenant and its neighbours
      co_tenants.csv   one row per other GPU process seen at a measured request's send
      power.csv        one row per power sample (GPU and board watts, GPU clock)
      gpu_util.csv     one row per GPU load sample
      manifests/       each cell's manifest.json, verbatim
      figures/         made from the tables above, never edited by hand

Every table has a `cell` column (the raw directory's name) to join on.
Times are the raw `time.monotonic()` seconds the harness recorded: comparable
within a cell, not across cells. Subtract a cell's first `t_arrival` to plot.
A value the harness did not record is an empty field.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from harness.device import PowerSample
from harness.records import RequestRecord

CELLS = "cells.csv"
REQUESTS = "requests.csv"
CO_TENANTS = "co_tenants.csv"
POWER = "power.csv"
GPU_UTIL = "gpu_util.csv"
MANIFESTS = "manifests"
FIGURES = "figures"

MEASURED_TENANT = 0  # `tenant` of the measured tenant's rows; neighbours are 1, 2, ...

# Settings and provenance, flattened from the manifest. Anything else the
# manifest holds (prompt, device snapshot, partitions) stays in manifests/.
CELL_COLUMNS = [
    "cell",
    "cell_id",
    "raw_dir",
    "wall_start",
    "wall_end",
    "harness_commit",
    "config_path",
    "config_sha256",
    "engine_kind",
    "engine",
    "model",
    "quant",
    "model_file_sha256",
    "power_mode",
    "cadence_s",
    "frames_requested",
    "warmup_frames",
    "tenants",
    "load",
    "sharing",
    "sms_per_tenant",
    "placement",
    "background_period_s",
    "burst_size",
    "aborted",
]

# Every field of RequestRecord except the four that are fixed per cell (engine,
# model, quant, power_mode: in cells.csv) and co_tenants (in co_tenants.csv).
REQUEST_COLUMNS = [
    "cell",
    "tenant",
    "req_id",
    "tile_id",
    "outcome",
    "error",
    "t_scheduled",
    "t_arrival",
    "t_send",
    "t_first_token",
    "t_last_token",
    "deadline",
    "n_prompt_tokens",
    "n_output_tokens",
    "output_text",
    "engine_prompt_ms",
    "engine_decode_ms",
    "engine_stages_ms",  # JSON object, robot policies only
    "n_actions",
]
PER_CELL_RECORD_FIELDS = ("engine", "model", "quant", "power_mode")
CO_TENANT_COLUMNS = ["cell", "req_id", "pid", "name", "used_mib"]
POWER_COLUMNS = ["cell", "t", "gpu_w", "board_w", "gpu_mhz"]
GPU_UTIL_COLUMNS = ["cell", "t", "gpu_util_pct"]

INT_FIELDS = {"tenant", "req_id", "n_prompt_tokens", "n_output_tokens", "n_actions"}
STR_FIELDS = {"cell", "tile_id", "outcome", "error", "output_text"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write rows; None becomes an empty field. Fails on a key not in `columns`."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})


def number(value: str) -> int | float | None:
    """A CSV field as a number: '' -> None, '3' -> 3, '3.5' -> 3.5."""
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_cell_row(row: dict[str, str]) -> dict:
    """cells.csv row with numbers as numbers and empty fields as None."""
    parsed: dict = {}
    for key, value in row.items():
        if value == "":
            parsed[key] = None
        elif key in (
            "cadence_s",
            "frames_requested",
            "warmup_frames",
            "tenants",
            "sms_per_tenant",
            "background_period_s",
            "burst_size",
        ):
            parsed[key] = number(value)
        else:
            parsed[key] = value
    return parsed


def parse_request_row(row: dict[str, str]) -> dict:
    """requests.csv row typed as RequestRecord expects. Empty text stays '' for output_text."""
    parsed: dict = {}
    for key, value in row.items():
        if key == "output_text":
            parsed[key] = value
        elif value == "":
            parsed[key] = None
        elif key in INT_FIELDS:
            parsed[key] = int(value)
        elif key in STR_FIELDS:
            parsed[key] = value
        elif key == "engine_stages_ms":
            parsed[key] = json.loads(value)
        else:
            parsed[key] = float(value)
    return parsed


@dataclass
class Results:
    """One results folder, loaded. Rows are grouped by cell, in cells.csv order."""

    folder: Path
    cells: list[dict]
    requests: dict[str, list[dict]] = field(default_factory=dict)
    co_tenants: dict[tuple[str, int], list[dict]] = field(default_factory=dict)
    power_rows: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    gpu_util_rows: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def cell(self, name: str) -> dict:
        return next(c for c in self.cells if c["cell"] == name)

    def manifest(self, cell: str) -> dict:
        return json.loads((self.folder / MANIFESTS / f"{cell}.json").read_text())

    def records(self, cell: str) -> list[RequestRecord]:
        """The measured tenant's requests, as the harness recorded them."""
        static = {k: self.cell(cell)[k] for k in PER_CELL_RECORD_FIELDS}
        records = []
        for row in self.requests.get(cell, []):
            if row["tenant"] != MEASURED_TENANT:
                continue
            fields = {k: v for k, v in row.items() if k not in ("cell", "tenant")}
            fields["co_tenants"] = self.co_tenants.get((cell, row["req_id"]), [])
            records.append(RequestRecord(**fields, **static))
        return records

    def neighbour_requests(self, cell: str) -> dict[int, list[dict]]:
        """Background tenants' requests by tenant number (1, 2, ...)."""
        by_tenant: dict[int, list[dict]] = {}
        for row in self.requests.get(cell, []):
            if row["tenant"] != MEASURED_TENANT:
                by_tenant.setdefault(row["tenant"], []).append(row)
        return dict(sorted(by_tenant.items()))

    def power(self, cell: str) -> list[PowerSample]:
        return [
            PowerSample(
                t=float(r["t"]),
                gpu_w=float(r["gpu_w"]),
                board_w=float(r["board_w"]),
                gpu_mhz=float(r["gpu_mhz"]) if r["gpu_mhz"] else None,
            )
            for r in self.power_rows.get(cell, [])
        ]

    def gpu_util(self, cell: str) -> list[tuple[float, float]]:
        """(t, percent) rows; samples nvidia-smi could not read are skipped."""
        rows = []
        for r in self.gpu_util_rows.get(cell, []):
            try:
                rows.append((float(r["t"]), float(r["gpu_util_pct"])))
            except ValueError:
                continue
        return rows


def group_by_cell(rows: list[dict]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["cell"], []).append(row)
    return grouped


def load(folder: Path) -> Results:
    """Read a results folder written by harness.collect."""
    cells = [parse_cell_row(r) for r in read_csv(folder / CELLS)]
    if not cells:
        raise FileNotFoundError(f"{folder / CELLS} is missing or empty")
    co_tenants: dict[tuple[str, int], list[dict]] = {}
    for r in read_csv(folder / CO_TENANTS):
        co_tenants.setdefault((r["cell"], int(r["req_id"])), []).append(
            {"pid": int(r["pid"]), "name": r["name"], "used_mib": r["used_mib"]}
        )
    return Results(
        folder=folder,
        cells=cells,
        requests=group_by_cell([parse_request_row(r) for r in read_csv(folder / REQUESTS)]),
        co_tenants=co_tenants,
        power_rows=group_by_cell(read_csv(folder / POWER)),
        gpu_util_rows=group_by_cell(read_csv(folder / GPU_UTIL)),
    )
