"""What every run does, whichever stack: command line, run directory, device capture, report.

A stack's run script supplies two things: its config, and a `send_one` coroutine
that sends one request and returns its record. This module does the rest.
"""

import argparse
import asyncio
import json
import subprocess
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from harness.device import PowerSampler, Tegrastats
from harness.feeder import SendOne, run_closed_loop, run_fixed_rate
from harness.records import RequestRecord, write_records
from harness.report import report

RUNS_DIR = Path("/data/projects/pulak/runs")


def mode_arguments(description: str, default_period_s: float) -> argparse.ArgumentParser:
    """The two run modes every stack offers, as `latency` and `throughput` subcommands."""
    parser = argparse.ArgumentParser(description=description)
    modes = parser.add_subparsers(dest="mode", required=True)

    latency = modes.add_parser("latency", help="fixed-rate input; report latency and misses")
    latency.add_argument(
        "--period",
        dest="period_s",
        type=float,
        default=default_period_s,
        help=f"seconds between inputs; also the deadline (default {default_period_s:g})",
    )
    latency.add_argument("--n-requests", type=int, default=100)

    throughput = modes.add_parser("throughput", help="back-to-back requests; report requests/s")
    throughput.add_argument(
        "--duration",
        dest="duration_s",
        type=float,
        default=300.0,
        help="seconds to keep sending (default 300)",
    )
    return parser


def new_run_dir(stack: str, mode: str) -> Path:
    run_dir = RUNS_DIR / f"{stack}-{mode}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    return run_dir


def harness_commit() -> str:
    """This repo's commit, with '-dirty' if there are uncommitted changes."""
    result = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    return result.stdout.strip()


async def send_on_schedule(args: argparse.Namespace, send_one: SendOne) -> list[RequestRecord]:
    """Run the schedule the command line asked for."""
    if args.mode == "latency":
        return await run_fixed_rate(send_one, args.period_s, args.n_requests)
    return await run_closed_loop(send_one, args.duration_s)


def measure_and_report(
    run_dir: Path, config: dict, measure: Callable[[], Awaitable[list[RequestRecord]]]
) -> None:
    """Write the config, run `measure` while capturing power and tegrastats, then report."""
    config = config | {
        "harness_commit": harness_commit(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "run.json").write_text(json.dumps(config, indent=2) + "\n")

    with Tegrastats(run_dir / "tegrastats.log"), PowerSampler() as power:
        records = asyncio.run(measure())

    write_records(run_dir / "records.jsonl", records)
    power.write_csv(run_dir / "power.csv")
    print(json.dumps(report(run_dir), indent=2))
    print(f"run directory: {run_dir}")
