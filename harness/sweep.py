"""Run one model at several input rates, then plot latency and GPU clock against the period.

    uv run python -m harness.sweep run smolvla      # or pi05, a1
    uv run python -m harness.sweep plot /data/projects/pulak/runs/sweep-smolvla-<time>

The period is the time between inputs. The GPU idles for the period minus the
latency, so the sweep shows whether idle time between requests changes how
long a request takes. Each step is one ordinary run in its own directory
inside the sweep directory; before each step the sweep waits for no other
process to be using the GPU.
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; the Thor runs headless
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = Path("/data/projects/pulak/runs")
STACK_A_PYTHON = REPO / ".venv/bin/python"
STACK_B_PYTHON = Path("/data/projects/pulak/envs/lerobot/bin/python")


@dataclass
class Step:
    """One run of the sweep: back to back for `duration_s`, or one input per `period_s`."""

    period_s: float | None = None
    n_requests: int | None = None
    duration_s: float | None = None

    def mode_arguments(self) -> list[str]:
        if self.period_s is None:
            return ["throughput", "--duration", str(self.duration_s)]
        return ["latency", "--period", str(self.period_s), "--n-requests", str(self.n_requests)]


# 1000 requests per robot step gives a p99 resting on the 10 slowest requests.
# Gemma takes 1.7 s per request, so its long periods get fewer requests to keep
# each step under an hour.
ROBOT_STEPS = [
    Step(duration_s=200),
    Step(period_s=0.2, n_requests=1000),
    Step(period_s=0.5, n_requests=1000),
    Step(period_s=1.0, n_requests=1000),
    Step(period_s=3.0, n_requests=1000),
]
SWEEPS = {
    "smolvla": ([STACK_B_PYTHON, "-m", "harness.run_b1", "--policy", "smolvla"], ROBOT_STEPS),
    "pi05": ([STACK_B_PYTHON, "-m", "harness.run_b1", "--policy", "pi05"], ROBOT_STEPS),
    "a1": (
        [STACK_A_PYTHON, "-m", "harness.run_a1"],
        [
            Step(duration_s=300),
            Step(period_s=3.0, n_requests=1000),
            Step(period_s=10.0, n_requests=300),
            Step(period_s=30.0, n_requests=100),
        ],
    ),
}


def gpu_processes() -> list[str]:
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line.strip()]


def wait_for_idle_gpu(poll_s: float = 30) -> None:
    """Block until no process is using the GPU, so other people's jobs stay out of the numbers."""
    while others := gpu_processes():
        print(f"{datetime.now():%H:%M:%S} GPU busy, waiting: {'; '.join(others)}", flush=True)
        time.sleep(poll_s)


def run_sweep(name: str) -> Path:
    command, steps = SWEEPS[name]
    sweep_dir = RUNS_DIR / f"sweep-{name}-{datetime.now():%Y%m%d-%H%M%S}"
    sweep_dir.mkdir(parents=True)

    for i, step in enumerate(steps):
        wait_for_idle_gpu()
        print(f"{datetime.now():%H:%M:%S} step {i + 1}/{len(steps)}: {step}", flush=True)
        log = sweep_dir / f"step-{i + 1}.log"
        with log.open("w") as f:
            subprocess.run(
                [*map(str, command), "--out", str(sweep_dir), *step.mode_arguments()],
                cwd=REPO,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,  # a failed step stops the sweep; its log says why
            )

    plot_sweep(sweep_dir)
    return sweep_dir


def load_steps(sweep_dir: Path) -> list[dict]:
    """One row per finished run in the sweep, sorted by period (back to back first)."""
    rows = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if (p / "summary.json").exists()):
        config = json.loads((run_dir / "run.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        clocks = summary.get("gpu_clock_mhz", {})
        rows.append(
            {
                "run": run_dir.name,
                "period_s": config.get("period_s"),
                "n_requests": summary["n_requests"],
                "latency_p50_ms": summary["latency_ms"]["p50"],
                "latency_p99_ms": summary["latency_ms"]["p99"],
                "throughput_rps": summary["throughput_rps"],
                "gpu_mhz_at_send_p50": clocks.get("at_send", {}).get("p50"),
                "gpu_mhz_during_p50": clocks.get("during_request", {}).get("p50"),
                "missed": summary.get("deadline_misses", {}).get("missed"),
                "tail_by_stage_ms": summary["tail_ms"]["by_stage"],
            }
        )
    return sorted(rows, key=lambda row: -1 if row["period_s"] is None else row["period_s"])


def plot_sweep(sweep_dir: Path) -> None:
    """Write sweep.json (the table) and sweep.png (latency and GPU clock against period)."""
    rows = load_steps(sweep_dir)
    (sweep_dir / "sweep.json").write_text(json.dumps(rows, indent=2) + "\n")

    back_to_back = [row for row in rows if row["period_s"] is None]
    paced = [row for row in rows if row["period_s"] is not None]
    periods = [row["period_s"] for row in paced]

    fig, (latency_ax, clock_ax) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    latency_ax.plot(periods, [r["latency_p50_ms"] for r in paced], "o-", label="p50")
    latency_ax.plot(periods, [r["latency_p99_ms"] for r in paced], "s--", label="p99")
    clock_ax.plot(periods, [r["gpu_mhz_at_send_p50"] for r in paced], "o-", label="at send")
    clock_ax.plot(periods, [r["gpu_mhz_during_p50"] for r in paced], "s--", label="during request")
    for row in back_to_back:  # reference lines: the GPU never idles
        latency_ax.axhline(row["latency_p50_ms"], color="grey", linestyle=":", label="back to back")
        if row["gpu_mhz_during_p50"] is not None:
            clock_ax.axhline(row["gpu_mhz_during_p50"], color="grey", linestyle=":")

    latency_ax.set_ylabel("latency (ms)")
    latency_ax.set_title(sweep_dir.name)
    clock_ax.set_ylabel("GPU clock, median (MHz)")
    clock_ax.set_xlabel("time between inputs (s)")
    clock_ax.set_xscale("log")
    for ax in (latency_ax, clock_ax):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(sweep_dir / "sweep.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("run", help="run a sweep, then plot it").add_argument(
        "name", choices=sorted(SWEEPS)
    )
    actions.add_parser("plot", help="re-plot a sweep from its run directories").add_argument(
        "sweep_dir", type=Path
    )
    args = parser.parse_args()

    if args.action == "run":
        print(f"sweep directory: {run_sweep(args.name)}")
    else:
        plot_sweep(args.sweep_dir)


if __name__ == "__main__":
    main()
