"""Put the cells of a results folder on one axis.

    uv run python -m harness.compare results/<YYYY-MM-DD>-<question>

Prints a table with one row per cell and draws figures/miss_vs_deadline.png
(one curve per cell). Every number is computed here from the folder's raw
tables (harness.report does the arithmetic); the table is printed, not saved.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness import results
from harness.report import miss_rate_at, miss_rate_curve, summarize

COLUMNS = [
    ("cell", lambda s: s.get("cell_id") or s["run"]),
    ("frames", lambda s: s["n_requests"]),
    ("p50 ms", lambda s: s.get("latency_ms", {}).get("p50")),
    ("p99 ms", lambda s: s.get("latency_ms", {}).get("p99")),
    ("p99/p50", lambda s: s.get("latency_ms", {}).get("p99_over_p50")),
    ("miss @ D=T", lambda s: s.get("miss_rate_at_cadence")),
    ("overwritten", lambda s: s["outcomes"]["overwritten"]),
    ("wait max ms", lambda s: s.get("waiting_ms", {}).get("max")),
    ("prefill mean ms", lambda s: s.get("prefill_ms", {}).get("mean")),
    ("J/frame GPU", lambda s: s.get("energy_per_request_j", {}).get("gpu")),
    ("GPU MHz mean", lambda s: s.get("gpu_mhz", {}).get("mean")),
]


def summary_table(summaries: list[dict]) -> str:
    header = "| " + " | ".join(name for name, _ in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    rows = ["| " + " | ".join(str(get(s)) for _, get in COLUMNS) + " |" for s in summaries]
    return "\n".join([header, rule, *rows]) + "\n"


def cell_summary(loaded: results.Results, cell: dict) -> dict:
    records = loaded.records(cell["cell"])
    summary = {"run": cell["cell"], "cell_id": cell["cell_id"]}
    summary |= summarize(records, loaded.power(cell["cell"]))
    if cell["cadence_s"] is not None and records:
        summary["miss_rate_at_cadence"] = round(miss_rate_at(records, cell["cadence_s"]), 4)
    return summary


def plot_miss_rate_curves(loaded: results.Results, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    longest = 0.0
    curves = []
    for cell in loaded.cells:
        records = loaded.records(cell["cell"])
        if not records:
            continue
        period_s = cell["cadence_s"]
        latencies = [r.latency_s for r in records if r.latency_s is not None]
        longest = max(longest, max(latencies, default=0.0), period_s or 0.0)
        curves.append((cell["cell_id"] or cell["cell"], records, period_s))
    for label, records, period_s in curves:
        deadlines, rates = miss_rate_curve(records, longest * 1.5)
        (line,) = ax.plot(deadlines, rates, label=label)
        if period_s is not None:
            ax.axvline(period_s, color=line.get_color(), linestyle=":", alpha=0.6)
    ax.set_xlabel("deadline D (s); dotted lines are each cell's cadence")
    ax.set_ylabel("miss rate")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="a results folder written by harness.collect")
    args = parser.parse_args()

    loaded = results.load(args.folder)
    print(summary_table([cell_summary(loaded, c) for c in loaded.cells]))
    figures = args.folder / results.FIGURES
    figures.mkdir(exist_ok=True)
    plot_miss_rate_curves(loaded, figures / "miss_vs_deadline.png")
    print(f"figure: {figures / 'miss_vs_deadline.png'}")


if __name__ == "__main__":
    main()
