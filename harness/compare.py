"""Put several cells on one axis.

    uv run python -m harness.compare runs/alone-T20-* runs/alone-T5-* -o reports/alone

Writes miss_vs_deadline.png (one curve per cell) and summary.md (one row per
cell) into the output directory. Every number comes from the cells' logs via
harness.report; nothing is typed in.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness.report import load_cell, miss_rate_curve, report

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


def plot_miss_rate_curves(cell_dirs: list[Path], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    longest = 0.0
    curves = []
    for cell_dir in cell_dirs:
        manifest, records, _ = load_cell(cell_dir)
        if not records:
            continue
        period_s = manifest.get("period_s")
        latencies = [r.latency_s for r in records if r.latency_s is not None]
        longest = max(longest, max(latencies, default=0.0), period_s or 0.0)
        curves.append((manifest.get("cell_id") or cell_dir.name, records, period_s))
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
    parser.add_argument("cell_dirs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = [report(d) for d in args.cell_dirs]
    table = summary_table(summaries)
    (args.out / "summary.md").write_text(table)
    plot_miss_rate_curves(args.cell_dirs, args.out / "miss_vs_deadline.png")
    print(table)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
