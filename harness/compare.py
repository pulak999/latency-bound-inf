"""Put the cells of a results folder on one axis.

    uv run python -m harness.compare results/<YYYY-MM-DD>-<question>

Prints a table with one row per cell and draws figures/miss_vs_deadline.png and
.pdf (one curve per cell, in the paper style of harness/paper_style.py). Every
number is computed here from the folder's raw tables (harness.report does the
arithmetic); the table is printed, not saved.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness import paper_style as ps
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


def is_alone(cell: dict) -> bool:
    return (cell.get("tenants") or 1) == 1 and (cell.get("load") or "none") == "none"


def plot_miss_rate_curves(loaded: results.Results, path: Path) -> None:
    """Share of frames that miss each deadline D, one line per cell, in the paper style.

    Blue and orange keep their one meaning: if the folder has exactly one cell
    running alone and the others have neighbours, alone is blue and the rest are
    orange. Otherwise (e.g. a sweep over T) every cell is black, told apart by its
    dash pattern. Each cell's frame period T is marked with a dotted line.
    """
    ps.apply()
    curves = []
    longest = 0.0
    for cell in loaded.cells:
        records = loaded.records(cell["cell"])
        if not records:
            continue
        latencies = [r.latency_s for r in records if r.latency_s is not None]
        longest = max(longest, max(latencies, default=0.0), cell["cadence_s"] or 0.0)
        curves.append((cell, records))
    alone = [cell for cell, _ in curves if is_alone(cell)]
    colour_by_neighbours = len(alone) == 1 and len(curves) > 1

    fig, ax = plt.subplots(figsize=(ps.COLUMN_IN, 2.4))
    dashes = iter(ps.DASHES * len(curves))
    for cell, records in curves:
        deadlines, rates = miss_rate_curve(records, longest * 1.5)
        if colour_by_neighbours and is_alone(cell):
            colour, style = ps.BLUE, "-"
        else:
            colour, style = (ps.ORANGE if colour_by_neighbours else ps.INK), next(dashes)
        ax.step(
            deadlines,
            100 * rates,
            where="post",
            color=colour,
            ls=style,
            lw=1.1,
            label=cell["cell_id"] or cell["cell"],
        )
    for period_s in sorted({cell["cadence_s"] for cell, _ in curves if cell["cadence_s"]}):
        ax.axvline(period_s, color=ps.INK, lw=ps.LW, ls=ps.DEADLINE_LS)
        ax.text(
            period_s,
            1.0,
            f" $T$ = {period_s:g} s",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=ps.F_SMALL,
            color=ps.INK,
        )
    ax.set_xlabel("deadline $D$ (s)")
    ax.set_ylabel("frames that miss $D$ (%)")
    ax.set_ylim(-2, 103)
    ax.grid(axis="y")
    fig.legend(
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=ps.F_SMALL, handlelength=2.6
    )
    rows = (len(curves) + 1) // 2
    printed = ps.footnote(fig, ps.D_MEANS, ps.T_MEANS)
    legend_in = 0.14 * rows
    notes_in = 0.11 * printed
    fig.set_figheight(2.2 + legend_in + notes_in)
    for legend in fig.legends:
        legend.set_bbox_to_anchor((0.5, notes_in / fig.get_figheight()))
    fig.tight_layout(rect=(0, (legend_in + notes_in + 0.05) / fig.get_figheight(), 1, 1))
    ps.save(fig, path)


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
