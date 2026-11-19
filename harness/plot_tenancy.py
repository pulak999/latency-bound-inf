"""Miss rate against deadline, alone and with other tenants, from multi-tenant cell logs.

    uv run python -m harness.plot_tenancy results/2026-11-19-space-multitenancy-initial

Reads a results folder (harness/results.py) and draws, into its figures/, one
figure with a panel per sharing mode (time-slicing, MPS partitions). Each panel
has the alone cell and one line per background load, with the miss rate at
every deadline applied to the raw requests here. Prints a summary table; every
number in it is computed from the folder's tables, none is stored.

A second figure, `timeline_bursty.png`, shows why the kinds of sharing differ:
one bursty cell per kind of sharing, the neighbours' requests above the
measured tenant's frames, each frame coloured by whether it made the deadline.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import results  # noqa: E402
from harness.report import miss_rate_at, wilson_interval  # noqa: E402

# Reference categorical palette, fixed order; the alone baseline is neutral ink.
LOAD_COLORS = {"steady": "#2a78d6", "bursty": "#eb6834", "saturating": "#1baf7a"}
ALONE_COLOR = "#5f5e5a"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
PANELS = [("timeslice", "Time-slicing (no MPS)"), ("mps-partitioned", "MPS, equal SM partitions")]


def background_busy(
    neighbours: dict[int, list[dict]], t_start: float, t_end: float
) -> float | None:
    """Mean share of the cell each background tenant spent inside a request."""
    shares = []
    for requests in neighbours.values():
        busy = 0.0
        for r in requests:
            busy += max(0.0, min(r["t_last_token"], t_end) - max(r["t_send"], t_start))
        shares.append(busy / (t_end - t_start))
    return float(np.mean(shares)) if shares else None


def load_cell(loaded: results.Results, cell: dict) -> dict:
    """A cell's settings, its requests, and the statistics the figures and table need."""
    name = cell["cell"]
    records = loaded.records(name)
    neighbours = loaded.neighbour_requests(name)
    t_start = min(r.t_arrival for r in records)
    t_end = max((r.t_last_token or r.t_arrival) for r in records)
    latencies = np.array([r.latency_s for r in records if r.latency_s is not None])
    return {
        "name": name,
        "records": records,
        "neighbours": neighbours,
        "tenants": cell["tenants"],
        "load": cell["load"] or "none",
        "sharing": cell["sharing"] or "timeslice",
        "sms": cell["sms_per_tenant"],
        "busy": background_busy(neighbours, t_start, t_end),
        "p50_s": float(np.percentile(latencies, 50)) if len(latencies) else None,
        "p99_s": float(np.percentile(latencies, 99)) if len(latencies) else None,
        "n": len(records),
        "cadence_s": cell["cadence_s"],
        "overwritten": sum(r.outcome == "overwritten" for r in records),
    }


def curve(records, deadlines: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates, lows, highs = [], [], []
    for d in deadlines:
        rate = miss_rate_at(records, d)
        low, high = wilson_interval(round(rate * len(records)), len(records))
        rates.append(rate)
        lows.append(low)
        highs.append(high)
    return np.array(rates), np.array(lows), np.array(highs)


def label(cell: dict) -> str:
    if cell["tenants"] == 1:
        return "alone"
    busy = f", neighbours {cell['busy']:.0%} busy" if cell["busy"] is not None else ""
    return f"{cell['tenants']} tenants, {cell['load']}{busy}"


def plot(cells: list[dict], out: Path, max_deadline_s: float, mark_s: float | None) -> None:
    deadlines = np.linspace(0.5, max_deadline_s, 300)
    alone = [c for c in cells if c["tenants"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for ax, (sharing, title) in zip(axes, PANELS, strict=True):
        panel = alone + [c for c in cells if c["tenants"] > 1 and c["sharing"] == sharing]
        panel.sort(
            key=lambda c: list(LOAD_COLORS).index(c["load"]) if c["load"] in LOAD_COLORS else -1
        )
        for cell in panel:
            color = ALONE_COLOR if cell["tenants"] == 1 else LOAD_COLORS[cell["load"]]
            rate, low, high = curve(cell["records"], deadlines)
            ax.fill_between(deadlines, low * 100, high * 100, color=color, alpha=0.12, lw=0)
            ax.plot(deadlines, rate * 100, color=color, lw=2, label=label(cell))
        sms = next((c["sms"] for c in panel if c["sms"]), None)
        ax.set_title(
            title + (f" ({sms} SMs each)" if sms else ""), color=INK, fontsize=11, loc="left"
        )
        ax.set_xlim(max_deadline_s, 0.5)  # deadlines get tighter to the right
        if mark_s is not None:
            ax.axvline(mark_s, color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.text(mark_s, 50, f" {mark_s:g} s deadline", color=MUTED, fontsize=8, ha="right")
        ax.set_ylim(-2, 102)
        ax.set_xlabel("Deadline (s), tighter →", color=MUTED)
        ax.grid(color=GRID, lw=0.8)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
        legend = ax.legend(fontsize=8.5, loc="upper left", facecolor=SURFACE, edgecolor=GRID)
        for text in legend.get_texts():
            text.set_color(INK)
    axes[0].set_ylabel("Frames missing the deadline (%)", color=MUTED)
    n = max(c["n"] for c in cells)
    fig.suptitle(
        "Gemma 3 4B (llama.cpp) on Jetson Thor: deadline misses alone vs 3 VLM tenants",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        f"Each tenant in its own container. Measured tenant: one frame every "
        f"{cells[0]['cadence_s']:g} s, {n} frames "
        "per cell. Neighbours: steady and bursty have the same mean request rate. "
        "Shaded: 95% Wilson interval.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out / "miss_vs_deadline.png", dpi=160, facecolor=SURFACE)


SHARING_STYLE = {
    "timeslice": ("Time-slicing (no MPS)", "#2a78d6"),
    "mps-partitioned": ("MPS, equal SM partitions", "#eb6834"),
}
MISSED, MADE = "#d03b3b", "#0ca30c"  # status colours, always with a legend label


def style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_timeline(cells: list[dict], out: Path, load: str, mark_s: float) -> None:
    """One cell per sharing mode: when the neighbours were busy, and tenant 0's frames under it."""
    panels = [c for c in cells if c["tenants"] > 1 and c["load"] == load]
    panels.sort(key=lambda c: list(SHARING_STYLE).index(c["sharing"]))
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 1.9 * len(panels) + 0.8), facecolor=SURFACE, sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, cell in zip(axes, panels, strict=True):
        records = [r for r in cell["records"] if r.latency_s is not None]
        t0 = min(r.t_arrival for r in cell["records"])
        neighbours = cell["neighbours"]
        lanes = [f"neighbour {i}" for i in sorted(neighbours, reverse=True)] + [
            "tenant 0 (measured)"
        ]
        for lane, requests in neighbours.items():
            for request in requests:
                start, end = request["t_send"] - t0, request["t_last_token"] - t0
                ax.barh(lane, end - start, left=start, height=0.6, color=MUTED, alpha=0.55, lw=0)
        for r in records:
            missed = r.latency_s > mark_s
            ax.barh(
                0,
                r.t_last_token - r.t_send,
                left=r.t_send - t0,
                height=0.6,
                color=MISSED if missed else MADE,
                lw=0,
            )
            ax.text(
                r.t_send - t0 + (r.t_last_token - r.t_send) / 2,
                -0.55,
                f"{r.latency_s:.1f}",
                ha="center",
                va="top",
                color=INK,
                fontsize=7.5,
            )
        ax.set_yticks(range(len(lanes) - 1, -1, -1), lanes)
        ax.set_ylim(-1.0, len(neighbours) + 0.5)
        name, _ = SHARING_STYLE[cell["sharing"]]
        missed = sum(r.latency_s > mark_s for r in records)
        ax.set_title(
            f"{name}: {missed} of {len(records)} frames missed {mark_s:g} s",
            color=INK,
            fontsize=10.5,
            loc="left",
        )
        style_axes(ax)
        ax.grid(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=a)
        for c, a in ((MUTED, 0.55), (MADE, 1), (MISSED, 1))
    ]
    labels = ["neighbour busy", f"frame made {mark_s:g} s", f"frame missed {mark_s:g} s"]
    legend = axes[0].legend(
        handles,
        labels,
        fontsize=8.5,
        loc="upper right",
        ncol=3,
        facecolor=SURFACE,
        edgecolor=GRID,
        bbox_to_anchor=(1, 1.45),
    )
    for text in legend.get_texts():
        text.set_color(INK)
    axes[-1].set_xlabel(
        "Time since the first frame (s); numbers under frames: seconds taken", color=MUTED
    )
    axes[-1].set_xlim(-2, None)  # neighbours start before tenant 0's first frame
    fig.suptitle(
        f"Same {load} neighbours, two ways of sharing the GPU",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / f"timeline_{load}.png", dpi=160, facecolor=SURFACE)


def summary_table(cells: list[dict], mark_s: float) -> str:
    lines = [
        "| cell | tenants | sharing | neighbours | neighbours busy "
        f"| p50 (s) | p99 (s) | missed at {mark_s:g} s | overwritten |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(cells, key=lambda c: (c["tenants"], c["sharing"], c["load"])):
        busy = f"{c['busy']:.0%}" if c["busy"] is not None else "-"
        sharing = c["sharing"] + (f" ({c['sms']} SMs)" if c["sms"] else "")
        p50 = f"{c['p50_s']:.2f}" if c["p50_s"] is not None else "-"
        p99 = f"{c['p99_s']:.2f}" if c["p99_s"] is not None else "-"
        missed = miss_rate_at(c["records"], mark_s)
        lines.append(
            f"| {c['name']} | {c['tenants']} | {sharing} | {c['load']} | {busy} | "
            f"{p50} | {p99} | {missed:.0%} | {c['overwritten']}/{c['n']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="a results folder written by harness.collect")
    parser.add_argument("--max-deadline", type=float, default=10.0)
    parser.add_argument("--mark", type=float, default=3.0, help="deadline to mark (s)")
    args = parser.parse_args()

    loaded = results.load(args.folder)
    cells = [load_cell(loaded, c) for c in loaded.cells]
    figures = args.folder / results.FIGURES
    figures.mkdir(exist_ok=True)
    plot(cells, figures, args.max_deadline, args.mark)
    plot_timeline(cells, figures, "bursty", args.mark)
    print(summary_table(cells, args.mark))
    print(f"figures: {figures}")


if __name__ == "__main__":
    main()
