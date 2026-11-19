"""Miss rate against deadline, alone and with other tenants, from multi-tenant cell logs.

    uv run python -m harness.plot_tenancy RUNS_DIR --prefix q- -o reports/space-multitenancy-quick

Takes the newest directory of each cell id under RUNS_DIR that starts with
--prefix, and draws one figure with a panel per sharing mode (time-slicing,
MPS partitions). Each panel has the alone cell and one line per background
load, with the miss rate at every deadline applied to the logs afterwards.
Writes the figure, the curves as CSV, and a summary table.

A second figure, `latency_vs_busy_neighbours.png`, puts every served frame at
the average number of neighbours that were inside a request while it ran, so
the cost of each busy neighbour can be read off directly for each kind of sharing.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness.records import read_records  # noqa: E402
from harness.report import miss_rate_at, wilson_interval  # noqa: E402

# Reference categorical palette, fixed order; the alone baseline is neutral ink.
LOAD_COLORS = {"steady": "#2a78d6", "bursty": "#eb6834", "saturating": "#1baf7a"}
ALONE_COLOR = "#5f5e5a"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
PANELS = [("timeslice", "Time-slicing (no MPS)"), ("mps-partitioned", "MPS, equal SM partitions")]


def newest_cells(runs_dir: Path, prefix: str) -> dict[str, Path]:
    """Cell id -> its newest directory (names are <cell id>-<YYYYmmdd-HHMMSS>)."""
    cells: dict[str, Path] = {}
    for path in sorted(runs_dir.glob(f"{prefix}*")):
        if (path / "manifest.json").exists():
            cells[path.name.rsplit("-", 2)[0]] = path  # sorted, so the newest wins
    return cells


def background_busy(cell_dir: Path, t_start: float, t_end: float) -> float | None:
    """Mean share of the cell each background tenant spent inside a request."""
    shares = []
    for log in sorted(cell_dir.glob("background-*.jsonl")):
        busy = 0.0
        for line in log.read_text().splitlines():
            request = json.loads(line)
            busy += max(0.0, min(request["t_done"], t_end) - max(request["t_send"], t_start))
        shares.append(busy / (t_end - t_start))
    return float(np.mean(shares)) if shares else None


def load_cell(cell_dir: Path) -> dict:
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    records = read_records(cell_dir / "records.jsonl")
    tenancy = manifest.get("tenancy", {})
    t_start = min(r.t_arrival for r in records)
    t_end = max((r.t_last_token or r.t_arrival) for r in records)
    latencies = np.array([r.latency_s for r in records if r.latency_s is not None])
    return {
        "dir": cell_dir,
        "records": records,
        "tenants": tenancy.get("tenants", 1),
        "load": tenancy.get("load", "none"),
        "sharing": tenancy.get("sharing", "timeslice"),
        "sms": tenancy.get("sms_per_tenant"),
        "busy": background_busy(cell_dir, t_start, t_end),
        "p50_s": float(np.percentile(latencies, 50)) if len(latencies) else None,
        "p99_s": float(np.percentile(latencies, 99)) if len(latencies) else None,
        "n": len(records),
        "cadence_s": manifest.get("cadence_s"),
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
    rows = []
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
            if sharing == PANELS[0][0] or cell["tenants"] > 1:
                rows += [
                    [cell["dir"].name, sharing, d, r] for d, r in zip(deadlines, rate, strict=True)
                ]
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
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "miss_vs_deadline.png", dpi=160, facecolor=SURFACE)
    with (out / "miss_vs_deadline.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "panel", "deadline_s", "miss_rate"])
        writer.writerows(rows)


def busy_neighbours(cell_dir: Path, t_start: float, t_end: float) -> float:
    """Average number of neighbours inside a request between t_start and t_end."""
    overlap = 0.0
    for log in cell_dir.glob("background-*.jsonl"):
        for line in log.read_text().splitlines():
            request = json.loads(line)
            overlap += max(0.0, min(t_end, request["t_done"]) - max(t_start, request["t_send"]))
    return overlap / (t_end - t_start)


def plot_busy_neighbours(cells: list[dict], out: Path, mark_s: float | None) -> list[list]:
    """Each served frame's time against how many neighbours were busy during it."""
    series = {
        "timeslice": ("Time-slicing (no MPS)", "#2a78d6", "o"),
        "mps-partitioned": ("MPS, equal SM partitions", "#eb6834", "s"),
    }
    alone = [r.latency_s for c in cells if c["tenants"] == 1 for r in c["records"] if r.latency_s]
    alone_s = float(np.median(alone))
    rows = []
    nudge = np.random.default_rng(0)  # sideways only, so stacked frames stay visible
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    for sharing, (name, color, marker) in series.items():
        points = [
            (busy_neighbours(c["dir"], r.t_send, r.t_last_token), r.latency_s)
            for c in cells
            if c["tenants"] > 1 and c["sharing"] == sharing
            for r in c["records"]
            if r.latency_s is not None
        ]
        if not points:
            continue
        x, y = np.array(points).T
        x = x + nudge.uniform(-0.04, 0.04, len(x))
        ax.scatter(
            x, y, s=42, color=color, marker=marker, edgecolor=SURFACE, lw=1.5, label=name, zorder=3
        )
        rows += [[sharing, round(a, 3), round(b, 3)] for a, b in points]
    ax.scatter(
        nudge.uniform(-0.04, 0.04, len(alone)),
        alone,
        s=42,
        color=ALONE_COLOR,
        edgecolor=SURFACE,
        lw=1.5,
        label="alone",
        zorder=3,
    )
    k = np.linspace(0, 2, 50)
    ax.plot(k, alone_s * (1 + k), color="#2a78d6", lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(
        1.2,
        2.55,
        "fair share of the GPU:\nalone × (1 + busy neighbours)",
        color="#2a78d6",
        fontsize=8,
        ha="left",
        va="top",
    )
    ax.text(
        1.0,
        4.68,
        "partitions: about the same time whatever the neighbours do",
        color="#eb6834",
        fontsize=8,
        ha="center",
    )
    if mark_s is not None:
        ax.axhline(mark_s, color=MUTED, lw=1)
        ax.text(0, mark_s, f"{mark_s:g} s deadline ", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylim(0, max(5.0, alone_s * 3.2))
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel("Neighbours busy during the frame (average over the frame)", color=MUTED)
    ax.set_ylabel("Time for the frame (s)", color=MUTED)
    ax.set_title(
        "Time-slicing: each busy neighbour adds a full frame time",
        color=INK,
        fontsize=12,
        loc="left",
    )
    ax.grid(color=GRID, lw=0.8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    legend = ax.legend(fontsize=8.5, loc="lower right", facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.text(
        0.01,
        0.01,
        "One point per frame of the measured tenant; 3 tenants, each in its own container, "
        "neighbours bursty or saturating.\nPoints nudged sideways by up to 0.04 so stacked "
        "frames show.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "latency_vs_busy_neighbours.png", dpi=160, facecolor=SURFACE)
    with (out / "latency_vs_busy_neighbours.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sharing", "busy_neighbours", "latency_s"])
        writer.writerows(rows)
    return rows


def summary_table(cells: list[dict]) -> str:
    lines = [
        "| cell | tenants | sharing | neighbours | neighbours busy "
        "| p50 (s) | p99 (s) | missed at 3 s | overwritten |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(cells, key=lambda c: (c["tenants"], c["sharing"], c["load"])):
        busy = f"{c['busy']:.0%}" if c["busy"] is not None else "-"
        sharing = c["sharing"] + (f" ({c['sms']} SMs)" if c["sms"] else "")
        p50 = f"{c['p50_s']:.2f}" if c["p50_s"] is not None else "-"
        p99 = f"{c['p99_s']:.2f}" if c["p99_s"] is not None else "-"
        lines.append(
            f"| {c['dir'].name} | {c['tenants']} | {sharing} | {c['load']} | {busy} | "
            f"{p50} | {p99} | {miss_rate_at(c['records'], 3.0):.0%} | {c['overwritten']}/{c['n']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--prefix", default="q-")
    parser.add_argument("--max-deadline", type=float, default=10.0)
    parser.add_argument("--mark", type=float, default=3.0, help="deadline to mark (s)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    cells = [load_cell(d) for d in newest_cells(args.runs_dir, args.prefix).values()]
    if not cells:
        parser.error(f"no cells starting with {args.prefix!r} in {args.runs_dir}")
    plot(cells, args.out, args.max_deadline, args.mark)
    plot_busy_neighbours(cells, args.out, args.mark)
    table = summary_table(cells)
    (args.out / "summary.md").write_text(table)
    print(table)
    print(f"figure: {args.out / 'miss_vs_deadline.png'}")


if __name__ == "__main__":
    main()
