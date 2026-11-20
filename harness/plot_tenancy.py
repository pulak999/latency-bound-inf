"""Time-slicing vs MPS partitions, from a multi-tenant results folder, in the paper style.

    uv run python -m harness.plot_tenancy results/2026-11-19-space-multitenancy-initial

Reads a results folder (harness/results.py), prints a summary table (every
number computed from the folder's raw tables, none stored), and draws three
figures into its figures/, each as .png and .pdf (harness/paper_style.py):

- miss_rate_timeslice_vs_mps: (a, b) the share of frames that miss each
  deadline, one panel per neighbour load; (c) the miss rate at --mark.
- p99_timeslice_vs_mps: (a) p99 latency per cell; (b) every frame, p50 and p99.
- timeline_bursty: the bursty cells over time, neighbours' requests above the
  measured tenant's frames, each frame on time (blue) or late (hatched orange).

Blue is the measured tenant alone; orange is with neighbours (dashed or
diagonal hatch for time-slicing, dotted for MPS partitions). No headlines in
the figures: the captions in the results README carry them.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

from harness import paper_style as ps  # noqa: E402
from harness import results  # noqa: E402
from harness.report import miss_rate_at  # noqa: E402

ps.apply()
SHARING = [  # (key, label, line dash, bar hatch): both perturbed, so both orange
    ("timeslice", "time-slicing", ps.DASHES[0], ps.HATCH),
    ("mps-partitioned", "MPS partitions", ps.DASHES[1], "...."),
]
LOADS = [("bursty", "bursty neighbours"), ("saturating", "saturating neighbours")]


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


def miss_curve(records, deadlines: np.ndarray) -> np.ndarray:
    """Share of frames (overwritten and failed included) that miss each deadline, in %."""
    return np.array([100 * miss_rate_at(records, d) for d in deadlines])


def find(cells: list[dict], tenants: int, load: str = "none", sharing: str = "timeslice"):
    for cell in cells:
        if (cell["tenants"], cell["load"], cell["sharing"]) == (tenants, load, sharing):
            return cell
    return None


def legend_below(fig, ncol: int) -> None:
    """One legend row under the panels: each kind of sharing as its line and its bar."""
    handles = [(plt.Line2D([], [], color=ps.BLUE, lw=1.1), ps.swatch(ps.BLUE))]
    for _, _, dash, hatch in SHARING:
        handles.append(
            (plt.Line2D([], [], color=ps.ORANGE, lw=1.1, ls=dash), ps.swatch(ps.ORANGE, hatch))
        )
    labels = ["alone"] + [f"3 tenants, {label}" for _, label, _, _ in SHARING]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.055),
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        handlelength=4.2,
        columnspacing=2.0,
    )


def grouped_bars(ax, cells: list[dict], value, text, y_max: float) -> None:
    """Alone, then bursty and saturating neighbours, each with time-slicing and MPS bars."""
    width, gap = 0.36, 0.04
    alone = find(cells, 1)
    groups = [[(alone, ps.BLUE, None, 0.0)]] + [
        [
            (find(cells, 3, load, sharing), ps.ORANGE, hatch, sign * (width + gap) / 2)
            for (sharing, _, _, hatch), sign in zip(SHARING, (-1, 1), strict=True)
        ]
        for load, _ in LOADS
    ]
    for i, bars in enumerate(groups):
        for cell, color, hatch, dx in bars:
            if cell is None:
                continue
            v = value(cell)
            ax.bar(
                i + dx,
                max(v, y_max * 0.008),
                width,
                color=color,
                edgecolor="white",
                lw=0,
                hatch=hatch,
            )
            ax.text(
                i + dx,
                v + y_max * 0.025,
                text(v),
                ha="center",
                va="bottom",
                fontsize=ps.F_SMALL,
                color=ps.INK,
            )
    ax.set_xticks(range(3), ["alone", "bursty\nneighbours", "saturating\nneighbours"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y")


def plot_miss_rates(cells: list[dict], out: Path, max_deadline_s: float, mark_s: float) -> None:
    """Time-slicing vs MPS partitions: miss rate as the deadline tightens, and at mark_s.

    (a) and (b): one panel per neighbour load, alone (blue) next to both kinds of
    sharing (orange, dashed and dotted). (c): the miss rate at mark_s as bars.
    """
    alone = find(cells, 1)
    deadlines = np.linspace(0.5, max_deadline_s, 1101)
    fig, axes = plt.subplots(
        1, 3, figsize=(ps.WIDTH_IN, 2.2), gridspec_kw={"width_ratios": [1, 1, 0.85]}
    )
    for ax, letter, (load, load_label) in zip(axes[:2], "ab", LOADS, strict=True):
        if alone:
            ax.step(
                deadlines,
                miss_curve(alone["records"], deadlines),
                where="post",
                color=ps.BLUE,
                lw=1.1,
            )
        for sharing, _, dash, _ in SHARING:
            cell = find(cells, 3, load, sharing)
            if cell:
                ax.step(
                    deadlines,
                    miss_curve(cell["records"], deadlines),
                    where="post",
                    color=ps.ORANGE,
                    lw=1.1,
                    ls=dash,
                )
        ax.axvline(mark_s, color=ps.INK, lw=ps.LW, ls=ps.DEADLINE_LS)
        ax.text(
            mark_s - 0.08,
            80,  # clear of the curves, which sit at 0, 50-60 or 100 % around D
            f"$D$ = {mark_s:g} s",
            ha="left",
            va="center",
            fontsize=ps.F_SMALL,
            color=ps.INK,
            bbox={"fc": "white", "ec": "none", "pad": 1},
        )
        ax.set_xlim(max_deadline_s, 0.5)  # deadlines get tighter to the right
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y")
        ax.set_xlabel(r"deadline $D$ (s), tighter $\rightarrow$")
        ps.panel_title(ax, letter, load_label)
    axes[0].set_ylabel("frames that miss $D$ (%)")
    axes[1].set_yticklabels([])
    grouped_bars(
        axes[2],
        cells,
        lambda c: 100 * miss_rate_at(c["records"], mark_s),
        lambda v: f"{v:.0f}",
        115,
    )
    axes[2].set_yticks([0, 25, 50, 75, 100])
    ps.panel_title(axes[2], "c", f"frames that miss $D$ = {mark_s:g} s (%)")
    legend_below(fig, 3)
    ps.footnote(fig, ps.D_MEANS)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=1.2)
    ps.save(fig, out / "miss_rate_timeslice_vs_mps")


def plot_p99(cells: list[dict], out: Path, mark_s: float) -> None:
    """Time-slicing vs MPS partitions on tail latency.

    (a) p99 latency per cell, deadline dotted. (b) every frame as a dot, with p50
    and p99 marked, so the reader sees how few frames a p99 rests on.
    """
    order = [(find(cells, 1), "alone", ps.BLUE, None)] + [
        (find(cells, 3, load, sharing), f"{load}\n{label}", ps.ORANGE, hatch)
        for load, _ in LOADS
        for sharing, label, _, hatch in SHARING
    ]
    order = [o for o in order if o[0] is not None]
    y_max = max(5.0, 1.12 * max(c["p99_s"] for c, *_ in order))
    fig, axes = plt.subplots(
        1, 2, figsize=(ps.WIDTH_IN, 2.25), gridspec_kw={"width_ratios": [0.9, 1.1]}
    )
    grouped_bars(axes[0], cells, lambda c: c["p99_s"], lambda v: f"{v:.2f}", y_max)
    ps.panel_title(axes[0], "a", "p99 latency (s)")

    ax = axes[1]
    nudge = np.random.default_rng(0)
    for i, (cell, _, color, _) in enumerate(order):
        lat = np.array([r.latency_s for r in cell["records"] if r.latency_s is not None])
        ax.scatter(
            i + nudge.uniform(-0.12, 0.12, len(lat)),
            lat,
            s=7,
            color=color,
            lw=0,
            alpha=0.85,
            zorder=3,
        )
        for q, style in ((50, (0, (1.5, 1.2))), (99, "-")):
            ax.plot(
                [i - 0.3, i + 0.3],
                [np.percentile(lat, q)] * 2,
                color=ps.INK,
                lw=0.9,
                ls=style,
                zorder=4,
            )
    ax.set_xticks(range(len(order)), [label for _, label, _, _ in order], fontsize=ps.F_SMALL)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y")
    ax.set_ylabel(f"{ps.TE} − {ps.TO} (s)")
    n = max(len(c["records"]) for c, *_ in order)
    ps.panel_title(ax, "b", f"every frame ({n} per cell); solid: p99, dotted: p50")
    for a in axes:
        a.axhline(mark_s, color=ps.INK, lw=ps.LW, ls=ps.DEADLINE_LS)
        a.text(
            a.get_xlim()[0] + 0.05,  # left end, over the alone bar, which is far below D
            mark_s,
            f"$D$ = {mark_s:g} s",
            ha="left",
            va="bottom",
            fontsize=ps.F_SMALL,
            color=ps.INK,
        )
    axes[0].set_ylabel("latency (s)")
    legend_below(fig, 3)
    ps.footnote(fig, ps.D_MEANS)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=1.5)
    ps.save(fig, out / "p99_timeslice_vs_mps")


def plot_timeline(cells: list[dict], out: Path, load: str, mark_s: float) -> None:
    """One cell per sharing mode, drawn like the worked example: frames on time in blue,
    late in hatched orange, each frame's deadline dotted, neighbours' requests in grey."""
    panels = [find(cells, 3, load, sharing) for sharing, *_ in SHARING]
    panels = [(c, label) for c, (_, label, _, _) in zip(panels, SHARING, strict=True) if c]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(ps.WIDTH_IN, 1.25 * len(panels) + 0.45), sharex=True
    )
    axes = np.atleast_1d(axes)
    end = 0.0
    for ax, letter, (cell, label) in zip(axes, "abcd", panels, strict=False):
        t0 = min(r.t_arrival for r in cell["records"])
        neighbours = cell["neighbours"]
        rows = sorted(neighbours, reverse=True)  # neighbour n on top, tenant 0 at the bottom
        for row, tenant in enumerate(rows, start=1):
            y = len(rows) - row + 1
            for request in neighbours[tenant]:
                start, stop = request["t_send"] - t0, request["t_last_token"] - t0
                ax.add_patch(
                    plt.Rectangle(
                        (start, y - 0.28), stop - start, 0.56, fc=ps.CONTEXT, ec="white", lw=0.4
                    )
                )
        late = 0
        for r in cell["records"]:
            obs, deadline = r.t_arrival - t0, r.t_arrival - t0 + mark_s
            ax.plot([deadline] * 2, [-0.36, 0.36], color=ps.INK, lw=ps.LW, ls=ps.DEADLINE_LS)
            if r.t_last_token is None:  # never ran
                late += 1
                continue
            start, stop = r.t_send - t0, r.t_last_token - t0
            missed = r.t_last_token - r.t_arrival > mark_s
            late += missed
            if start - obs > 0.05:  # waited in the mailbox
                ax.add_patch(
                    plt.Rectangle(
                        (obs, -0.28),
                        start - obs,
                        0.56,
                        fc="white",
                        ec=ps.ORANGE if missed else ps.BLUE,
                        lw=ps.LW,
                    )
                )
            ax.add_patch(
                plt.Rectangle(
                    (start, -0.28),
                    stop - start,
                    0.56,
                    fc=ps.ORANGE if missed else ps.BLUE,
                    ec="white",
                    lw=0,
                    hatch=ps.HATCH if missed else None,
                )
            )
            ax.text(
                (start + stop) / 2,
                -0.5,
                f"{r.t_last_token - r.t_arrival:.1f}",
                ha="center",
                va="top",
                fontsize=ps.F_SMALL,
                color=ps.INK,
            )
            end = max(end, stop)
        ax.set_yticks(
            range(len(rows), -1, -1),
            [f"neighbour {t}" for t in rows] + ["tenant 0"],
            fontsize=ps.F_TEXT,
        )
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.95, len(rows) + 0.5)
        ax.grid(axis="x")
        n = len(cell["records"])
        ps.panel_title(ax, letter, f"{label}: {late} of {n} frames late at $D$ = {mark_s:g} s")
    axes[-1].set_xlim(-1, end + 1)
    axes[-1].set_xlabel(
        f"time since tenant 0's first frame (s); under each frame: {ps.TE} − {ps.TO} (s)"
    )
    handles = [
        ps.swatch(ps.BLUE),
        ps.swatch(ps.ORANGE, ps.HATCH),
        ps.swatch(ps.CONTEXT),
        plt.Line2D([], [], color=ps.INK, lw=ps.LW, ls=ps.DEADLINE_LS),
    ]
    labels = [
        "tenant 0 frame, on time",
        "tenant 0 frame, late",
        "neighbour request",
        f"deadline of that frame ({ps.TO} + $D$)",
    ]
    fig.legend(
        handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.055), handlelength=1.8
    )
    ps.footnote(fig, ps.D_MEANS)
    fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=0.6)
    ps.save(fig, out / f"timeline_{load}")


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
    parser.add_argument("--max-deadline", type=float, default=6.0)
    parser.add_argument("--mark", type=float, default=3.0, help="deadline to mark (s)")
    args = parser.parse_args()

    loaded = results.load(args.folder)
    cells = [load_cell(loaded, c) for c in loaded.cells]
    figures = args.folder / results.FIGURES
    figures.mkdir(exist_ok=True)
    plot_miss_rates(cells, figures, args.max_deadline, args.mark)
    plot_p99(cells, figures, args.mark)
    plot_timeline(cells, figures, "bursty", args.mark)
    print(summary_table(cells, args.mark))
    print(f"figures: {figures}")


if __name__ == "__main__":
    main()
