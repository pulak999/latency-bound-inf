"""Statistics and plots of one cell, computed from its raw logs and never stored with them.

    uv run python -m harness.report /data/projects/rohan/runs/<cell> [--figures DIR]

Reads records.jsonl, power.csv and manifest.json (or run.json from the older
runner) and prints the summary. With --figures, also draws latency_cdf.png,
miss_vs_deadline.png and timeline.png (each also as .pdf, in the paper style of
harness/paper_style.py) into DIR. Nothing is written into the
cell directory: it holds only what was measured.

The deadline is applied here, not during the run: one log gives the miss rate
at every deadline D, and the default reported D is the cadence T.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; the Thor runs headless
import matplotlib.pyplot as plt
import numpy as np

from harness import paper_style as ps
from harness.device import PowerSample, read_gpu_utilization_csv, read_power_csv
from harness.records import RequestRecord, read_records


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion. Stays sensible when the count is 0 or n."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    centre = (p + z**2 / (2 * trials)) / (1 + z**2 / trials)
    half_width = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / (1 + z**2 / trials)
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def percentiles_ms(values_s: list[float]) -> dict:
    """p50, p99, their ratio, and the max, in milliseconds."""
    values_ms = np.array(values_s) * 1000
    p50, p99 = np.percentile(values_ms, [50, 99])
    return {
        "p50": round(float(p50), 1),
        "p99": round(float(p99), 1),
        "p99_over_p50": round(float(p99 / p50), 3),
        "max": round(float(values_ms.max()), 1),
    }


def mean_max_ms(values_ms: list[float]) -> dict:
    return {"mean": round(float(np.mean(values_ms)), 1), "max": round(float(np.max(values_ms)), 1)}


def energy_joules(samples: list[PowerSample], t_start: float, t_end: float, rail: str) -> float:
    """Energy on one rail ('gpu_w' or 'board_w') between two monotonic times.

    Mean power of the samples in the window times its length. Samples are 10 ms
    apart, so a request lasting over a second has around a hundred of them.
    """
    in_window = [getattr(s, rail) for s in samples if t_start <= s.t <= t_end]
    if not in_window:
        return float("nan")
    return float(np.mean(in_window)) * (t_end - t_start)


def mean_energy_per_request(
    answered: list[RequestRecord], power: list[PowerSample], rail: str
) -> float:
    """Average energy per request on one rail, from send to last token."""
    per_request = [energy_joules(power, r.t_send, r.t_last_token, rail) for r in answered]
    return round(float(np.mean(per_request)), 2)


def answered_records(records: list[RequestRecord]) -> list[RequestRecord]:
    return [r for r in records if r.outcome == "ok"]


def miss_rate_at(records: list[RequestRecord], deadline_s: float) -> float:
    """Share of frames that were overwritten, failed, or finished later than deadline_s."""
    misses = sum(1 for r in records if r.latency_s is None or r.latency_s > deadline_s)
    return misses / len(records)


def miss_rate_curve(
    records: list[RequestRecord], max_deadline_s: float, n_points: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Miss rate as a function of deadline, from 0 to max_deadline_s."""
    deadlines = np.linspace(0, max_deadline_s, n_points)
    return deadlines, np.array([miss_rate_at(records, d) for d in deadlines])


def summarize(records: list[RequestRecord], power: list[PowerSample]) -> dict:
    answered = answered_records(records)
    by_outcome = {outcome: sum(r.outcome == outcome for r in records) for outcome in ("ok",)}
    by_outcome["overwritten"] = sum(r.outcome == "overwritten" for r in records)
    by_outcome["error"] = sum(r.outcome == "error" for r in records)
    summary = {"n_requests": len(records), "n_errors": by_outcome["error"], "outcomes": by_outcome}
    # Every emitted frame, served or not: the feeder's lateness does not depend on the model.
    feeder_late = [r.feeder_late_s for r in records if r.feeder_late_s is not None]
    if feeder_late:
        summary["feeder_late_ms"] = percentiles_ms(feeder_late)
    if not answered:
        return summary

    first_send = min(r.t_send for r in answered)
    last_answer = max(r.t_last_token for r in answered)
    summary |= {
        "latency_ms": percentiles_ms([r.latency_s for r in answered]),
        "waiting_ms": mean_max_ms([r.waiting_s * 1000 for r in answered]),
        # Requests finished per second while the run was going.
        # Meaningful in throughput mode; in latency mode it is set by the input period.
        "throughput_rps": round(len(answered) / (last_answer - first_send), 4),
        "energy_per_request_j": {
            "gpu": mean_energy_per_request(answered, power, "gpu_w"),
            "board": mean_energy_per_request(answered, power, "board_w"),
        },
    }

    clocks = [s.gpu_mhz for s in power if s.gpu_mhz is not None]
    if clocks:
        summary["gpu_mhz"] = {"mean": round(float(np.mean(clocks)), 0), "min": min(clocks)}

    # Only streaming engines have a first token separate from the last.
    first_token_s = [r.t_first_token - r.t_arrival for r in answered if r.t_first_token is not None]
    if first_token_s:
        summary["time_to_first_token_ms"] = percentiles_ms(first_token_s)

    prefill = [r.engine_prompt_ms for r in answered if r.engine_prompt_ms is not None]
    if prefill:
        summary["prefill_ms"] = mean_max_ms(prefill)

    # How much of the latency the engine itself accounts for.
    # The rest is image upload, queueing in the server, and client overhead.
    with_engine_timing = [r for r in answered if r.engine_total_ms is not None]
    if with_engine_timing:

        def engine_s(r: RequestRecord) -> float:
            return r.engine_total_ms / 1000

        def outside_engine_s(r: RequestRecord) -> float:
            return (r.t_last_token - r.t_send) - engine_s(r)

        summary["engine_ms"] = percentiles_ms([engine_s(r) for r in with_engine_timing])
        summary["outside_engine_ms"] = percentiles_ms(
            [outside_engine_s(r) for r in with_engine_timing]
        )

    with_deadline = [r for r in records if r.deadline is not None]
    if with_deadline:
        n_missed = sum(r.missed for r in with_deadline)
        low, high = wilson_interval(n_missed, len(with_deadline))
        summary["deadline_misses"] = {
            "missed": n_missed,
            "of": len(with_deadline),
            "rate": round(n_missed / len(with_deadline), 4),
            "ci95": [round(low, 4), round(high, 4)],
        }
    return summary


def mark_deadline(ax, period_s: float | None, vertical: bool = True) -> None:
    """The cell's deadline D = T, dotted black, labelled at the axis edge."""
    if period_s is None:
        return
    if vertical:
        ax.axvline(period_s, color=ps.INK, lw=ps.LW, ls=ps.DEADLINE_LS)
        ax.text(
            period_s,
            1.0,
            f" $D$ = $T$ = {period_s:g} s",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=ps.F_SMALL,
            color=ps.INK,
        )
    else:
        ax.axhline(period_s, color=ps.INK, lw=ps.LW, ls=ps.DEADLINE_LS)
        ax.text(
            0.0,
            period_s,
            f" $D$ = $T$ = {period_s:g} s",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=ps.F_SMALL,
            color=ps.INK,
        )


def definitions(period_s: float | None) -> tuple[str, ...]:
    """What D and T mean, for the figures that show them."""
    return (ps.D_MEANS, ps.T_MEANS) if period_s is not None else ()


def plot_latency_cdf(records: list[RequestRecord], period_s: float | None, path: Path) -> None:
    """Share of all frames finished by each latency, with the deadline marked if there is one.

    Overwritten and failed frames never finish, so the curve stops below 100 %.
    """
    latencies = np.sort([r.latency_s for r in answered_records(records)])
    share_finished = 100 * np.arange(1, len(latencies) + 1) / len(records)
    notes = definitions(period_s)

    fig, ax = plt.subplots(figsize=(ps.COLUMN_IN, 2.0 + 0.2 * len(notes)))
    ax.step(latencies, share_finished, where="post", color=ps.BLUE, lw=1.1)
    mark_deadline(ax, period_s)
    ax.set_xlabel(f"latency, {ps.TE} − {ps.TO} (s)")
    ax.set_ylabel("frames finished (%)")
    ax.set_ylim(0, 103)
    ax.grid(axis="y")
    printed = ps.footnote(fig, *notes) if notes else 0
    fig.set_figheight(2.0 + 0.11 * printed)
    fig.tight_layout(rect=(0, 0.11 * printed / fig.get_figheight(), 1, 1))
    ps.save(fig, path)


def plot_miss_rate_vs_deadline(
    records: list[RequestRecord], period_s: float | None, path: Path
) -> None:
    """Share of frames (overwritten and failed included) that miss each deadline D."""
    latencies = [r.latency_s for r in answered_records(records)]
    longest = max(latencies, default=1.0)
    max_deadline = max(longest * 1.5, (period_s or 0) * 1.5)
    deadlines, rates = miss_rate_curve(records, max_deadline)
    notes = definitions(period_s) or (ps.D_MEANS,)

    fig, ax = plt.subplots(figsize=(ps.COLUMN_IN, 2.0 + 0.2 * len(notes)))
    ax.step(deadlines, 100 * rates, where="post", color=ps.BLUE, lw=1.1)
    mark_deadline(ax, period_s)
    ax.set_xlabel("deadline $D$ (s)")
    ax.set_ylabel("frames that miss $D$ (%)")
    ax.set_ylim(-2, 103)
    ax.grid(axis="y")
    printed = ps.footnote(fig, *notes)
    fig.set_figheight(2.0 + 0.11 * printed)
    fig.tight_layout(rect=(0, 0.11 * printed / fig.get_figheight(), 1, 1))
    ps.save(fig, path)


def plot_timeline(
    records: list[RequestRecord],
    power: list[PowerSample],
    gpu_util: list[tuple[float, float]],
    period_s: float | None,
    path: Path,
) -> None:
    """Per-frame latency over the cell, with GPU clock and GPU load in panels beneath it.

    A slow frame can be matched by eye to a clock drop or a load spike. Frames
    later than D = T are orange; overwritten or failed frames are marked at the
    bottom of panel (a).
    """
    t0 = min(r.t_arrival for r in records)
    notes = definitions(period_s)
    fig, (ax_lat, ax_clock, ax_load) = plt.subplots(
        3,
        1,
        figsize=(ps.WIDTH_IN, 3.6 + 0.15 * len(notes)),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    ok = answered_records(records)
    late = [r for r in ok if period_s is not None and r.latency_s > period_s]
    on_time = [r for r in ok if r not in late]
    ax_lat.scatter(
        [r.t_arrival - t0 for r in on_time],
        [r.latency_s for r in on_time],
        s=7,
        color=ps.BLUE,
        lw=0,
        label="frame, on time",
        zorder=3,
    )
    if late:
        ax_lat.scatter(
            [r.t_arrival - t0 for r in late],
            [r.latency_s for r in late],
            s=7,
            color=ps.ORANGE,
            lw=0,
            label="frame, late",
            zorder=3,
        )
    dropped = [r for r in records if r.outcome != "ok"]
    if dropped:
        ax_lat.scatter(
            [r.t_arrival - t0 for r in dropped],
            [0] * len(dropped),
            s=12,
            marker="x",
            color=ps.ORANGE,
            lw=0.8,
            label="overwritten or failed",
            zorder=3,
        )
    mark_deadline(ax_lat, period_s, vertical=False)
    tallest = max([r.latency_s for r in ok] + [period_s or 0.0])
    ax_lat.set_ylim(0, 1.18 * tallest)  # room above the deadline for its label
    ax_lat.set_ylabel("latency (s)")
    ax_lat.grid(axis="y")
    ax_lat.legend(loc="upper right", ncol=3, handletextpad=0.3)
    ps.panel_title(ax_lat, "a", f"every frame, {ps.TE} − {ps.TO}")

    clocks = [(s.t - t0, s.gpu_mhz) for s in power if s.gpu_mhz is not None]
    if clocks:
        ax_clock.plot(*zip(*clocks, strict=True), color=ps.INK, lw=0.6)
    else:
        ax_clock.text(
            0.5,
            0.5,
            "no GPU clock samples",
            ha="center",
            va="center",
            transform=ax_clock.transAxes,
            color=ps.MUTED,
        )
    ax_clock.set_ylabel("MHz")
    ax_clock.grid(axis="y")
    ps.panel_title(ax_clock, "b", "GPU clock")

    if gpu_util:
        ax_load.plot(
            [t - t0 for t, _ in gpu_util], [u for _, u in gpu_util], color=ps.MUTED, lw=0.6
        )
        ax_load.set_ylim(0, 103)
    else:
        ax_load.text(
            0.5,
            0.5,
            "no GPU load samples",
            ha="center",
            va="center",
            transform=ax_load.transAxes,
            color=ps.MUTED,
        )
    ax_load.set_ylabel("%")
    ax_load.grid(axis="y")
    ps.panel_title(ax_load, "c", "GPU load, all processes")
    ax_load.set_xlabel("time since the first frame (s)")
    printed = ps.footnote(fig, *notes) if notes else 0
    fig.set_figheight(3.6 + 0.11 * printed)
    fig.tight_layout(rect=(0, 0.11 * printed / fig.get_figheight(), 1, 1), h_pad=0.4)
    ps.save(fig, path)


def load_cell(run_dir: Path) -> tuple[dict, list[RequestRecord], list[PowerSample]]:
    """The manifest (or older run.json), records and power samples of a directory."""
    if (run_dir / "manifest.json").exists():
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest.setdefault("period_s", manifest.get("cadence_s"))
    else:
        manifest = json.loads((run_dir / "run.json").read_text())
        manifest.setdefault("cell_id", run_dir.name)
    records = read_records(run_dir / "records.jsonl")
    power = read_power_csv(run_dir / "power.csv")
    return manifest, records, power


def report(run_dir: Path, figures_dir: Path | None = None) -> dict:
    """Summarize a directory and return the summary; draw the plots into figures_dir if given."""
    manifest, records, power = load_cell(run_dir)
    period_s = manifest.get("period_s")

    summary = {"run": run_dir.name, "cell_id": manifest.get("cell_id")} | summarize(records, power)
    if period_s is not None and records:
        summary["miss_rate_at_cadence"] = round(miss_rate_at(records, period_s), 4)

    if figures_dir is not None and answered_records(records):
        figures_dir.mkdir(parents=True, exist_ok=True)
        ps.apply()
        plot_latency_cdf(records, period_s, figures_dir / "latency_cdf.png")
        plot_miss_rate_vs_deadline(records, period_s, figures_dir / "miss_vs_deadline.png")
        gpu_util = read_gpu_utilization_csv(run_dir / "gpu_util.csv")
        plot_timeline(records, power, gpu_util, period_s, figures_dir / "timeline.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--figures", type=Path, metavar="DIR", help="also draw the plots here")
    args = parser.parse_args()
    print(json.dumps(report(args.run_dir, args.figures), indent=2))


if __name__ == "__main__":
    main()
