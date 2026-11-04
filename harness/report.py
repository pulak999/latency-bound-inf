"""Turn a cell (or run) directory into numbers and plots, from the logs alone.

    uv run python -m harness.report /data/projects/rohan/runs/<cell>

Reads records.jsonl, power.csv and manifest.json (or run.json from the older
runner). Writes summary.json, latency_cdf.png, miss_vs_deadline.png and
timeline.png next to them and prints the summary.

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


def plot_latency_cdf(records: list[RequestRecord], period_s: float | None, path: Path) -> None:
    """Share of all frames finished by each latency, with the deadline marked if there is one.

    Overwritten and failed frames never finish, so the curve stops below 1.
    """
    latencies_ms = np.sort([r.latency_s * 1000 for r in answered_records(records)])
    share_finished = np.arange(1, len(latencies_ms) + 1) / len(records)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(latencies_ms, share_finished, where="post")
    if period_s is not None:
        ax.axvline(period_s * 1000, color="red", linestyle="--", label=f"D = T = {period_s:g} s")
        ax.legend()
    ax.set_xlabel("latency, arrival to last token (ms)")
    ax.set_ylabel("share of frames finished")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_miss_rate_vs_deadline(
    records: list[RequestRecord], period_s: float | None, path: Path
) -> None:
    latencies = [r.latency_s for r in answered_records(records)]
    longest = max(latencies, default=1.0)
    max_deadline = max(longest * 1.5, (period_s or 0) * 1.5)
    deadlines, rates = miss_rate_curve(records, max_deadline)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(deadlines, rates)
    if period_s is not None:
        ax.axvline(period_s, color="red", linestyle="--", label=f"D = T = {period_s:g} s")
        ax.legend()
    ax.set_xlabel("deadline D (s)")
    ax.set_ylabel("miss rate")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeline(
    records: list[RequestRecord],
    power: list[PowerSample],
    gpu_util: list[tuple[float, float]],
    period_s: float | None,
    path: Path,
) -> None:
    """Per-frame latency over the cell, with GPU clock and load beneath it.

    A slow frame can be matched by eye to a clock drop or a load spike.
    """
    t0 = min(r.t_arrival for r in records)
    fig, (ax_lat, ax_dev) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ok = answered_records(records)
    ax_lat.plot(
        [r.t_arrival - t0 for r in ok], [r.latency_s * 1000 for r in ok], ".", label="latency"
    )
    dropped = [r for r in records if r.outcome != "ok"]
    if dropped:
        ax_lat.plot(
            [r.t_arrival - t0 for r in dropped],
            [0] * len(dropped),
            "x",
            color="red",
            label="overwritten / error",
        )
    if period_s is not None:
        ax_lat.axhline(period_s * 1000, color="red", linestyle="--", label="D = T")
    ax_lat.set_ylabel("latency (ms)")
    ax_lat.legend(loc="upper right")
    ax_lat.grid(alpha=0.3)

    clocks = [(s.t - t0, s.gpu_mhz) for s in power if s.gpu_mhz is not None]
    if clocks:
        ax_dev.plot(*zip(*clocks, strict=True), color="tab:green", linewidth=0.8, label="GPU MHz")
        ax_dev.set_ylabel("GPU clock (MHz)")
    if gpu_util:
        ax_util = ax_dev.twinx()
        ax_util.plot(
            [t - t0 for t, _ in gpu_util],
            [u for _, u in gpu_util],
            color="tab:gray",
            linewidth=0.6,
            alpha=0.7,
            label="GPU load %",
        )
        ax_util.set_ylabel("GPU load (%)")
        ax_util.set_ylim(0, 100)
    if not clocks and not gpu_util:
        ax_dev.text(
            0.5, 0.5, "no GPU clock or load samples", ha="center", transform=ax_dev.transAxes
        )
    ax_dev.set_xlabel("time since first frame (s)")
    ax_dev.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def report(run_dir: Path) -> dict:
    """Summarize a directory, write summary.json and the plots, and return the summary."""
    manifest, records, power = load_cell(run_dir)
    period_s = manifest.get("period_s")

    summary = {"run": run_dir.name, "cell_id": manifest.get("cell_id")} | summarize(records, power)
    if period_s is not None and records:
        summary["miss_rate_at_cadence"] = round(miss_rate_at(records, period_s), 4)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if answered_records(records):
        plot_latency_cdf(records, period_s, run_dir / "latency_cdf.png")
        plot_miss_rate_vs_deadline(records, period_s, run_dir / "miss_vs_deadline.png")
        gpu_util = read_gpu_utilization_csv(run_dir / "gpu_util.csv")
        plot_timeline(records, power, gpu_util, period_s, run_dir / "timeline.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(report(parser.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
