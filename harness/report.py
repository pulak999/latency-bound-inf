"""Turn a run directory into numbers and a CDF plot, from the logs alone.

    uv run python -m harness.report /data/projects/pulak/runs/<run>

Reads records.jsonl, device.csv (power.csv in older runs) and run.json. Writes
summary.json and latency_cdf.png next to them and prints the summary.

Besides latency percentiles, the summary answers where the variation comes from:
- `stages_ms` splits each request into consecutive stages that add up to its latency.
- `tail_ms` compares the slowest 1% of requests with the median request, stage by stage.
- `gpu_clock_mhz` gives the GPU clock when requests were sent and while they ran.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; the Thor runs headless
import matplotlib.pyplot as plt
import numpy as np

from harness.device import DeviceSample, read_device_samples
from harness.records import RequestRecord, read_records


class DeviceTrace:
    """A run's power and clock samples as arrays, for fast lookups by time window."""

    def __init__(self, samples: list[DeviceSample]):
        self.t = np.array([s.t for s in samples])
        self.power_w = {
            "gpu": np.array([s.gpu_w for s in samples]),
            "board": np.array([s.board_w for s in samples]),
        }
        clocks = [s.gpu_mhz for s in samples]
        self.gpu_mhz = None if None in clocks else np.array(clocks)

    def _window(self, t_start: float, t_end: float) -> slice:
        first, last = np.searchsorted(self.t, [t_start, t_end], side="left")
        return slice(first, last)

    def energy_j(self, rail: str, t_start: float, t_end: float) -> float:
        """Energy on one rail ('gpu' or 'board') between two monotonic times.

        Mean power of the samples in the window times its length. Samples are
        10 ms apart, so a 200 ms request has about twenty of them.
        """
        in_window = self.power_w[rail][self._window(t_start, t_end)]
        if len(in_window) == 0:
            return float("nan")
        return float(in_window.mean()) * (t_end - t_start)

    def gpu_clock_at(self, t: float) -> float:
        """GPU clock in MHz at the last sample taken at or before t."""
        index = np.searchsorted(self.t, t, side="right") - 1
        return float(self.gpu_mhz[index]) if index >= 0 else float("nan")

    def median_gpu_clock(self, t_start: float, t_end: float) -> float:
        in_window = self.gpu_mhz[self._window(t_start, t_end)]
        return float(np.median(in_window)) if len(in_window) else float("nan")


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion. Stays sensible when the count is 0 or n."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    centre = (p + z**2 / (2 * trials)) / (1 + z**2 / trials)
    half_width = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / (1 + z**2 / trials)
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def percentiles(values: list[float]) -> dict:
    """p50, p99, their ratio, and the max, in the units given."""
    values = np.array(values)
    p50, p99 = np.percentile(values, [50, 99])
    return {
        "p50": round(float(p50), 1),
        "p99": round(float(p99), 1),
        "p99_over_p50": round(float(p99 / p50), 3) if p50 else None,
        "max": round(float(values.max()), 1),
    }


def percentiles_ms(values_s: list[float]) -> dict:
    """`percentiles` of values given in seconds, reported in milliseconds."""
    return percentiles([v * 1000 for v in values_s])


def stages_ms(record: RequestRecord) -> dict[str, float]:
    """Split one request's latency into consecutive stages that add up to it.

    wait            arrival to send: the feeder's timing, or queueing behind an earlier request
    engine stages   as the engine reports them: prompt and decode for llama-server
                    (prompt includes encoding the image), preprocess / inference /
                    postprocess for a robot policy
    outside_engine  the rest: transport, server-side queueing, client overhead
    """
    stages = {"wait": (record.t_send - record.t_arrival) * 1000}
    if record.engine_stages_ms:
        stages |= record.engine_stages_ms
    elif record.engine_prompt_ms is not None:
        stages |= {"prompt": record.engine_prompt_ms, "decode": record.engine_decode_ms}
    engine_ms = record.engine_total_ms or 0.0
    stages["outside_engine"] = (record.t_last_token - record.t_send) * 1000 - engine_ms
    return stages


def slowest_one_percent(records: list[RequestRecord]) -> list[RequestRecord]:
    """The slowest 1% of requests by latency, and at least one."""
    n_slowest = max(1, math.ceil(len(records) / 100))
    return sorted(records, key=lambda r: r.latency_s)[-n_slowest:]


def tail_by_stage(records: list[RequestRecord]) -> dict:
    """How much longer each stage takes in the slowest 1% of requests than in a median request.

    The stage deltas add up to the gap between the slowest requests' mean latency
    and the median latency, so the largest delta says where the tail comes from.
    """
    slowest_records = slowest_one_percent(records)
    all_stages = [stages_ms(r) for r in records]
    slowest_stages = [stages_ms(r) for r in slowest_records]

    def slowest_minus_median(values_all: list[float], values_slowest: list[float]) -> float:
        return round(float(np.mean(values_slowest)) - float(np.median(values_all)), 1)

    return {
        "slowest_1pct_mean_minus_median": slowest_minus_median(
            [r.latency_s * 1000 for r in records], [r.latency_s * 1000 for r in slowest_records]
        ),
        "by_stage": {
            stage: slowest_minus_median(
                [s[stage] for s in all_stages], [s[stage] for s in slowest_stages]
            )
            for stage in all_stages[0]
        },
    }


def gpu_clocks(records: list[RequestRecord], device: DeviceTrace) -> dict:
    """The GPU clock when each request was sent and while it ran, in MHz."""
    at_send = [device.gpu_clock_at(r.t_send) for r in records]
    during = [device.median_gpu_clock(r.t_send, r.t_last_token) for r in records]
    slowest_during = [
        device.median_gpu_clock(r.t_send, r.t_last_token) for r in slowest_one_percent(records)
    ]
    return {
        "at_send": percentiles(at_send),
        "during_request": percentiles(during),
        "during_slowest_1pct": round(float(np.mean(slowest_during)), 1),
    }


def summarize(records: list[RequestRecord], device: DeviceTrace) -> dict:
    answered = [r for r in records if r.error is None]
    first_send = min(r.t_send for r in answered)
    last_answer = max(r.t_last_token for r in answered)

    def mean_energy(rail: str) -> float:
        per_request = [device.energy_j(rail, r.t_send, r.t_last_token) for r in answered]
        return round(float(np.mean(per_request)), 2)

    summary = {
        "n_requests": len(records),
        "n_errors": len(records) - len(answered),
        "latency_ms": percentiles_ms([r.latency_s for r in answered]),
        # Requests finished per second while the run was going.
        # Meaningful in throughput mode; in latency mode it is set by the input period.
        "throughput_rps": round(len(answered) / (last_answer - first_send), 4),
        "energy_per_request_j": {"gpu": mean_energy("gpu"), "board": mean_energy("board")},
    }

    # Only streaming engines have a first token separate from the last.
    first_token_s = [r.t_first_token - r.t_arrival for r in answered if r.t_first_token is not None]
    if first_token_s:
        summary["time_to_first_token_ms"] = percentiles_ms(first_token_s)

    with_engine_timing = [r for r in answered if r.engine_total_ms is not None]
    if with_engine_timing:
        summary["engine_ms"] = percentiles([r.engine_total_ms for r in with_engine_timing])

    per_request_stages = [stages_ms(r) for r in answered]
    summary["stages_ms"] = {
        stage: percentiles([s[stage] for s in per_request_stages])
        for stage in per_request_stages[0]
    }
    summary["tail_ms"] = tail_by_stage(answered)

    if device.gpu_mhz is not None:
        summary["gpu_clock_mhz"] = gpu_clocks(answered, device)

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
    """Share of requests finished by each latency, with the deadline marked if there is one."""
    latencies_ms = np.sort([r.latency_s * 1000 for r in records if r.error is None])
    share_finished = np.arange(1, len(latencies_ms) + 1) / len(latencies_ms)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(latencies_ms, share_finished, where="post")
    if period_s is not None:
        ax.axvline(period_s * 1000, color="red", linestyle="--", label="deadline")
        ax.legend()
    ax.set_xlabel("latency, arrival to last token (ms)")
    ax.set_ylabel("share of requests finished")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(run_dir: Path) -> dict:
    """Summarize a run directory, write summary.json and the CDF plot, and return the summary."""
    config = json.loads((run_dir / "run.json").read_text())
    records = read_records(run_dir / "records.jsonl")
    device = DeviceTrace(read_device_samples(run_dir))

    summary = {"run": run_dir.name, "mode": config["mode"]} | summarize(records, device)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_latency_cdf(records, config.get("period_s"), run_dir / "latency_cdf.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(report(parser.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
