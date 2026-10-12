"""Turn a run directory into numbers and a CDF plot, from the logs alone.

    uv run python -m harness.report /data/projects/pulak/runs/<run>

Reads records.jsonl, power.csv and run.json. Writes summary.json and
latency_cdf.png next to them and prints a short table.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; the Thor runs headless
import matplotlib.pyplot as plt
import numpy as np

from harness.device import PowerSample, read_power_csv
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


def summarize(records: list[RequestRecord], power: list[PowerSample]) -> dict:
    answered = [r for r in records if r.error is None]
    first_send = min(r.t_send for r in answered)
    last_answer = max(r.t_last_token for r in answered)

    summary = {
        "n_requests": len(records),
        "n_errors": len(records) - len(answered),
        "latency_ms": percentiles_ms([r.latency_s for r in answered]),
        "time_to_first_token_ms": percentiles_ms(
            [r.t_first_token - r.t_arrival for r in answered if r.t_first_token is not None]
        ),
        # Requests finished per second while the run was going.
        # Meaningful in throughput mode; in latency mode it is set by the input period.
        "throughput_rps": round(len(answered) / (last_answer - first_send), 4),
        "energy_per_request_j": {
            "gpu": mean_energy_per_request(answered, power, "gpu_w"),
            "board": mean_energy_per_request(answered, power, "board_w"),
        },
    }

    # How much of the latency the engine itself accounts for.
    # The rest is image upload, queueing in the server, and client overhead.
    with_engine_timing = [r for r in answered if r.engine_prompt_ms is not None]
    if with_engine_timing:

        def engine_s(r: RequestRecord) -> float:
            return (r.engine_prompt_ms + r.engine_decode_ms) / 1000

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
    power = read_power_csv(run_dir / "power.csv")

    summary = {"run": run_dir.name, "mode": config["mode"]} | summarize(records, power)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_latency_cdf(records, config.get("period_s"), run_dir / "latency_cdf.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(report(parser.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
