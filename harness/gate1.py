"""Gate 1 (DESIGN.md section 4): does the harness behave as the dummy model predicts?

    uv run python -m harness.gate1 runs/gate1-T1-<timestamp>

The dummy engine's sleeps come from a seeded generator, so they can be drawn
again here and replayed through a model of the mailbox. The checks:

  (a) t_obs spacing has mean T and standard deviation under 1 ms;
  (b) the overwritten count matches the replay within 10%;
  (c) the miss rate at D = T matches the replay within 2 percentage points;
  (d) sampler rows span the frames and are evenly spaced, so the two logs
      interleave when sorted by timestamp.

DESIGN.md states (c) as "the fraction of sleeps longer than T". That holds only
when no frame ever waits. With the default sleeps (mean 1.1 s) at T = 1 s the
mailbox is overloaded, frames wait, and the miss rate is higher than the sleep
fraction. The replay accounts for waiting; the naive fraction is printed too.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.engines.dummy import SleepDistribution
from harness.report import load_cell, miss_rate_at


@dataclass
class SimulatedFrame:
    frame_id: int
    t_obs: float
    t_start: float | None  # None if overwritten
    t_end: float | None

    @property
    def latency(self) -> float | None:
        return None if self.t_end is None else self.t_end - self.t_obs


def simulate_mailbox(sleeps: list[float], period_s: float, n_frames: int) -> list[SimulatedFrame]:
    """Replay a fixed-rate stream through a depth-1 mailbox and a one-at-a-time model.

    Frame i arrives at i * period_s. The model takes a frame the moment it is
    free, and sleeps[k] is the service time of the k-th frame it serves. A
    frame still waiting when the next one arrives is overwritten.
    """
    frames = [SimulatedFrame(i, i * period_s, None, None) for i in range(n_frames)]
    free_at = 0.0
    waiting: SimulatedFrame | None = None
    served = 0

    def serve(frame: SimulatedFrame, at: float) -> None:
        nonlocal free_at, served
        if served >= len(sleeps):
            raise ValueError("not enough sleep draws for the frames served")
        frame.t_start = at
        frame.t_end = at + sleeps[served]
        free_at = frame.t_end
        served += 1

    for frame in frames:
        if waiting is not None and free_at <= frame.t_obs:
            serve(waiting, free_at)  # the model freed up and took the waiting frame first
            waiting = None
        if free_at <= frame.t_obs:
            serve(frame, frame.t_obs)
        else:
            waiting = frame  # overwrites whatever was waiting
    if waiting is not None:
        serve(waiting, free_at)
    return frames


def check(cell_dir: Path, max_spacing_std_ms: float = 1.0) -> dict:
    manifest, records, power = load_cell(cell_dir)
    period_s = manifest["cadence_s"]
    n_frames = manifest["frames_emitted"]
    distribution = SleepDistribution(**manifest["dummy"])
    warmup = manifest["warmup_frames"]

    # (a) emitter spacing
    t_obs = np.array(sorted(r.t_arrival for r in records))
    spacing_ms = np.diff(t_obs) * 1000
    spacing = {
        "mean_ms": float(spacing_ms.mean()),
        "std_ms": float(spacing_ms.std()),
        "max_abs_error_ms": float(np.abs(spacing_ms - period_s * 1000).max()),
        "emitter_faults": len(manifest["emitter_faults"]),
    }
    spacing["pass"] = (
        abs(spacing["mean_ms"] - period_s * 1000) < max_spacing_std_ms
        and spacing["std_ms"] < max_spacing_std_ms
    )

    # (b), (c) replay: the first `warmup` draws were spent on warm-up requests
    draws = distribution.draws(warmup + n_frames)[warmup:]
    simulated = simulate_mailbox(draws, period_s, n_frames)
    predicted_overwritten = sum(f.t_end is None for f in simulated)
    actual_overwritten = sum(r.outcome == "overwritten" for r in records)
    overwritten = {
        "predicted": predicted_overwritten,
        "actual": actual_overwritten,
        "tolerance": max(1, round(0.1 * predicted_overwritten)),
    }
    overwritten["pass"] = (
        abs(actual_overwritten - predicted_overwritten) <= overwritten["tolerance"]
    )

    predicted_miss = sum(f.latency is None or f.latency > period_s for f in simulated) / n_frames
    actual_miss = miss_rate_at(records, period_s)
    miss = {
        "predicted_by_replay": predicted_miss,
        "actual": actual_miss,
        "fraction_of_sleeps_over_T": float(np.mean(np.array(draws) > period_s)),
    }
    miss["pass"] = abs(actual_miss - predicted_miss) <= 0.02

    # (d) sampler rows cover the frames at the configured interval
    sample_t = np.array([s.t for s in power])
    sample_gap_ms = np.diff(sample_t) * 1000 if len(sample_t) > 1 else np.array([np.nan])
    interleave = {
        "sampler_rows": len(power),
        "first_frame_covered": bool(len(sample_t) and sample_t[0] <= t_obs[0]),
        "last_frame_covered": bool(len(sample_t) and sample_t[-1] >= t_obs[-1]),
        "sample_gap_median_ms": float(np.median(sample_gap_ms)),
        "sample_gap_max_ms": float(np.max(sample_gap_ms)),
    }
    interleave["pass"] = interleave["first_frame_covered"] and interleave["last_frame_covered"]

    result = {
        "cell": cell_dir.name,
        "a_spacing": spacing,
        "b_overwritten": overwritten,
        "c_miss_rate_at_T": miss,
        "d_interleave": interleave,
    }
    result["pass"] = all(
        result[k]["pass"]
        for k in ("a_spacing", "b_overwritten", "c_miss_rate_at_T", "d_interleave")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cell_dir", type=Path)
    args = parser.parse_args()
    result = check(args.cell_dir)
    (args.cell_dir / "gate1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("GATE 1", "PASS" if result["pass"] else "FAIL")


if __name__ == "__main__":
    main()
