# Gate 1: harness against the dummy model (Thor, 2026-11-04) — PASS

Cell `gate1-T1` from `configs/gate1.toml`: 200 frames at T = 1 s, 5 warm-up frames,
sleeps uniform 0.2 to 2.0 s (seed 0), `--engine dummy`, run on the Thor at 120W while a
labmate's idle deploy server held 18 GB of GPU memory (load average 3.7, GPU 0%).
Checker output: `checks-thor.json`. The raw cell (`gate1-T1-20261104-014448`) was kept
only in gitignored `runs/` directories and is no longer on the Thor.

| Check | Result | Pass |
|---|---|---|
| (a) t_obs spacing | mean 999.999 ms, std 0.49 ms, worst tick 1.6 ms off, 0 emitter faults | yes |
| (b) overwritten frames | replay predicts 35, log has 35 | yes |
| (c) miss rate at D = T | replay 84.0%, log 84.0% (naive "sleeps over T" would say 59.5%) | yes |
| (d) sampler/frame interleave | 20,546 rows cover every frame, gap median 10.0 ms, max 21 ms | yes |

The same cell on a MacBook failed (a) because macOS descheduled the process
(below); on Linux the single-process emitter holds the schedule to
half a millisecond, so no separate emitter process or ZeroMQ is needed.

Next: gate 2, the real model on 50 tiles back to back, after the keep/drop probe.

**What is stored here.** Both raw cells are lost, so this folder breaks the results rule:
`checks-thor.json` and `checks-mac.json` are the `harness.gate1` checker's output (derived
numbers), not measurements. Rerunning gate 1 (`configs/gate1.toml --engine dummy`, about four
minutes) and collecting the cell would replace them with raw data.

## Earlier: macOS dry run (2026-11-01)

Cell `gate1-T1` from `configs/gate1.toml`: 200 frames at T = 1 s, 5 warm-up frames,
sleeps uniform 0.2 to 2.0 s (seed 0), run on a MacBook with `--engine dummy`.
Checker output: `checks-mac.json`.

| Check | Result | Pass |
|---|---|---|
| (a) t_obs spacing | mean 1000.005 ms, std 21.7 ms, worst tick 188 ms late, 1 emitter fault | no |
| (b) overwritten frames | replay predicts 35, log has 35 | yes |
| (c) miss rate at D = T | replay 84.0%, log 85.5% (naive "sleeps over T" would say 59.5%) | yes |
| (d) sampler/frame interleave | 20,546 sampler rows cover every frame, median gap 9.9 ms | yes |

(a) fails for a reason outside the harness: the sampler thread shows the same stalls
(22 gaps over 50 ms, up to 256 ms) at the same moments as the emitter, so the whole
process was being descheduled by macOS. Excluding ticks off by more than 5 ms, the
spacing std is 1.00 ms. The laptop also slept for about 2.5 h during the run
(wall clock 17:18 to 19:52 for a 3.5-minute cell), which the monotonic clock hides and
the manifest's wall-clock stamps expose. Check (a) must be redone on the Thor, where the
old feeder already woke within 1 ms on Linux.

(c) as written in DESIGN.md ("fraction of sleeps longer than T") does not hold at this
cadence: mean sleep 1.1 s exceeds T, frames wait in the mailbox, and the miss rate is
85%, not 60%. The checker compares against a replay of the same sleep draws through a
model of the mailbox, which matches the overwrite count exactly.

Next: rerun `gate1.toml` on the Thor and expect (a) to pass; then gate 2 with the real model.
