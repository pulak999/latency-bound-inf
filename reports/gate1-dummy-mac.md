# Gate 1: harness against the dummy model (macOS dry run, 2026-11-01)

Cell `gate1-T1` from `configs/gate1.toml`: 200 frames at T = 1 s, 5 warm-up frames,
sleeps uniform 0.2 to 2.0 s (seed 0), run on a MacBook with `--engine dummy`.
Full numbers in `gate1-dummy-mac.json`.

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
