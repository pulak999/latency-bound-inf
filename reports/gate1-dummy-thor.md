# Gate 1: harness against the dummy model (Thor, 2026-11-04) — PASS

Cell `gate1-T1` from `configs/gate1.toml`: 200 frames at T = 1 s, 5 warm-up frames,
sleeps uniform 0.2 to 2.0 s (seed 0), `--engine dummy`, run on the Thor at 120W while a
labmate's idle deploy server held 18 GB of GPU memory (load average 3.7, GPU 0%).
Full numbers in `gate1-dummy-thor.json`; the cell directory is `runs/thor/gate1-T1-20261104-014448`
on the Mac and `~/latency-bound-inf/runs/` on the Thor.

| Check | Result | Pass |
|---|---|---|
| (a) t_obs spacing | mean 999.999 ms, std 0.49 ms, worst tick 1.6 ms off, 0 emitter faults | yes |
| (b) overwritten frames | replay predicts 35, log has 35 | yes |
| (c) miss rate at D = T | replay 84.0%, log 84.0% (naive "sleeps over T" would say 59.5%) | yes |
| (d) sampler/frame interleave | 20,546 rows cover every frame, gap median 10.0 ms, max 21 ms | yes |

The same cell on a MacBook failed (a) because macOS descheduled the process
(`gate1-dummy-mac.md`); on Linux the single-process emitter holds the schedule to
half a millisecond, so no separate emitter process or ZeroMQ is needed.

Next: gate 2, the real model on 50 tiles back to back, after the keep/drop probe.
