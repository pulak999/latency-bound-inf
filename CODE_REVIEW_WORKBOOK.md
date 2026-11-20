# Systems-Engineering Code Review Workbook

Repository: `latency-bound-inf`  
Working branch when this was written: `deadline-harness` at `c4a4e5f`  
Rounds 2–5 refreshed and Round 6 (the code written after Round 1) added on 2026-11-19, against `visualizer` at `a637676`  
Common ancestor with `main`: `7f6c1f6`  
Purpose: learn to judge a measurement system, not merely recognize its files.

Responses, concept explanations, and evolving merge recommendations are tracked in
[`CODE_REVIEW_NOTES.md`](CODE_REVIEW_NOTES.md).

## How to use this workbook

Work through one round at a time. For each question:

1. Write a prediction in the answer space before running the command.
2. Point to the code or raw record that supports your answer.
3. State what observation would falsify your answer.
4. Run the verification command.
5. Only then open the `<details>` verdict.

The verdicts are not intended as an answer key for every design decision. A command can establish facts such as which branch contains a capability or which stage grew during a latency spike. It cannot decide a value judgment such as whether simplicity is worth losing an analysis feature. For those questions, evaluate the argument against the stated experiment.

### Review standard

This repository is a scientific instrument. Judge code in this order:

1. **Validity:** Does the recorded number represent the quantity named?
2. **Reproducibility:** Can the number be regenerated from immutable inputs and raw logs?
3. **Observer effect:** Does measuring the system materially change it?
4. **Failure visibility:** Does an unsupported or contaminated run fail loudly?
5. **Extensibility:** Does the design survive the next planned experiment without duplicating semantics?
6. **Readability:** Can another researcher audit all of the above?

Clean style is useful, but it cannot rescue an invalid measurement.

### Working agreement

When discussing a question with a reviewer, answer in this form:

```text
Claim:
Evidence:
Counterexample or uncertainty:
Decision:
```

Line references below refer to `deadline-harness` unless prefixed with `main:`. They can drift after edits; the function names are the durable landmarks.

---



## System map: from thesis claim to recorded number



### The thesis-level model

The project asks where latency-bound inference fails, why it varies, and what makes it predictable. The experiment is organized around four axes:

- **W — workload/world trace:** observations arrive according to the world's clock.
- **NM — non-model load:** other work shares CPU, memory, and GPU resources.
- **M — model × engine:** model weights and inference runtime.
- **D — deployment:** device, power mode, clocks, and eventually GPU/FPGA placement.

The first concrete experiment narrows this broad thesis to one question:

> If a Gemma 3 4B VLM must classify each satellite tile before the next tile arrives, what fraction of frames are late or never processed, and which part of the system causes the misses?

`DESIGN.md:4-28` states that experiment. `control-folder/notes/core-questions.md:1-15` supplies the broader research arc.

### The causal chain implemented by the code

```text
TOML cell
  │
  ▼
controller creates run + starts telemetry + engine
  │
  ▼
feeder emits Frame at world time t_obs
  │
  ▼
depth-1 Mailbox keeps newest unclaimed frame
  │
  ├── replaced before take() ──> outcome=overwritten
  │
  ▼
engine adapter stamps t_send and returns t_last_token
  │
  ▼
RequestRecord JSONL + device CSVs + manifest
  │
  ▼
offline report applies deadline D and attributes latency
```

The principal measurement is:

```text
latency = t_last_token - t_arrival
missed  = no answer OR t_last_token > deadline
```

This intentionally includes mailbox waiting. `records.py:105-124` implements those semantics.

### Invariants to keep in mind

These are the properties the rest of the workbook audits:

- The emitter follows `start + iT`; it must not drift with prior wake-up errors.
- The world never waits for inference.
- At most one frame waits to be claimed.
- Every emitted frame receives exactly one terminal outcome: `ok`, `overwritten`, or `error`.
- All causal timestamps use one monotonic host clock.
- Input preparation and logging I/O stay outside the timed engine path.
- A run records enough provenance to reject contaminated data later.
- Reporting is regenerable from raw logs; no result is typed into a report manually.

---



## Round 1 — Architecture and branch divergence



### Goal

Understand why there are now two experiment architectures, identify the real capabilities on each branch, and decide what should become authoritative.

### 1.1 Start from the raw branch history

Both branches share `7f6c1f6`, then diverge:

```text
7f6c1f6
├── main:              c2faa2d  stage/tail attribution, clock analysis, sweeps
└── deadline-harness:  63117cb  mailbox/config/controller/gate 1
                       c4a4e5f  real-model probe and gate 2 report
```

Verify it:

```bash
git merge-base main deadline-harness
git log --oneline --decorate --graph --all --max-count=10
```

This matters because `deadline-harness` is not simply “main plus the new harness.” Both branches edited `device.py`, `feeder.py`, `records.py`, `report.py`, and tests independently.

### 1.2 What `deadline-harness` added



#### Experiment authority and configuration

- `DESIGN.md:1-155` defines the deadline-miss experiment and gates.
- `configs/stack_a.toml:1-60` defines every Stack A experimental parameter.
- `harness/config.py:21-160` parses TOML into immutable dataclasses, hashes the file, rejects duplicate cells, and refuses unsupported co-runners and power modes.
- `harness/controller.py:130-227` owns the cell lifecycle, manifests, telemetry, engine, records, and report.

The fail-fast checks at `config.py:82-102` are strong measurement engineering. Silently ignoring `co_runner="heavy"` would produce a plausible but false experiment. Raising `NotImplementedError` prevents that.

#### World semantics

- `harness/mailbox.py:17-53` defines a depth-1 newest-frame slot.
- `harness/feeder.py:86-139` independently emits and consumes frames.
- `harness/records.py:72-94` gives overwritten frames an explicit terminal record.

This changes the experiment fundamentally. The v1 fixed-rate feeder creates every request task and allows requests to queue. The new feeder models a sensor pipeline where stale unclaimed frames are replaced.

#### Validation and real inputs

- `harness/gate1.py:42-143` replays seeded service times through an independent mailbox simulation.
- `harness/probe.py:26-98` checks the prompt, output vocabulary, latency spread, and an SCL-derived label.
- `harness/tiles.py:20-35` loads real Sentinel-2 tiles or deterministic synthetic fallback.
- `setup/tiles_sentinel2.py:31-145` creates RGB/SCL tiles and `index.csv`.

The gate checker is valuable because it tests system behavior against a model, not merely against another code path that shares all the same assumptions.

### 1.3 What was left behind on `main`



#### Stage-level tail attribution

`main:harness/report.py` contains:

- `DeviceTrace` at lines 29-58: vectorized arrays and `np.searchsorted` time-window lookup.
- `stages_ms()` at lines 93-111: decomposes each request into wait, engine stages, and outside-engine time.
- `tail_by_stage()` at lines 118-141: asks which stage explains the slowest 1%.
- `gpu_clocks()` at lines 144-156: compares clock at send, during all requests, and during the slowest requests.

That analysis answers thesis question Q2: *why is latency variable?*

The deadline branch still reports `prefill_ms`, `engine_ms`, and `outside_engine_ms` percentiles (`report.py:120-143`), but it no longer computes the stage contribution to the tail or clock specifically during slow requests.

#### Period sweeps

`main:harness/sweep.py`:

- Defines robot and A1 period matrices at lines 49-69.
- Waits for an apparently idle GPU at lines 80-85.
- Runs each rate as a separate ordinary run at lines 87-106.
- Plots latency and GPU clock against idle gap at lines 133-162.

This probes the hypothesis that longer idle gaps let the GPU clock governor lower clocks, increasing the next request’s latency.

The raw queue log `/data/projects/pulak/runs/sweeps-20261104.log` is also evidence of a design problem: the sweep waited for hours because processes *held GPU contexts*, regardless of whether they were actively using the GPU.

### 1.4 The two orchestration architectures



#### v1: `runner.py` + stack-specific scripts

```text
run_a1.py ─┐
           ├── runner.send_on_schedule()
run_b1.py ─┘       ├── run_fixed_rate()
                   └── run_closed_loop()
```

- `runner.py:23-46` defines common latency/throughput CLI modes.
- `runner.py:65-89` runs a selected schedule and wraps telemetry/reporting.
- `run_a1.py:37-82` builds Stack A requests and records.
- `run_b1.py:41-86` runs LeRobot through a one-worker executor.

Strength: both stacks share record/report plumbing.

Limitation: `run_fixed_rate()` at `feeder.py:50-59` creates every request even when the engine is behind. It models a queue, not a depth-1 sensor mailbox.

#### v2: config-driven `controller.py`

```text
stack_a.toml ──> controller ──> Engine protocol ──> run_mailbox()
```

- `controller.py:46-62` constructs either `DummyEngine` or `LlamaServer`.
- `controller.py:71-127` prepares bodies, warms up, and defines record creation.
- `controller.py:130-194` owns one complete cell lifecycle.

Strength: reproducible experiment cells and correct Stack A deadline semantics.

Limitation: it is presently Stack A-shaped. Its engine protocol returns token-oriented `Reply`, and it knows about model prompt, max tokens, image tiles, and llama-server settings. LeRobot returns `ChunkReply`, has no `prepare(image, prompt, tokens)`, and still lives in v1.

### Question 1 — Which branch should be the integration base?

Do not answer “the newest one.” Choose based on experiment semantics, raw-data compatibility, and cost of preserving capabilities.

**Your answer**

```text
Claim:

Evidence:

What would falsify it:

Merge strategy:
```

**How to verify the factual premises**

```bash
git diff --stat 7f6c1f6..main
git diff --stat 7f6c1f6..deadline-harness
git diff main..deadline-harness -- \
  harness/feeder.py harness/device.py harness/report.py harness/runner.py
```

Staff-engineer verdict

Use `deadline-harness` as the semantic base for the Stack A experiment because its mailbox, cell configuration, manifests, and gates encode the newly approved experimental contract. Port the analysis capabilities from `main` deliberately: `DeviceTrace`, stage/tail attribution, clock correlation, and a revised sweep mechanism.

A blind merge is risky. Both branches changed the same data path for different reasons, and accepting one side wholesale can silently restore queue semantics or discard deadline-aware CDF behavior.



### Question 2 — Is there one source of truth for “run an experiment”?

Trace how a Stack A run and a Stack B run choose:

- output directory,
- schedule semantics,
- warm-up policy,
- telemetry files,
- manifest schema,
- error behavior.

Is the duplication merely temporary, or can two scripts produce identically named metrics with different meanings?

**Your answer**

```text
Shared concepts: runner based

Conflicting semantics: toml, fail fast

Risk to a future reader: diverging branches

Boundary I would make authoritative: i think the toml + fail fast version is more mature
```

**How to verify**

```bash
rg -n "RUNS_DIR|runs_dir|warmup|run_fixed_rate|run_mailbox|run.json|manifest.json" \
  harness/runner.py harness/run_a1.py harness/run_b1.py harness/controller.py \
  harness/config.py
```

Staff-engineer verdict

There are two sources of truth. A v1 “latency” run queues all arrivals and writes `run.json`; a v2 cell overwrites stale frames and writes `manifest.json`. Both eventually expose `latency_ms`, `deadline_misses`, and `throughput_rps`.

That is acceptable during migration only if every report preserves the scheduling policy and schema version. Long term, lifecycle/configuration should be common while workload adapters and scheduling policy remain explicit plug-ins.



### Question 3 — Did `deadline-harness` improve or regress reporting?

Give a two-column answer. A good systems review should resist reducing a branch to “better” or “worse.”

**Your answer**

```text
Capabilities gained: simpler to understand and read, cleaner code

Capabilities lost: stage wise breakdown

What the thesis needs from both: take code quality + capabilities from dealdine and incorporate the ones missed from main
```

**How to verify**

```bash
git diff main..deadline-harness -- harness/report.py
git show main:harness/report.py | rg -n \
  "DeviceTrace|stages_ms|tail_by_stage|gpu_clocks"
rg -n "miss_rate_curve|plot_timeline|outcome|overwritten" \
  harness/report.py harness/records.py
```

Staff-engineer verdict

Gained: overwritten-aware CDFs, arbitrary deadline curves, timeline plots, outcome counts, compatibility with both `manifest.json` and `run.json`, and graceful empty/failed-frame handling.

Lost: stage-level tail decomposition, per-request clock windows, explicit slowest-1% clock comparison, memory-clock sampling, and `sweep.py`.

The correct combined reporter needs both deadline semantics and causal attribution.



### Round 1 exit criterion

Before moving on, you should be able to explain in two minutes:

1. Why `run_fixed_rate` and `run_mailbox` answer different scientific questions.
2. Why `deadline-harness` cannot simply replace `main`.
3. Which module should own experiment lifecycle, which should own scheduling, and which should own workload-specific preparation.

---



## Round 2 — Timing critical path and observer effect



### Goal

Trace every operation around the measured interval and distinguish intended system cost from harness-induced cost.

> **Since this round was written (`c4a4e5f`):**
> - Frames and records now carry `t_scheduled` (the tick's due time) next to `t_arrival` (when it actually fired), and `RequestRecord.feeder_late_s` is their difference. Lateness below the 10%-of-T fault threshold used to be thrown away.
> - Measured feeder lateness: about 0.7 ms on every tick in a quiet test (about 0.4 ms for the OS to wake a sleeping thread, plus about 0.3 ms from asyncio's millisecond timer), and up to about 2 ms under load. Issue #15 covers making it tighter; Question 4 below asks what the deadline should be anchored to.
> - The controller can now start neighbour tenants and MPS around tenant 0 (`harness/tenancy.py`); Round 6 reviews that code. The timed path of tenant 0 itself is unchanged.

### 2.1 The timestamp path

For one successful Stack A frame:

1. `run_mailbox.emit()` sleeps until the scheduled arrival (`feeder.py:110-117`).
2. It records actual publication time as `t_obs = time.monotonic()` (`feeder.py:115`), and keeps the scheduled time as `t_scheduled`.
3. It calls `Mailbox.put()` (`feeder.py:120`).
4. `consume()` awaits `Mailbox.take()` (`feeder.py:132-135`).
5. `controller.send_frame()` creates a co-tenant task, then stamps `t_send` (`controller.py:93-96`).
6. The engine sends the prepared HTTP body and stamps token arrival in `llama_server.send()` (`llama_server.py:152-191`).
7. `RequestRecord.latency_s` subtracts `t_arrival` from `t_last_token` (`records.py:105-111`).

The intended decomposition is:

```text
t_arrival                 t_send                         t_last_token
    │                        │                                │
    ├──── mailbox wait ──────┼──── engine + transport ────────┤
    └──────────────────── total deadline latency ──────────────┘
```



### 2.2 Why `sleep_until` uses two sleeps

`feeder.py:36-47` first sleeps until roughly 50 ms before the deadline, then performs a short final sleep. The code comment says long Linux sleeps may wake up late by up to roughly 0.1% of their duration.

Good choice:

- Arrival times are computed from the original start (`arrival_times`, lines 31-33), so errors do not accumulate.
- The final sleep is short, reducing long-sleep overshoot.

Trade-off:

- `asyncio.sleep` still provides no hard real-time guarantee.
- A busy event loop or OS descheduling can delay the final wake-up.
- The Mac Gate 1 result is direct evidence: spacing std `21.7 ms`, worst error `188 ms`; Thor Linux achieved `0.49 ms` std.

The harness records emitter faults instead of pretending the schedule was perfect.

### 2.3 Mailbox synchronization

`Mailbox` uses one field and one `asyncio.Event` (`mailbox.py:26-53`), with no lock.

That is safe under its current assumptions:

- emitter and consumer execute on the same event loop,
- `put()` contains no `await`,
- `take()` does not yield between checking `_frame` and clearing `_changed`,
- there is exactly one producer and one consumer.

This is not automatically safe if a future implementation calls `put()` from another thread or process. The lack of a lock is a deliberate consequence of the execution model, not a universally thread-safe mailbox.

### 2.4 Precomputation and warm-up

`controller.measure()` builds all request bodies before opening the timed session (`controller.py:80-86`). `llama_server.encode_request()` performs PNG base64 encoding and JSON serialization (`llama_server.py:111-137`).

That is strong measurement hygiene: image serialization is not accidentally attributed to inference.

However, it answers a specific question: *latency of a preprepared request through the serving engine*. A deployed sensor pipeline may need image decode, resize, base64 encoding, or zero-copy transfer. Excluding those costs is valid only if the result is named accordingly.

Warm-ups are sent back-to-back and discarded (`controller.py:88-90`). They initialize CUDA and the model but do not model idle-gap clock behavior before the first measured request.

### 2.5 `--max-minutes`

`stop_at` is an absolute monotonic deadline (`controller.py:216-217`). `run_mailbox()` stops emitting frames whose scheduled arrival is at or after that deadline (`feeder.py:111-114`) and allows the in-flight request to finish.

But the preflight estimate is:

```python
(warmup_frames + cell.frames) * cadence_s + 120
```

at `controller.py:197-198`.

Warm-ups are not cadence-paced; they are back-to-back. At `T=20 s`, 20 warm-ups are estimated as 400 seconds even though the model takes roughly 1.5-1.7 seconds each. This is conservative but can skip a cell that would fit.

More importantly, `--max-minutes` is not a hard process deadline: model startup, hashing, warm-up, and the final in-flight request may cross it. The option's help text says “cut the running one short,” so its exact contract deserves clarification.

### 2.6 Co-tenant sampling and observer effect

`controller.py:93-96` does:

```python
co_tenants = asyncio.create_task(gpu_co_tenants(...))
t_send = time.monotonic()
reply = await send(...)
```

Creating the task does not necessarily execute `nvidia-smi` before `t_send`; it schedules it for the next event-loop opportunity. Once `send()` yields for network I/O, `gpu_co_tenants()` starts a subprocess (`device.py:217-238`) concurrently with the measured request.

This avoids adding subprocess startup directly to `t_send - t_arrival`, but it introduces CPU/process activity during `t_send - t_last_token`. The measurement tool can perturb the thing being measured.

Whether that perturbation is material is an empirical question. The right response is not to assume it is harmless, but to run an A/B measurement with per-request co-tenant sampling on and off.

### Question 1 — Predict mailbox outcomes before running the code

Suppose frames arrive at `0, 1, 2, 3, 4` seconds and every served frame takes `1.5` seconds. For each frame, predict:

- start time,
- end time,
- outcome,
- deadline miss at `D=1 s`.

**Your answer**

```text
frame 0:
frame 1:
frame 2:
frame 3:
frame 4:

Why:
```

**How to verify**

```bash
uv run python - <<'PY'
from harness.gate1 import simulate_mailbox
for f in simulate_mailbox([1.5] * 5, period_s=1.0, n_frames=5):
    print(f.frame_id, f.t_start, f.t_end, f.latency)
PY
```

Verdict

Frame 0 runs `0→1.5`; frame 1 waits and runs `1.5→3`; frame 2 runs `3→4.5`; frame 3 is overwritten by frame 4; frame 4 runs `4.5→6`.

The latencies are `1.5, 2.0, 2.5, None, 2.0`, so all five miss a one-second deadline. Only one frame is overwritten; the others finish late.

This distinction is why “fraction of service times longer than T” does not equal miss rate under queueing.



### Question 2 — Is `gpu_co_tenants()` inside or outside the measurement?

Answer separately for:

- mailbox wait,
- client-perceived inference latency,
- engine-reported `prompt_ms`,
- total GPU conditions during inference.

Then propose the smallest experiment that would quantify the observer effect.

**Your answer**

```text
Mailbox wait:

Client latency:

Engine prompt timing:

GPU conditions:

A/B experiment:
```

**How to verify the scheduling order**

```bash
nl -ba harness/controller.py | rg -C 5 \
  "co_tenants =|t_send =|await send"
nl -ba harness/device.py | rg -C 8 "async def gpu_co_tenants"
```

Staff-engineer verdict

The subprocess is scheduled after the frame has been taken and immediately before `t_send`, so it is not part of the recorded mailbox wait in the usual execution order. It can overlap client latency and alter CPU scheduling or driver activity. It is not included in llama.cpp's internal `prompt_ms`, but it may still affect that engine time indirectly through shared resources.

A defensible A/B test runs the same seeded inputs and cadence with co-tenant sampling disabled versus enabled, recording at least 500 requests and comparing stage/tail distributions. A cheaper design is to sample GPU processes at cell boundaries and on detected anomalies, rather than spawning a process per request.



### Question 3 — Does `--max-minutes` mean what its name suggests?

Calculate the estimate for `alone-T20`. Then estimate actual warm-up cost using the earlier A1 p50 of `1.659 s`.

**Your answer**

```text
Preflight estimate:

Approximate actual warm-up:

Behavior at a 72-minute budget:

Contract I would document or implement:
```

**How to verify**

```bash
uv run python - <<'PY'
from pathlib import Path
from harness.config import load
from harness.controller import cell_estimate_s
c = load(Path("configs/stack_a.toml"))
cell = c.cell("alone-T20")
print("estimated seconds:", cell_estimate_s(c, cell))
print("estimated warmup seconds:", c.harness.warmup_frames * cell.cadence_s)
print("warmup at 1.659 s/request:", c.harness.warmup_frames * 1.659)
PY
```

Verdict

The estimator returns `4520 s` (75.3 minutes): `(20 warm-up + 200 measured) × 20 s + 120 s`. An approximate real duration is `200 × 20 + 20 × 1.659 + startup`, around 69-71 minutes depending on startup and final completion.

At a 72-minute budget the controller can skip a cell that would probably fit. This is safe but inconsistent with a precise scheduling tool. More serious is that the budget does not stop model hashing/startup/warm-up and lets a final request finish after `stop_at`.



### Question 4 — Should the deadline start at the scheduled tick or the actual one?

`run_mailbox()` sets each frame's deadline to `t_obs + T`, where `t_obs` is when the tick actually fired. Since records now keep `t_scheduled` too, the deadline could instead be `t_scheduled + T`.

Predict, for a tick that fires 2 ms late at T = 3 s and a frame that takes 2.999 s: is it a miss under each rule? Then argue which rule matches the deployed system.

**Your answer**

```text
Deadline from t_obs:

Deadline from t_scheduled:

Which rule matches the satellite (a camera that fires on its own clock):

When the difference matters (T, lateness):

What to record so old runs can be re-scored either way:
```

**How to verify**

```bash
nl -ba harness/feeder.py | rg -C 3 "t_obs = |Frame\(|t_scheduled"
nl -ba harness/records.py | rg -C 3 "t_scheduled|def feeder_late_s|def missed"
```

<details><summary>Verdict</summary>

- **From `t_obs`:** deadline at fired + 3 s; the frame finishes 2.999 s after firing, so it makes it.
- **From `t_scheduled`:** deadline at due + 3 s; the frame finishes 2.001 + 2.999 = 3.001 s after the due time, so it misses.
- **The satellite's camera captures at its own scheduled instant.** The world's deadline starts when the observation happened, which the harness models as the scheduled tick; the feeder's lateness is a harness cost, not something the deployed system would have. So `t_scheduled + T` is the deployment deadline, and `t_obs + T` hides the feeder's own lateness from the miss rate.
- **When it matters:** only when the feeder's lateness is a meaningful fraction of the slack. At T = 3 s and ~1 ms lateness it almost never changes a verdict; for a 30 Hz robot (33 ms period) a 2 ms late tick is 6% of the budget.
- **What to record:** both times, which the records now do. The report can then score either rule after the fact, and should say which one it used.

</details>



### Round 2 exit criterion

Draw the critical path from `t_obs` to `t_last_token` and mark every allocation, syscall, subprocess, event-loop yield, and GPU synchronization. You should be able to defend which costs are intentionally included.

---



## Round 3 — Engine adapters and hardware telemetry



### Goal

Judge whether adapters expose the right abstraction and whether telemetry is sufficiently accurate without becoming a major workload itself.

> **Since this round was written (`c4a4e5f`):**
> - `llama_server.py` gained `server_command()` and `docker_command()`: the same binary, CUDA libraries and model, mounted into a container, with host network and IPC (needed for MPS clients). Round 6 reviews it.
> - **What `prompt_ms` contains:** the visualizer's first profiles show that llama-server's `prompt_ms` includes the **vision encoder**, which is about 89% of the GPU time of a frame. One kernel, `flash_attn_tile` (27 calls, one per encoder layer), is 69% of the frame and runs on the FMA pipe with the tensor cores idle. "Prefill" in this repo means vision encoder + language-model prefill.
> - The client-minus-engine gap the dual timing exposes is about 150–180 ms per frame for Stack A and 0.3 ms for Stack B in-process (issue #7). The dual timing did its job.
> - Question 4 below (new) uses the multi-tenant results to test what the unpinned GPU clock does to a baseline.

### 3.1 The Stack A engine protocol

`engines/base.py:13-48` defines:

- `Reply`: token timestamps, text, counts, and engine timing.
- `Engine.prepare()`: move deterministic request construction off the timed path.
- `Engine.session()`: reuse a connection/client across requests.
- `Engine.pids()`: exclude owned processes from co-tenant metadata.

This is a clean seam for Stack A because both dummy and llama-server can implement it.

It is not yet a generic inference protocol. `prepare(image_png, prompt, max_tokens)` and token-specific `Reply` encode VLM assumptions. The LeRobot adapter returns `ChunkReply` with `n_actions` and named stages (`lerobot_policy.py:24-29`) and is invoked directly by `run_b1.py`.

### 3.2 llama.cpp adapter



#### Process lifecycle

`engines/llama_server.py:47-83` starts one server slot, disables continuous batching, redirects logs, and polls `/health`. One slot isolates single-request service time from scheduler batching.

Failure visibility is good: an early process exit or health timeout raises and points to the server log.

#### Request construction

`encode_request()` at lines 111-137:

- embeds the image as a base64 data URL,
- sets temperature 0,
- enables streaming and usage data,
- sets `ignore_eos=True`,
- sets `cache_prompt=False`.

Interpretation:

- `cache_prompt=False` ensures each request pays full prefill, improving comparability across repeated tiles.
- It may not represent an optimized deployment that legitimately caches a shared text prefix.
- With `max_tokens=1`, `ignore_eos=True` has little practical effect.
- With the four-token diagnostic request in `probe.py`, it forces continued output after EOS. That helps inspect continuation behavior but makes the probe request unlike the measured cell.



#### Streaming timestamps

`send()` at lines 152-191 stamps first and last token when SSE content chunks arrive at the Python client. This is client-visible latency, which includes server-to-client transport and parsing. Engine timings are separately extracted from llama.cpp.

That dual timing is strong: the gap between client and engine can expose transport, server queueing, or Python overhead.

### 3.3 LeRobot adapter

`lerobot_policy.py:69-98` times:

1. observation conversion + preprocessing,
2. policy inference,
3. postprocessing and CPU transfer.

`torch.cuda.synchronize()` at lines 63-67 makes each boundary a real completion boundary. Without synchronization, Python would time asynchronous kernel launch rather than GPU completion.

Trade-off: synchronization changes execution. It prevents overlap across stages and may overstate latency compared with a pipeline that intentionally overlaps preprocessing, inference, and transfer. It is correct for answering “how long does each serial stage take?” but not automatically for “how fast can an optimized pipeline run?”

`run_b1.py:50-60` uses `ThreadPoolExecutor(max_workers=1)`. The event loop can continue emitting requests while one worker serializes policy calls. Under `run_fixed_rate`, executor submissions accumulate instead of being overwritten.

### 3.4 Dummy engine

`dummy.py:23-38` defines a reproducible distribution. `DummyEngine` consumes draws in service order, including warm-up (`dummy.py:41-83`).

This enables independent replay, but only if the checker reproduces:

- the same random generator,
- the same number of warm-up draws,
- service-order consumption rather than frame-order consumption.

`gate1.py:98-101` deliberately discards the warm-up prefix before simulation.

### 3.5 Telemetry on `deadline-harness`

`device.py` collects three overlapping views:

- `PowerSampler` at lines 57-116: 100 Hz sysfs reads for GPU rail, board power, and GPU clock.
- `GpuUtilization` at lines 158-198: long-lived `nvidia-smi` stream at 100 ms.
- `Tegrastats` at lines 136-154: raw device/temperature log.

Good choices:

- all request-correlated samples use the same host monotonic clock;
- missing sensors are represented as NaN/None, so dummy runs work off-device;
- raw `tegrastats` remains available if parsing assumptions change;
- one long-lived `nvidia-smi` process is preferable to one process per utilization sample.

Trade-offs:

- `PowerSampler._read_once()` (`device.py:82-93`) opens and reads several sysfs files every 10 ms;
- the sample timestamp is taken after those reads, though voltage, current, power, and clock were not sampled atomically;
- 100 Hz polling and CSV retention can consume CPU and memory during long cells;
- mean power × duration is later used as energy, rather than time-aware trapezoidal integration.



### 3.6 What differs on `main`

`main:harness/device.py` has:

- `DeviceSampler` at lines 58-111,
- GPU and memory clock sampling,
- hard failure if power sensors are absent,
- `read_device_samples()` compatibility for old `power.csv`,
- no `GpuUtilization` stream.

`deadline-harness` improves portability and GPU-load observability but loses memory clock. Neither branch independently contains the full telemetry needed by the stated Q2 attribution.

### Question 1 — What exactly does `torch.cuda.synchronize()` buy and cost?

Suppose it is removed from `LeRobotPolicy.act()`. Predict what happens to:

- total `t_ready - t_send`,
- reported preprocess time,
- reported inference time,
- overlap potential,
- causal attribution.

**Your answer**

```text
Total latency:

Stage timings:

Potential throughput:

Validity trade-off:
```

**How to verify the locations**

```bash
nl -ba harness/engines/lerobot_policy.py | rg -C 8 \
  "_wait_for_gpu|t0 =|t1 =|t2 =|t3 ="
```

For an empirical answer, add an explicit experimental flag rather than deleting synchronization, run both variants with identical observations, and compare total latency plus stage sums.

Staff-engineer verdict

Without synchronization, stage timers mostly measure CPU launch and enqueue time. GPU work can spill across the apparent boundaries, so `inference_ms` may look artificially small and later stages absorb prior work. The final `.cpu()` is itself a synchronization point, so total latency may remain broadly correct while attribution becomes false.

Synchronization makes stage attribution credible but forbids overlap. Keep a synchronized diagnostic mode and separately measure an optimized end-to-end mode if overlap is a research variable.



### Question 2 — Are `cache_prompt=False` and pre-encoding “fair”?

Answer for two distinct claims:

1. comparing raw model/engine prefill determinism,
2. predicting end-to-end deployed satellite behavior.

**Your answer**

```text
For engine comparison:

For deployment prediction:

Missing costs or optimizations:

Result label I would use:
```

**How to verify**

```bash
nl -ba harness/controller.py | rg -C 6 "bodies ="
nl -ba harness/engines/llama_server.py | rg -C 12 \
  "def encode_request|ignore_eos|cache_prompt"
```

Staff-engineer verdict

For controlled engine prefill comparison, pre-encoding and disabling cache reuse are defensible: each request presents the same shape and pays full prefill.

For deployed end-to-end latency, the measurement excludes image acquisition/decoding/serialization and deliberately rejects one optimization. It should be reported as “arrival-to-client completion with request bodies prepared,” with a separate pipeline measurement if end-to-end system latency is claimed.



### Question 3 — Is 100 Hz telemetry precise enough and cheap enough?

Estimate:

- samples per 67-minute `T=20` cell,
- approximate number of sysfs reads,
- samples per 1.6-second inference,
- what transient behavior can be missed.

Then state what evidence you need before changing it.

**Your answer**

```text
Samples/cell:

Reads/cell:

Samples/request:

Possible blind spots:

Benchmark needed:
```

**How to verify the configured rate and code path**

```bash
rg -n "sampler_interval_s|_read_once|read_text" \
  configs/stack_a.toml harness/device.py
```

Staff-engineer verdict

At 100 Hz, a 67-minute measured period produces roughly 402,000 samples. With four sysfs reads per sample, that is about 1.6 million opens/reads/closes. A 1.6-second inference receives around 160 samples.

That temporal resolution is likely adequate for request-scale energy and clock state, but it can miss short frequency or power transients. Do not optimize based on intuition alone: measure emitter jitter and latency with telemetry off, at 10 Hz, and at 100 Hz. The Gate 1 sampler gaps are useful initial evidence but do not quantify inference perturbation.



### Question 4 — What did the GPU clock do to the alone baseline?

Predict, for the alone cell and for a cell with saturating neighbours: the GPU clock when a frame is sent, and during the frame. Then predict how pinning the clock would change the alone time and the measured slowdown from neighbours.

**Your answer**

```text
Alone, clock at send:

Alone, clock during the frame:

With saturating neighbours:

Effect of pinning on the alone time:

Effect on the measured slowdown:
```

**How to verify**

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np
from harness import results
r = results.load(Path("results/2026-11-19-space-multitenancy-initial"))
for c in r.cells:
    recs = [x for x in r.records(c["cell"]) if x.latency_s is not None]
    p = r.power(c["cell"])
    t = np.array([s.t for s in p]); mhz = np.array([s.gpu_mhz or np.nan for s in p])
    at_send = [mhz[np.searchsorted(t, x.t_send)] for x in recs]
    during = [np.nanmean(mhz[(t >= x.t_send) & (t <= x.t_last_token)]) for x in recs]
    print(c["cell"][:30], "send", np.nanmedian(at_send), "during", round(np.nanmedian(during)))
PY
```

<details><summary>Verdict</summary>

Measured on 2026-11-19:

| cell | clock at send (median) | clock during the frame (median) |
|---|---|---|
| alone | 315 MHz | 1,261 MHz |
| every cell with neighbours | 1,386 MHz | 1,386 MHz |

With one frame every 6 s, the governor drops the GPU to its idle clock between frames, so every alone frame starts at 315 MHz and ramps up while it runs. Neighbours keep the GPU busy, so their cells run at full clock throughout.

So the alone baseline is slowed by the governor and the neighbour cells are not: the measured slowdown from neighbours is **understated**. This is the same effect as TODO.md's idle-gap item (SmolVLA 289 ms at 1 Hz vs 188 ms back to back). Pinning the clock (`sudo jetson_clocks`) is a validity requirement for comparing alone with shared cells, not a nicety. The clock's maximum here is 1,386 MHz, not the 1,575 MHz device maximum; record which one a run used.

</details>

### Round 3 exit criterion

For each engine and telemetry source, state:

- what boundary it measures,
- which costs it intentionally excludes,
- which synchronization assumptions make the number valid,
- how the instrument could perturb the workload.

---



## Round 4 — Reporting mathematics and validation



### Goal

Check that every named metric matches its implementation and that validation tests the right hypothesis.

> **Since this round was written (`c4a4e5f`):**
> - Results are stored **raw only**, in `results/<date>-<question>/` (tidy CSVs via `harness/collect.py`, read back by `harness/results.py`). Runs no longer write `summary.json`, per-cell PNGs, or derived fields in `records.jsonl`; `report`, `compare` and `plot_tenancy` print their tables. Section 4.1 and the exit criterion are updated for this, and Question 4 below (new) tests it.
> - `frames_delivered` is gone; the manifest has `frames_served`, `frames_overwritten`, `frames_error` next to `frames_emitted` (the conservation law in 4.6).
> - All figures now use `harness/paper_style.py` (Round 6).
> - The probe's interleaving (4.7, Question 3) has a fix on branch `probe-separate-passes`, not merged yet.

### 4.1 Record semantics

`RequestRecord` at `records.py:15-125` is the stable raw fact table.

Important distinctions:

- `t_arrival`: when the frame actually entered the mailbox, not its ideal scheduled time.
- `t_send`: when the model client began handling the claimed frame.
- `t_last_token`: client receipt time; absent for overwrite/error.
- `deadline`: currently `t_obs + cadence`, stored per frame.
- `waiting_s`: mailbox waiting only for frames that were served.
- `missed`: true for a late answer or no answer.

`write_records()` writes the raw fields only; `latency_s`, `missed` and `feeder_late_s` are properties computed on read (`records.py`). Earlier runs also wrote derived `latency_s` and `missed` into each line; `read_records()` drops them on load, so a stale derived field can never become authoritative.

### 4.2 Percentiles and outcome population

`report.summarize()` at `report.py:92-154` computes latency percentiles only over `outcome=="ok"`, while miss counts include overwritten and errors.

That is sensible if clearly presented:

- p50/p99 describe completed inference latency,
- miss rate describes all frames.

The CDF at `report.py:157-176` divides completed ranks by total frames, so it stops below 1 when frames never finish. This visualizes completion probability and latency together.

### 4.3 “Throughput” in paced runs

`report.py:102-111` calculates:

```text
number answered / (last answer - first send)
```

For a run paced every 30 seconds, the result is approximately `0.033 requests/s`, even though the engine can process roughly one request every 1.66 seconds. The code comment acknowledges that the value is meaningful as throughput only in closed-loop mode.

This is not an arithmetic bug. It is a naming hazard: in latency mode, it measures achieved completion rate under the offered cadence.

The raw A1 summary demonstrates it:

- p50 latency: `1659.1 ms`,
- `throughput_rps`: `0.0337`,
- period: 30 seconds.



### 4.4 Energy

`energy_joules()` at `report.py:54-64` selects samples between `t_send` and `t_end`, computes mean rail power, then multiplies by duration.

Consequences:

- mailbox waiting energy is excluded;
- overwritten frames receive no per-frame energy;
- board/GPU baseline power is included;
- co-runner power on the same rail is included;
- irregular sample spacing is ignored;
- the result is gross rail energy during request service, not incremental model energy.

This may be exactly the deployment quantity of interest, but it must not be described as isolated model energy without a baseline-subtraction method.

### 4.5 Confidence intervals

`wilson_interval()` at `report.py:27-36` computes a 95% Wilson score interval. This behaves sensibly at 0 misses and 100% misses.

Sample count still matters. Twenty requests with zero misses gives an upper 95% bound around 16%, as shown in the existing pi0.5 and SmolVLA runs. A “0% observed miss rate” is not evidence of a reliable 0% system.

### 4.6 Gate 1: model-based validation

`simulate_mailbox()` at `gate1.py:42-73` is an independent discrete-event model. The checker:

- validates emitter spacing,
- predicts overwrite count,
- predicts deadline miss rate,
- checks sampler coverage.

Raw evidence:

- Mac: spacing failed (`21.7 ms` std, `188 ms` worst), but mailbox outcomes matched.
- Thor: spacing passed (`0.49 ms` std), predicted/actual overwritten both 35, predicted/actual miss rate both 84%.

The checker correctly rejects the original naive criterion “fraction of sleeps longer than T.” In an overloaded depth-1 system, served frames can wait and miss even when their own service time is below T.

Potential limitation: the replay shares the same conceptual mailbox policy as the implementation. It is independent code, but not an external oracle. Boundary tests and conservation invariants remain useful:

```text
emitted = ok + overwritten + error
every req_id appears exactly once
no served frame waits more than one cadence, absent emitter faults
```



### 4.7 Gate 2: inspect the raw probe, not just the report

`probe.py:60-78` sends two requests per tile:

1. a timed one-token request,
2. an untimed four-token diagnostic request.

Thus the logged “50 requests back to back” are 50 measured requests interleaved with 50 extra GPU requests. This keeps the GPU busy and changes the sequence seen by co-tenants and the clock governor.

The raw JSONL shows:

- 50/50 one-token outputs are configured labels,
- normal requests cluster around 1.5-1.6 seconds,
- slow requests reach 2.3-3.14 seconds,
- slow-request growth is mainly in `prompt_ms`, not client-only overhead,
- 48/50 agree with a heuristic derived from SCL fractions.

That last number is not model “accuracy” in a general sense. It is agreement with the rule:

```text
drop if cloud_fraction > 0.5 OR water_fraction > 0.5
```

SCL provides a useful proxy, not human-validated downstream utility.

### 4.8 Cross-cell comparison

`compare.py:20-39` selects headline columns and writes a Markdown table. `compare.py:42-64` overlays miss-rate curves.

This is useful, but comparison validity still depends on matched:

- config hash and prompt,
- model/runtime commit,
- tile set/order,
- power mode,
- co-tenant state,
- telemetry behavior.

The table does not itself enforce those controls. A professional comparison tool should warn when provenance differs on axes that are supposed to be held constant.

### Question 1 — Where did the Gate 2 tail come from?

Use the raw JSONL to identify slow requests and compare:

- client latency,
- engine prompt time,
- outside-engine remainder.

**Your prediction**

```text
Slow threshold:

Dominant stage:

Evidence:

Alternative explanation:
```

**How to verify**

```bash
uv run python - <<'PY'
import csv
rows = list(csv.DictReader(open("results/2026-11-04-gate2-real-model/requests.csv", newline="")))
for r in rows:
    r["latency_ms"], r["prompt_ms"] = float(r["latency_ms"]), float(r["engine_prompt_ms"])
for r in rows:
    outside = r["latency_ms"] - r["prompt_ms"]
    if r["latency_ms"] > 2000:
        print(r["tile_id"], r["latency_ms"], r["prompt_ms"], round(outside, 1))
normal = [r for r in rows if r["latency_ms"] <= 2000]
slow = [r for r in rows if r["latency_ms"] > 2000]
for name, group in [("normal", normal), ("slow", slow)]:
    print(name, "n=", len(group),
          "lat=", round(sum(r["latency_ms"] for r in group)/len(group), 1),
          "prompt=", round(sum(r["prompt_ms"] for r in group)/len(group), 1),
          "outside=", round(sum(r["latency_ms"]-r["prompt_ms"] for r in group)/len(group), 1))
PY
```

Verdict

Nine requests exceed two seconds. Most extra latency appears inside llama.cpp's `prompt_ms`, while outside-engine time stays comparatively stable. That supports GPU-side interference or clock behavior rather than Python/HTTP overhead as the primary cause.

The report correlates the spikes with a labmate process periodically reaching high utilization, but the committed probe JSONL does not contain the utilization trace itself. Treat that causal attribution as supported by contemporaneous observation, not fully reproducible from this file alone.



### Question 2 — What does `energy_per_request_j` actually mean?

Give a precise name for the quantity. Then say how it behaves when:

- a co-runner consumes GPU power during the request,
- a frame waits 0.8 seconds before being served,
- a frame is overwritten,
- power samples are irregularly spaced.

**Your answer**

```text
Precise metric name:

Co-runner:

Waiting:

Overwritten:

Irregular sampling:

Metric needed for the thesis:
```

**How to verify**

```bash
nl -ba harness/report.py | rg -C 12 \
  "def energy_joules|def mean_energy_per_request"
```

Staff-engineer verdict

A precise name is “gross rail energy during served-request client interval.” Co-runner energy is included, mailbox waiting is excluded, overwritten frames are omitted, and mean-power multiplication assumes the selected samples adequately represent the interval despite irregular spacing.

For deployment capacity, gross board energy may be appropriate. For model attribution, add idle/co-runner baselines or controlled subtraction. For per-arriving-frame energy, define how idle, waiting, and overwritten frames share system energy.



### Question 3 — Is Gate 2 a clean test of the stated criterion?

Assess the effect of:

- interleaved four-token requests,
- lack of per-request co-tenant/utilization records in `probe.jsonl`,
- the SCL heuristic,
- the “factor of two” threshold.

**Your answer**

```text
What Gate 2 establishes:

What it does not establish:

Confounders:

Smallest redesign:
```

**How to verify**

```bash
nl -ba harness/probe.py | rg -C 12 \
  "bodies_1|bodies_4|one =|four =|expected_labels"
```

Staff-engineer verdict

It establishes that the chosen prompt reliably emits one configured token and has high agreement with the chosen SCL proxy on these 50 tiles. It also reveals a real latency instability.

It does not cleanly establish the latency distribution of 50 one-token requests because every timed request is paired with an extra four-token request. It also does not preserve enough telemetry in the probe artifact to reproduce the co-tenant diagnosis.

A cleaner design separates functional probing from latency probing: first inspect 1/4-token behavior on a small set, then run 50 one-token requests only under full telemetry and save a manifest.



### Question 4 — Can every number in a results README be regenerated, and from what?

Pick three numbers in `results/2026-11-19-space-multitenancy-initial/README.md`: a p99, a "neighbours busy" share, and a miss rate at 3 s.

**Your answer**

```text
Number 1, rows and columns it comes from:

Number 2:

Number 3:

What is NOT in the results folder, and where it is:

One thing collect.py would refuse:
```

**How to verify**

```bash
uv run python -m harness.plot_tenancy results/2026-11-19-space-multitenancy-initial   # prints the table
rg -n "raw_dir|refuse|do not know|stay in the raw" harness/collect.py harness/results.py
column -s, -t results/2026-11-19-space-multitenancy-initial/cells.csv | less -S
```

<details><summary>Verdict</summary>

- **p99:** tenant 0's rows of that cell in `requests.csv` with an answer, `t_last_token − t_arrival`, 99th percentile (linear interpolation over 12 values, so near the slowest frame).
- **Neighbours busy:** the neighbours' rows (`tenant` ≥ 1) of that cell, overlap of `[t_send, t_last_token]` with tenant 0's first-to-last frame, averaged over neighbours.
- **Miss rate at 3 s:** all of tenant 0's rows, overwritten and failed included, `latency is None or latency > 3`.
- **Not in the folder:** llama-server logs, `tegrastats.log`, and anything else the collector doesn't know. They stay in `raw_dir` on the Thor, which is not in git, so a results folder alone cannot answer questions about them.
- **Refused:** a background log or sample CSV with a column the collector doesn't know, two cell directories with the same name, and a cell whose engine/model/quant/power mode changes between records.

</details>

### Round 4 exit criterion

Choose any number in a printed summary (`harness.report`, `compare` or `plot_tenancy`) and be able to answer:

1. Which raw rows enter its population?
2. Which interval and clock define it?
3. What is excluded?
4. What confounder could make the number move?
5. What independent evidence validates it?

---



## Round 5 — Forward design



### Goal

Use the immediate roadmap to test whether current abstractions are good enough. Good architecture is not maximum generality; it is the smallest design that keeps the next experiments honest.

> **Since this round was written (`c4a4e5f`):**
> - Neighbours exist now, but not as DESIGN.md's light/heavy co-runners: they are the same VLM in their own containers (`tenants`, `load`, `sharing`, `placement` cell fields; `co_runner` is still `none` only). Several items in 5.2 (own process, ready handshake, owned identity, cleanup) are implemented in `harness/tenancy.py`; Round 6, Question 1 tests the cleanup.
> - We have sudo, so the power-mode transaction in 5.3 is buildable now (issue #10).
> - Most acceptance tests in Question 3 have issues: alone-cell admission #11, co-runners #12, power #10, Stack B on the controller #6, reporter port #4. The integration decision record is #3.

### 5.1 Step 3: alone cells

The next planned cells are `T={20,5,2,1.5}` with no co-runner.

Current strengths:

- cell matrix and provenance are explicit,
- real tiles and prompt are fixed,
- mailbox semantics are validated,
- telemetry and co-tenant snapshots exist,
- reports can be regenerated.

Current risks:

- `gpu_co_tenants()` records contamination but does not reject it;
- the old sweep’s `wait_for_idle_gpu()` treats “holds a context” as “busy” and can wait indefinitely;
- Gate 2’s baseline failed under a known co-tenant;
- the 20-second cell is long enough for environmental conditions to change.

A robust admission policy should distinguish:

- GPU process presence,
- sustained utilization,
- memory reservation,
- known allowed processes,
- contamination during the cell.

The experiment should define whether a contaminated run aborts immediately, completes but is marked invalid, or becomes an intentional NM cell.

### 5.2 Step 4: co-runners

`config.py:13-17` currently allows only `co_runner="none"`. That is correct until lifecycle management exists.

A co-runner implementation needs:

- its own OS process and CUDA context,
- ready handshake before frame emission,
- fixed, config-defined workload and rate,
- per-iteration timestamps and latency log,
- PID ownership so it is not labeled as an accidental co-tenant,
- startup/teardown ordering,
- cleanup after exception/Ctrl-C,
- manifest identity/version,
- verification that “light” and “heavy” actually create distinct load regimes.

Do not bury co-runner behavior inside the model process. The thesis question concerns process-level contention.

### 5.3 Power modes

The design requires:

1. capture original `nvpmodel`,
2. switch at a cell boundary,
3. verify the requested mode took effect,
4. run the cell,
5. restore original mode on normal exit, exception, and Ctrl-C.

`controller.run_cell()` already has a `try/finally`, but it only records the current mode; it does not switch or restore (`controller.py:140-188`). This is the natural lifecycle boundary to extend.

Failure handling is the hard part. If the Python process is killed with `SIGKILL` or the machine reboots, `finally` cannot restore anything. A run manifest and startup preflight must detect unexpected inherited state.

### 5.4 Stack B unification

Forcing LeRobot into the token-shaped Stack A `Engine` protocol would be the wrong abstraction.

A minimal general decomposition is:

```text
ExperimentController
├── Workload
│   ├── prepare_inputs()
│   ├── frame metadata
│   └── quality oracle (optional)
├── SchedulingPolicy
│   ├── mailbox
│   ├── queued fixed-rate
│   └── closed-loop throughput
├── EngineAdapter
│   ├── warmup()
│   ├── infer(prepared_input)
│   ├── owned_pids()
│   └── generic completion + named stages
├── DeviceContext
│   ├── telemetry
│   ├── co-runner
│   └── power mode
└── Recorder
```

The raw completion record should preserve general timestamps and named stage timings. Token counts and action counts can be optional workload-specific fields.

Avoid an abstract framework until both Stack A and Stack B can implement the interface concretely. Extract the shared boundary from those two implementations, not from imagination.

### 5.5 Reporting merge

The combined reporter should preserve:

- `deadline-harness`: terminal outcomes, overwritten-aware CDFs, arbitrary deadline curves, timeline, manifest compatibility;
- `main`: `DeviceTrace`, stage decomposition, slow-tail deltas, request-window clocks, memory clock, period sweep analysis.

It should also add:

- explicit metric names for paced completion rate vs saturation throughput,
- provenance mismatch warnings in `compare.py`,
- time-aware energy integration,
- schema versioning,
- separate gross-system and incremental energy metrics if baselines exist.



### Question 1 — What should the controller own?

Assign each responsibility to one of:

- controller,
- scheduling policy,
- workload adapter,
- engine adapter,
- device context,
- reporter.

Responsibilities:

- image base64 encoding,
- frame overwrite,
- warm-up count,
- co-runner process,
- `nvpmodel` restoration,
- action-chunk stage names,
- deadline application,
- output vocabulary/quality check,
- run-directory creation.

**Your answer**

```text
Controller:

Scheduling:

Workload:

Engine:

Device:

Reporter:

One boundary I am uncertain about:
```

**How to verify against current coupling**

```bash
rg -n "prompt|max_tokens|tiles|run_mailbox|power_mode|co_runner|report\\(" \
  harness/controller.py harness/config.py harness/run_a1.py harness/run_b1.py
```

Staff-engineer verdict

- Controller: lifecycle, run directory, warm-up orchestration, failure cleanup.
- Scheduling: overwrite/queue/closed-loop behavior and arrival/deadline creation.
- Workload: image preparation inputs, output vocabulary, quality oracle, robot observation/task.
- Engine: serialization required by the runtime, inference, engine timing, owned PIDs.
- Device: telemetry, co-runner lifecycle, power-mode transaction.
- Reporter: post-hoc deadline application, statistics, energy derivation, comparisons.

The exact placement of serialization depends on whether the research question includes it. Make that an explicit timed-stage decision rather than an accidental implementation detail.



### Question 2 — What should “GPU idle” mean for an alone cell?

Design an admission and invalidation policy that handles the raw reality shown in:

```text
/data/projects/pulak/runs/sweeps-20261104.log
```

where processes held contexts for hours.

**Your answer**

```text
Preflight conditions:

Allowed processes:

During-run checks:

Abort vs mark-invalid rule:

Metadata retained:
```

**How to inspect the raw evidence**

```bash
rg -n "GPU busy|step [0-9]+/|ALL_DONE|FAILED" \
  /data/projects/pulak/runs/sweeps-20261104.log
```

Staff-engineer verdict

Process presence alone is too strict because an idle server may reserve memory indefinitely. Utilization alone is too weak because a process can burst between samples.

A practical policy combines an allowlist, a preflight quiet window (for example, zero unexpected compute processes and low utilization for a defined duration), continuous utilization/process logging, and post-run contamination flags. Unexpected high utilization should mark an “alone” cell invalid; whether to abort immediately depends on the cost of finishing the run for diagnostic value.



### Question 3 — What is the minimum proof before running expensive cells?

Define acceptance tests for:

- alone-cell admission,
- co-runner readiness and cleanup,
- power-mode restore,
- Stack B mailbox conservation,
- merged reporter compatibility with old runs.

**Your answer**

```text
Alone:

Co-runner:

Power:

Stack B:

Reporter:
```

**How to inspect the existing test baseline**

```bash
uv run pytest -q
uv run ruff check
rg -n "^def test_" harness/tests
```

Staff-engineer verdict

At minimum:

- Alone: deterministic fake process/utilization snapshots test admission and contamination marking.
- Co-runner: fake subprocess reaches ready, writes iterations, and is terminated on normal exit and exception.
- Power: mocked mode switch restores the original mode for success, exception, and Ctrl-C-equivalent paths; a failed verification aborts before measurement.
- Stack B: every emitted frame appears exactly once as ok/overwritten/error under a slow fake policy.
- Reporter: golden old v1 and new v2 run fixtures both load; merged metrics preserve known expected values.

Only after these pass should long real cells consume scarce Thor time.



### Round 5 exit criterion

Produce a one-page design decision that states:

- authoritative lifecycle path,
- explicit scheduling policies,
- common record schema,
- preserved features from both branches,
- run admission/contamination policy,
- acceptance tests before Step 3 and Step 4.

---



## Round 6 — The new code: tenancy, MPS, results, figures, visualizer



### Goal

Rounds 1–5 were written against `c4a4e5f`. Everything below landed afterwards (PR #18 and the `visualizer` commits after it) and has not been reviewed. Judge it by the same standard: is each number what its name says, can it be regenerated, does measuring it change it, and does a broken run fail loudly?

### 6.1 What was added

| File | Role | Measured, modeled, or labelled |
|---|---|---|
| `harness/tenancy.py` | starts N−1 neighbour llama-server containers, their load generators, and MPS | lifecycle |
| `harness/mps.py` | private MPS daemon; `timeslice`, `mps`, or `mps-partitioned` (equal static SM partitions) | lifecycle |
| `harness/background.py` | a neighbour's client: `idle` / `steady` / `bursty` / `saturating`, one process per neighbour | workload |
| `harness/engines/llama_server.py` | `server_command()`, and `docker_command()` to run the same binary in a container | lifecycle |
| `harness/config.py` | cell fields `tenants`, `load`, `sharing`, `placement`; `[container]`, `[tenancy]` | configuration |
| `harness/results.py`, `harness/collect.py` | raw cell directories → `results/<date>-<question>/` tidy CSVs, raw only | storage |
| `harness/paper_style.py`, `harness/plot_tenancy.py` | the one figure style; time-slicing vs MPS figures | presentation |
| `visualizer/` | nsys capture → `bundle.json` → local viewer; ncu on top kernels | measurement + model |

### 6.2 The tenancy lifecycle

`controller.run_cell()` nests the contexts so that tenant 0 starts last and stops first:

```text
Tegrastats, GpuUtilization
  PowerSampler, Tenants            ← Tenants.__enter__: MPS → neighbour containers → load generators
    tenant 0 engine                ← host process or its own container
      measure()
```

`Tenants` uses an `ExitStack`, so on exit the generators stop first, then the containers (`docker stop`), then the MPS daemon (`quit`). The reverse order matters: an MPS daemon that quits under live clients, or a container stopped while its generator still sends, produces errors that look like data.

Two assumptions hold the design together:

- every process the cell starts is owned by exactly one context that stops it;
- that context is registered *before* the next step can fail.

### 6.3 MPS sharing

- **Private pipe directory** (`~/.lbi-mps`): labmates' CUDA processes use the default directory, so they never join our daemon. The path must stay short; Unix socket paths are limited to 108 bytes (the first attempt, in the scratchpad, failed silently for this reason).
- **Static partitions** (`-S`, `sm_partition add`): the Thor hands out SMs in chunks of 2 (10 chunks). Each tenant gets `floor(10 / tenants)` chunks: 2 tenants → 10 SMs each, 3 → 6 each with 2 idle, 4 → 4 each with 4 idle, 5 → 4 each.
- In static mode every client must name a partition; a client without one sees "no CUDA-capable device".
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` was rejected for fairness: it caps rather than partitions (caps may overlap), and rounds in 2-SM steps (25% → 4 SMs, not 5).

### 6.4 Background load

`send_times()` draws Poisson arrivals: `steady` at mean period `background_period_s` (3 s), `bursty` as bursts of 5 back to back, bursts at mean `5 × 3 s`. So steady and bursty have the same **offered** mean rate.

A neighbour has one server slot. If a request overruns its next due time, the next request goes out at once. So the **achieved** load depends on how long each neighbour request takes, which depends on how the GPU is shared.

Arrivals are random rather than periodic because the first smoke cell phase-locked: tenant 0 and a neighbour both at T = 3 s overlapped by the same 500 ms in every frame.

### 6.5 Results: raw only

`collect.py` copies raw cell directories into tidy CSVs (`cells.csv`, `requests.csv`, `power.csv`, `gpu_util.csv`, `co_tenants.csv`, `manifests/`) and refuses fields it doesn't know. `results.py` loads them back as `RequestRecord`s. No statistic is stored; tables are printed by `report`, `compare`, `plot_tenancy`. Round 4, Question 4 tests whether that holds. Engine logs and `tegrastats.log` stay in the raw directory (`raw_dir`). Timestamps are host monotonic per cell; they are comparable within a cell, not across cells.

### 6.6 Figures

`paper_style.py` fixes the style (see its docstring). `plot_tenancy.py` computes every plotted number from the folder's raw tables at plot time. With 12 frames per cell, a p99 is close to the slowest frame, and a miss rate moves in steps of 8.3 points.

### 6.7 The visualizer

- **Capture** runs tenant 0 on the host under `nsys launch`, tracing only between `nsys start` and `nsys stop`. In the cells, tenant 0 ran in a container; the container measured the same speed as the host (1.52 s back to back), but the placements differ.
- **Bundle**: frames are split wherever the GPU is idle for more than 300 ms (`FRAME_GAP_NS`) and matched to client requests by order. The stream that runs `im2col` is labelled "vision encoder". Blocks per SM and waves come from an occupancy model, not a counter.
- **ncu** replays one `llama-mtmd-cli` frame, profiles the first two launches of each top kernel name, at ncu's pinned clock (about 1,070 MHz), without `-lineinfo`. On the Thor its DRAM counter reads 0% even for kernels that miss L2.

### Question 1 — Does every process a cell starts get stopped on every failure path?

For each failure, list what is left running afterwards:

- starting neighbour 2's container fails (e.g. port in use),
- creating the second MPS partition fails,
- starting the second load generator fails,
- Ctrl-C during `measure()`,
- the controller is killed with `SIGKILL`.

**Your answer**

```text
Neighbour 2 container fails:

Second MPS partition fails:

Second load generator fails:

Ctrl-C during measure():

SIGKILL:

What a next run should check before it starts:
```

**How to verify**

```bash
nl -ba harness/tenancy.py | rg -C 3 "enter_context|callback|def __enter__|def _stop_generators|except BaseException"
nl -ba harness/mps.py | rg -C 3 "def __enter__|subprocess.run\(daemon|_partition_equally\(\)|def __exit__"
nl -ba harness/engines/llama_server.py | rg -C 3 "docker stop|def __exit__|_wait_until_healthy"
```

<details><summary>Verdict</summary>

- **Neighbour 2 container fails:** clean. `LlamaServer.__enter__` stops its own container if the health wait fails, and `Tenants.__enter__` closes the stack: neighbour 1's container, then MPS.
- **Second MPS partition fails: the daemon is left running.** `Mps.__enter__` starts the daemon, then `_partition_equally()` raises. Because `__enter__` raised, `ExitStack.enter_context` never registered `__exit__`, so nothing sends `quit`. The next partitioned cell finds a daemon on the same pipe directory and either refuses or fails to start a second one. Fix: start the daemon inside a `try` that quits it if partitioning fails.
- **Second load generator fails: the first generator is orphaned.** `_stop_generators` is registered with `self._stack.callback` only *after* the loop that starts the generators, so a failure inside the loop closes the containers but never terminates the generators already running. They then send requests to a dead port and log errors until killed. Fix: register the callback before the loop (it is safe on an empty list).
- **Ctrl-C during `measure()`:** clean. `KeyboardInterrupt` unwinds through every `with`: tenant 0, generators, containers, MPS, samplers.
- **SIGKILL:** nothing is cleaned up. `finally` blocks don't run. Containers keep running (they belong to the Docker daemon, and `--rm` only removes them after they stop), generators keep running (child processes are not killed with their parent), the MPS daemon stays, and the GPU is no longer "alone".
- **What a next run should check:** no running container named `lbi-*`, no MPS daemon on our pipe directory, no `harness.background` process, and no unexpected GPU process. That is the leftover-state half of issue #11's admission check.

</details>

### Question 2 — Is "bursty time-slicing vs bursty MPS" a fair comparison?

The two bursty cells used the same generator settings and seeds. The README reports neighbours 38% busy under time-slicing and 62% busy under MPS partitions.

**Your answer**

```text
Same offered load?

Same achieved load?

Why they differ:

Does it matter for tenant 0's result under partitions? Under time-slicing?

Other differences between the two cells:

What I would change before the full matrix:
```

**How to verify**

```bash
uv run python - <<'PY'
from pathlib import Path
from statistics import median
from harness import results
r = results.load(Path("results/2026-11-19-space-multitenancy-initial"))
for c in r.cells:
    if c["load"] != "bursty":
        continue
    for tenant, reqs in r.neighbour_requests(c["cell"]).items():
        took = [q["t_last_token"] - q["t_send"] for q in reqs]
        print(c["cell"][:30], "neighbour", tenant, "requests", len(reqs), "median s", round(median(took), 2))
PY
```

<details><summary>Verdict</summary>

- **Offered load: the same.** Same Poisson schedule and mean rate, from the same seeds.
- **Achieved load: not the same.** Each neighbour has one slot. On a 6-SM partition a neighbour's request takes a median 4.4 s, against 1.6 s under time-slicing, so the bursts last longer and the next burst often catches up with the last: 62% busy vs 38%.
- **For tenant 0 under partitions, it barely matters:** the partition isolates it (4.39 s with nobody busy, 4.47 s with both busy). Under time-slicing it is the whole effect: tenant 0's time is alone × (1 + busy neighbours).
- So the comparison is fair as a **deployment** question ("the same cameras send the same requests; which sharing serves tenant 0 better?"), not as "equal neighbour activity". Say which one a figure answers.
- **Other differences:** 3 tenants leave 2 of 20 SMs idle under partitions (6 SMs = 30%, not 33%); the GPU clock is not pinned (Round 3, Question 4); 12 frames per cell; neighbours' arrival phase relative to tenant 0 depends on warm-up length, which differs between the cells.
- **Before the full matrix:** pin the clock, use 2 and 5 tenants (exact splits), at least 60 frames per cell, and record achieved neighbour load next to every result.

</details>

### Question 3 — Which parts of a visualizer bundle are measured, which are modeled, and when does the bundle break?

Predict what `split_frames` does for: (a) Stack A at T = 3 s, (b) Stack A at T = 1.5 s with 1.4 s frames, (c) SmolVLA at 30 Hz. Then classify each column of the viewer's kernel table.

**Your answer**

```text
(a) T = 3 s:

(b) T = 1.5 s:

(c) 30 Hz:

What the bundle does when the frame count is wrong:

Measured columns:

Modeled columns:

Labelled by rule:
```

**How to verify**

```bash
nl -ba visualizer/bundle.py | rg -C 4 "FRAME_GAP_NS|def split_frames|def label_streams|def blocks_per_sm|warnings.append"
```

<details><summary>Verdict</summary>

- **(a) T = 3 s:** fine. The GPU is idle for about 1.6 s between frames, far above 300 ms.
- **(b) T = 1.5 s with 1.4 s frames:** the gap is about 0.1 s, below 300 ms, so consecutive frames merge into one "frame". **(c) 30 Hz** merges everything into one frame.
- **When the count is wrong:** the bundle is still built; it only adds a warning ("N frames in the trace, M sent by the client"), and it matches frames to requests by order, so every later frame is attributed to the wrong request. Splitting by the client's timestamps mapped onto the nsys clock (roadmap item in #17) fixes both.
- **Measured** (CUPTI): kernel start/end, grid, block, registers, shared memory, stream, memcpys, and in the ncu columns the counters themselves.
- **Modeled:** blocks per SM and waves, and from them any idle-SM tail.
- **Labelled by rule:** kernel family (name patterns) and stream role (`im2col` → vision encoder). A model without an `im2col` kernel gets every stream labelled "language model".

</details>

### Question 4 — What can the ncu summary support, and what can't it?

The #17 roadmap issue states: `flash_attn_tile` is at 85% compute throughput on the FMA pipe, tensor pipe 0%, and warps mostly wait for the math pipe, not memory.

**Your answer**

```text
Claims the ncu data supports:

Claims it does not support yet:

Why flash_attn_tile is more trustworthy than the other five kernels:

What would make the other five trustworthy:
```

**How to verify**

```bash
B=/data/projects/rohan/profiles/20261119-162806-alone
python3 -c "import json; b=json.load(open('$B/bundle.json')); [print(k, v['launches'], v['grid_blocks'], round(v['time_us']), round(v['clock_mhz'])) for k, v in b['ncu'].items()]"
rg -n "Profiling|No source files" $B/ncu/*.log
```

<details><summary>Verdict</summary>

- **Supported:** for `flash_attn_tile`, the pipe mix (FMA busy, tensor pipe idle), the stall mix (math-pipe throttle ≫ memory waits), occupancy limited by registers, and an L2 hit rate of 97%. All 27 calls have the same shape (grid 4,096 blocks), so two profiled launches represent them.
- **Not supported yet:**
  - absolute times: ncu pins the clock at about 1,070 MHz;
  - DRAM traffic: the counter reads 0% on the Thor;
  - source-line stalls: no `-lineinfo` build;
  - anything about the other five kernels, whose first launches have shapes unlike the typical ones (`k_bin_bcast`: 12 µs under ncu vs 211 µs average in the frame);
  - whether llama-server runs the same kernels as `llama-mtmd-cli`. It uses the same libraries, which is likely but unchecked.
- **To trust the other five:** profile the launches whose shape carries the most time in the bundle (choose `--launch-skip` per name), or profile all launches of one name and weight by shape. Add `--soc-metrics` from nsys for DRAM.

</details>

### Round 6 exit criterion

For each new module, write one line each:

- what it owns and what stops it on failure,
- which numbers it produces, and whether they are measured, modeled, or labelled,
- what would make a run that used it invalid, and whether the run would say so.

---



## Closing synthesis



### Your final assessment

Fill this after all six rounds.

```text
The strongest design decision in this codebase is:

The largest threat to measurement validity is:

The largest maintainability risk is:

One apparent code smell that is actually justified is:

One clean-looking abstraction that is misleading is:

The first change I would make is:

The first experiment I would run after that is:

Evidence that would change my mind:
```



### Discernment checklist

When reviewing future systems code, ask:

- What scientific or operational claim does this code support?
- What is the exact population and interval behind each metric?
- Is the world clock independent of the model?
- Can overload be distinguished from late completion?
- Is queue policy explicit?
- Are timestamp boundaries documented?
- Which operations occur inside those boundaries?
- Can telemetry perturb the result?
- Are unsupported configurations rejected?
- Are raw data and provenance immutable?
- Can analysis be regenerated?
- Does a validation oracle share the same bug?
- Do confidence intervals support the headline claim?
- Does the next planned experiment fit the current abstractions?

---



## Command reference

All commands are read-only except `pytest`, which may update `.pytest_cache`, and report commands, which can regenerate artifacts in a run directory.

### Branch history and divergence

```bash
git merge-base main deadline-harness
git log --oneline --decorate --graph --all --max-count=15
git diff --stat 7f6c1f6..main
git diff --stat 7f6c1f6..deadline-harness
git diff main..deadline-harness -- harness/
```



### Read a file from the other branch without switching

```bash
git show main:harness/report.py
git show main:harness/device.py
git show main:harness/sweep.py
```



### Raw committed gate data

```bash
column -s, -t results/2026-11-04-gate2-real-model/requests.csv | less -S
cat results/2026-11-04-gate1-dummy/checks-thor.json
cat results/2026-11-04-gate1-dummy/checks-mac.json
```



### Existing Thor runs

```bash
cat /data/projects/pulak/runs/a1-latency-20261012-210527/summary.json
cat /data/projects/pulak/runs/b1-smolvla-latency-20261012-203731/summary.json
cat /data/projects/pulak/runs/b1-pi05-latency-20261012-205216/summary.json
rg -n "GPU busy|step|ALL_DONE|FAILED" \
  /data/projects/pulak/runs/sweeps-20261104.log
```



### Local verification

```bash
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

