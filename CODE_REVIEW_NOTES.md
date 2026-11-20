# Code Review Learning and Merge Notes

## Purpose

This is the cumulative response journal for `CODE_REVIEW_WORKBOOK.md`.

It has three jobs:

1. Preserve what I learned while reading the code.
2. Separate facts established by code/data from design judgments.
3. Build an evidence-backed list of changes to request before `deadline-harness` is merged into `main`.

The intended final system is broader than the present Stack A harness:

- a space microdatacenter with several containerized tenants sharing one GPU;
- one or more VLM/VLA inference services;
- workload-specific deadlines and stale-input policies;
- controlled co-runners and power modes;
- sufficient telemetry to attribute deadline misses to workload, runtime, contention, or deployment state.

The current repository is therefore both an experiment and a prototype of that future measurement platform.

### Review status

- Round 1: completed and clarified below.
- Rounds 2-6: pending. Rounds 2-5 were refreshed on 2026-11-19 for the code merged in #18;
  Round 6 (new) reviews the code written after Round 1: tenancy, MPS, background load,
  results/, paper_style, visualizer.
- Pre-merge recommendations: provisional until all rounds are complete.

### Evidence rule

Each conclusion should eventually have one of these labels:

- **Code fact:** directly established by the current implementation.
- **Data fact:** directly established by a raw run artifact.
- **Design judgment:** a recommendation based on the project goal.
- **Open hypothesis:** requires an experiment.

## Round 1 — Architecture and branch divergence

### My Round 1 conclusion

`deadline-harness` should be the semantic integration base for Stack A because it contains the mature experiment definition:

- explicit cell configuration;
- a controller and manifest;
- depth-1 mailbox semantics;
- overwritten-frame accounting;
- dummy-model validation;
- real-tile prompt validation.

Before merging it into `main`, we should:

1. port the analysis capabilities that exist only on `main`;
2. avoid losing the existing LeRobot/Stack B support;
3. replace the two competing orchestration paths with one extensible experiment lifecycle;
4. make scheduling policy explicit rather than implicitly tied to a stack.

This agrees with the workbook's staff-engineer verdict, but the important reason is not simply that `deadline-harness` is “more mature.” Its semantics match the immediate experiment: a sensor continues producing frames while the model is busy, and stale unclaimed frames are replaced.

### Correction to my first exit answer

My initial answer was:

> The feeder modes represent different sensor systems.

That is only partly right.

The deeper distinction is **overload policy**, not sensor identity:

- `run_fixed_rate()` creates every request and lets requests queue.
- `run_mailbox()` allows one waiting frame and replaces it with the newest frame.
- `run_closed_loop()` produces a new input only after the previous request completes.

The same camera could use any of these policies. Conversely, different sensors could share one policy. The policy determines what a deadline miss means and which requests even reach the model.

### Correction to my second exit answer

My initial answer was:

> The branches differ in reporting/logging.

They do, but that understates the divergence.

`deadline-harness` changes:

- what system is simulated under overload;
- how experiments are configured;
- which process owns lifecycle;
- what provenance is written;
- how unsupported cells fail;
- how frames receive terminal outcomes;
- how the harness is validated.

`main` adds:

- stage-level latency attribution;
- slow-tail attribution;
- request-window GPU clock analysis;
- period sweeps.

The branches therefore diverge in both **experimental semantics** and **analysis capability**.

### Correction to my third exit answer

My initial ownership answer was roughly:

> deadline, deadline, main

A clearer ownership model is:

- **Controller/lifecycle:** evolve `deadline-harness`'s controller.
- **Scheduling semantics:** explicit policy object/module; preserve mailbox, fixed-rate queue, and closed-loop as named policies.
- **Workload semantics:** stack-specific adapter for tiles/prompts or robot observations/tasks.
- **Runtime semantics:** engine adapter for llama.cpp, LeRobot, future vLLM, etc.
- **Raw records:** one shared schema with optional workload-specific fields.
- **Reporting:** combine deadline-aware reporting from `deadline-harness` with stage/tail/clock attribution from `main`.

No branch should “own” every layer.

### Q1 — Which branch should be the integration base?

**Claim**

Use `deadline-harness` as the semantic base, then selectively port capabilities from `main`.

**Evidence**

- The common ancestor is `7f6c1f6`.
- `deadline-harness` adds `controller.py`, `config.py`, `mailbox.py`, `gate1.py`, `probe.py`, and `tiles.py`.
- `main` adds `DeviceTrace`, `tail_by_stage`, request-window clock analysis, and `sweep.py`.
- Both branches independently changed shared files, so a blind merge could silently select incompatible semantics.

**What would falsify this**

- Evidence that the approved Stack A experiment should queue every frame rather than replace stale frames.
- Evidence that `controller.py` cannot be generalized without more work than introducing mailbox semantics into the v1 runner.

**Merge strategy**

1. Keep the v2 experiment contract and gate artifacts.
2. Port main's analysis in isolated commits with regression tests.
3. Generalize controller boundaries using both Stack A and Stack B as concrete requirements.
4. Keep old run compatibility in the reporter.
5. Do not delete the v1 Stack B path until the unified path reproduces an existing run.

### Q2 — Is there one source of truth for running experiments?

No. There are currently two orchestration systems:

- `runner.py` + `run_a1.py`/`run_b1.py`;
- `controller.py` + TOML cells.

This is not merely duplicate code. The two systems attach different meanings to a “latency run”:

- v1 fixed-rate mode queues all requests;
- v2 mailbox mode can overwrite requests;
- v1 writes `run.json`;
- v2 writes `manifest.json`;
- warm-up, configuration, output roots, and telemetry differ.

The next version should have one lifecycle path and an explicit scheduling policy.

### Q3 — Did `deadline-harness` improve or regress reporting?

Both:

**Gained**

- overwritten-aware outcomes;
- a CDF whose ceiling reflects unfinished frames;
- miss rate as a function of arbitrary deadline;
- latency/GPU-load timeline;
- compatibility with `manifest.json` and older `run.json`;
- reporting when some or all frames fail.

**Lost**

- stage-level decomposition;
- slowest-1% stage attribution;
- GPU clock at send and during individual requests;
- memory-clock data;
- period sweeps.

The target reporter needs both sets.

## Python and asyncio concepts from Round 1

### Why warm-up frames exist

The first few inference requests are often not representative of steady-state operation. They can pay one-time costs such as:

- CUDA context creation;
- model memory allocation;
- kernel/module loading;
- JIT compilation or kernel selection;
- page faults and filesystem cache population;
- HTTP connection setup;
- runtime data-structure initialization;
- GPU clock ramp-up from idle.

`controller.measure()` sends `warmup_frames` requests sequentially and discards their records:

```text
controller.py:88-90
for each warm-up body:
    await send(body)
```

This makes the measured cell closer to steady-state service.

That choice has a scientific consequence:

- it is appropriate for an always-on inference tenant;
- it does **not** measure container/model cold-start latency;
- it can warm the GPU clock immediately before the cell and influence the first measured request.

For the future microdatacenter, we likely need two separate measurements:

1. **cold start/readiness:** container launch to service-ready;
2. **steady-state inference:** after a defined warm-up protocol.

Warm-up count belongs in the config and manifest because changing it can change the result.

#### Warm-up and the dummy gate

The dummy engine draws random sleeps for warm-ups too. `gate1.py` must discard the same number of random draws before replaying measured frames. Otherwise its predicted service sequence would be shifted.

### `load`, `_cell`, and `cell` in `config.py`

These names operate at three different levels.

#### `load(path)` — parse the whole configuration file

`config.py:105-160`

Input:

```text
Path("configs/stack_a.toml")
```

Output:

```text
Config(...)
```

It:

1. reads the exact TOML bytes;
2. parses TOML into ordinary dictionaries;
3. constructs nested typed dataclasses;
4. validates duplicate cell IDs;
5. computes the config hash;
6. applies defaults.

Use:

```python
config = load(Path("configs/stack_a.toml"))
```

#### `_cell(raw)` — validate and convert one raw TOML cell

`config.py:81-102`

Input:

```python
{
    "id": "alone-T2",
    "cadence_s": 2.0,
    "frames": 500,
}
```

Output:

```python
CellConfig(
    id="alone-T2",
    cadence_s=2.0,
    frames=500,
    co_runner="none",
    power_mode="current",
)
```

The leading underscore means “private implementation helper by convention.” Python does not make it truly inaccessible.

It is also the fail-fast validation point:

- cadence and frames must be positive;
- unsupported co-runners are rejected;
- unsupported power modes are rejected.

#### `Config.cell(cell_id)` — select one already-parsed cell

`config.py:74-79`

Input:

```python
config.cell("alone-T2")
```

Output:

```text
the existing CellConfig named "alone-T2"
```

It does not parse TOML and does not create a new cell. It searches `config.cells` and raises a useful error if the ID does not exist.

#### Mental model

```text
TOML file
  │
  └── load(path)
        │
        ├── _cell(raw cell 1)
        ├── _cell(raw cell 2)
        └── returns Config
                   │
                   └── config.cell("chosen-id")
```

### What are `controller.py:24-42`?

They are imports: the controller is assembling functionality defined in other modules.

#### Configuration

```text
config_module
CellConfig
Config
```

The module import supports `config_module.load(...)`; the class imports provide type annotations.

#### Device state and telemetry

```text
GpuUtilization
PowerSampler
Tegrastats
device_snapshot
gpu_co_tenants
power_mode
```

These capture deployment state `D` and non-model tenants `NM`.

#### Runtime adapters

```text
llama_server
Engine
NO_REPLY
DummyEngine
```

These select and operate the model/runtime `M`.

#### Scheduling

```text
MailboxRun
run_mailbox
Frame
```

These implement workload arrival and stale-frame policy `W`.

#### Records, analysis, provenance, and inputs

```text
RequestRecord
write_records
report
harness_commit
Tile
load_tiles
```

These turn a run into auditable data.

#### Design interpretation

The controller is acting as a **composition root**: the place where independent components are wired together. It is reasonable for a composition root to import many pieces.

The warning sign is that it directly knows Stack A concepts such as tiles, prompts, max tokens, and llama settings. For the future multi-tenant system, it should orchestrate generic interfaces while workload/runtime adapters own those details.

### How `controller.py:224-226` decides whether to skip a cell

First, `main()` converts the user budget into an absolute monotonic deadline:

```python
stop_at = now + max_minutes * 60
```

Before each cell, it asks:

```text
If I start now and add the estimated cell duration,
will I cross stop_at?
```

Equivalent logic:

```python
if current_time + estimated_duration > global_deadline:
    skip cell
```

The estimate comes from:

```python
(warmup_frames + frames) * cadence_s + 120 seconds
```

The 120 seconds are a fixed allowance for model startup.

#### Current flaw

Warm-up requests are sent back-to-back, not once per cadence. Multiplying warm-up count by `cadence_s` overestimates long-cadence cells.

This code does **not** inspect recent latency or compute a statistical runtime prediction. It is a conservative arithmetic estimate.

#### Better future contract

Separate:

- model hashing/setup allowance;
- measured frame schedule duration;
- expected warm-up service time;
- shutdown/report allowance;
- hard versus soft budget behavior.

For example:

```text
estimate =
    startup_allowance
  + estimated_warmup_service
  + (frames - 1) * cadence
  + final_request_allowance
  + teardown_allowance
```

### Why compute SHA-256?

It is not primarily used to create a unique run ID.

SHA-256 is a deterministic fingerprint:

```text
same bytes  -> same hash
changed bytes -> overwhelmingly likely different hash
```

The config hash answers:

> Did these results come from exactly the same configuration file bytes?

The model-file hash answers:

> Did these results use exactly the same weight file bytes?

Paths and names are insufficient because a file can be replaced without changing its path.

Run-directory uniqueness is handled separately by cell ID plus timestamp.

#### What a hash does not prove

A hash does not by itself capture:

- environment variables;
- container image;
- Python dependency versions;
- GPU driver/runtime;
- uncommitted code unless separately recorded;
- external service settings;
- whether the run was uncontaminated.

That is why the manifest also records the git commit, engine version, device snapshot, power mode, and config values.

#### Security note

Here, SHA-256 is used for reproducibility and change detection, not for authentication. It does not prove who produced the file.

### Why `measure()` is `async`

`async def` defines a coroutine: a function that can pause at `await` without blocking the entire thread.

The key requirement is:

> While one inference request is waiting for llama-server, the emitter must still wake up and publish the next frame on time.

If the whole controller blocked synchronously on HTTP inference, the “world” would wait for the model. That would invalidate the experiment.

#### How `measure()` is entered

`run_cell()` calls:

```python
asyncio.run(measure(...))
```

`asyncio.run` creates an event loop, runs the coroutine, and closes the loop when it finishes.

#### `measure()` broken down

##### 1. Prepare all request bodies synchronously

```python
bodies = [engine.prepare(...) for each tile]
```

This happens before measured emission. For llama.cpp it includes base64/JSON request construction.

##### 2. Build static record metadata

```python
static = {
    engine,
    model,
    quant,
    power_mode,
}
```

These fields are copied into every served or overwritten record.

##### 3. Open a reusable async engine session

```python
async with engine.session() as send:
```

For llama.cpp, this keeps an `httpx.AsyncClient` alive rather than creating a new connection for every frame.

##### 4. Warm up sequentially

```python
await send(body)
```

Each `await` lets the event loop run while waiting for the engine, though no emitter has started yet.

##### 5. Define how one claimed frame is served

`send_frame(frame)`:

1. schedules a co-tenant snapshot;
2. stamps `t_send`;
3. awaits the engine reply;
4. converts exceptions into data;
5. builds a `RequestRecord`.

This nested function closes over `send`, `bodies`, `engine`, and `static`. Such a function is called a **closure**.

##### 6. Start the mailbox experiment

```python
await run_mailbox(send_frame, ...)
```

`run_mailbox()` starts the emitter and consumer concurrently.

#### Why async is suitable here

Most waiting is:

- sleeping until a scheduled time;
- waiting for an HTTP response;
- waiting for a frame/event;
- waiting for the co-tenant subprocess.

These are I/O/waiting operations, where cooperative async works well.

Async does not make CPU-bound Python faster. If model inference runs directly in Python and blocks the event-loop thread, it must move to a worker thread or process. `run_b1.py` already does that for LeRobot with a one-worker executor.

### What `RequestRecord.overwritten()` does

Yes: it creates the permanent record of the frame that was replaced before the model claimed it.

When a new frame enters a full mailbox:

1. `Mailbox.put(new)` returns the previous waiting frame.
2. `run_mailbox()` calls `RequestRecord.overwritten(previous...)`.
3. The result is appended to `run.records`.

The special record has:

- original request/frame ID;
- original arrival and deadline;
- `t_send=None`;
- no token times;
- no engine timings;
- `outcome="overwritten"`.

Because `missed` treats no answer as a miss, the frame is counted correctly.

The method does not itself maintain a hidden list. It constructs one record; the feeder appends that record to the run.

### Class method versus property

#### Normal instance method

```python
class Example:
    def do_work(self, x):
        ...
```

Called on an existing object:

```python
obj.do_work(x)
```

`self` is that object.

#### Class method

```python
class Example:
    @classmethod
    def from_special_case(cls, x):
        return cls(...)
```

Called on the class:

```python
Example.from_special_case(x)
```

`cls` is the class itself. Class methods are commonly used as **alternate constructors**.

`RequestRecord.overwritten(...)` is an alternate constructor for a particular kind of record. Using `cls(...)` also allows a subclass to inherit it and construct the subclass.

#### Property

```python
class Example:
    @property
    def duration(self):
        return self.end - self.start
```

Used like data:

```python
obj.duration
```

No parentheses are written. A property is useful for derived state that conceptually behaves like an attribute.

Examples here:

- `record.latency_s`;
- `record.waiting_s`;
- `record.missed`;
- `record.engine_total_ms`.

These values are computed from authoritative raw fields rather than stored separately.

#### Static method

```python
class Example:
    @staticmethod
    def validate(x):
        ...
```

Receives neither `self` nor `cls`. It is namespaced under a class but does not use object/class state.

#### Module-level function

```python
def write_records(path, records):
    ...
```

Often the simplest option when behavior does not naturally belong to an object.

#### Async function

```python
async def send(...):
    result = await ...
```

Returns a coroutine and can cooperatively pause.

#### Nested function / closure

```python
def outer():
    value = ...
    async def inner():
        use(value)
```

`send_frame()` is an example. It keeps related state local to one measurement session.

#### Generator

```python
def rows():
    yield one_row
```

Produces values lazily rather than returning all at once.

### Mailbox architecture: one async process versus alternatives

#### Current architecture

```text
one Python process
└── one asyncio event loop
    ├── emitter coroutine
    └── consumer coroutine
            │
            └── async HTTP request to separate llama-server process
```

The emitter and consumer share one in-memory `Mailbox` object.

There are no locks because:

- only one OS thread touches the mailbox;
- coroutines switch only at `await`;
- `Mailbox.put()` contains no `await`;
- the critical read/update operations cannot be interrupted halfway by another coroutine.

The event loop is cooperative concurrency, not parallel Python execution.

#### Why an async loop is needed

During:

```python
reply = await send(...)
```

the model may take 1.5 seconds. `await` gives control back to the event loop. The emitter can wake at its scheduled time, place a new frame, and record overwrites while the HTTP request is still in flight.

Without concurrency:

```text
emit frame -> wait 1.5 s for model -> emit next frame
```

The input rate would be controlled by model speed, which contradicts the thesis.

#### Alternative 1: threads

```text
one process
├── emitter thread
└── consumer thread
```

This would require a thread-safe queue/condition and locks. It can work, but introduces OS-thread scheduling and GIL interactions. Blocking libraries are easier to use from threads.

#### Alternative 2: separate processes with Queue/ZeroMQ

```text
emitter process ── IPC ──> model-client process
```

Advantages:

- stronger fault and scheduling isolation;
- closer to a multi-service deployment;
- one blocked Python process cannot directly stop the other;
- natural bridge to containers.

Costs:

- serialization and IPC overhead;
- process lifecycle complexity;
- readiness and shutdown protocols;
- more difficult overwrite accounting;
- more system noise.

#### ZeroMQ `CONFLATE=1`

Conflation keeps only the newest unread message, which resembles the latest-frame mailbox.

However, “same semantics” needs qualification:

- latest-value delivery can match;
- explicit knowledge of *which* frames were overwritten may require sequence-number reconciliation or acknowledgments;
- PUB/SUB startup can drop messages before subscriptions are ready;
- process crashes and reconnects create new failure modes.

Therefore, replacing the in-memory mailbox is not only an implementation swap. It requires re-validating record conservation and overwrite observability.

#### Future container architecture

For a space microdatacenter, the experiment controller should likely run outside tenant containers:

```text
host experiment controller
├── workload/emitter service
├── VLM tenant container
├── VLA tenant container
├── controlled NM/co-runner containers
└── host telemetry service
```

The current async mailbox is still useful as one scheduling policy, but process/container boundaries should become explicit experimental axes.

### What happens at `feeder.py:55`

```python
for req_id, t_arrival in enumerate(arrival_times(start, period_s, n_requests)):
```

Breakdown:

1. `arrival_times(...)` returns absolute scheduled times:

   ```text
   start
   start + period
   start + 2*period
   ...
   ```

2. `enumerate(...)` pairs each time with an integer ID:

   ```text
   (0, start)
   (1, start + period)
   (2, start + 2*period)
   ```

3. Tuple unpacking assigns those values to:

   ```text
   req_id
   t_arrival
   ```

4. The loop sleeps until that absolute time, computes the deadline, and starts `send_one` as an independent task.

Using absolute times prevents drift. If request 3 wakes 5 ms late, request 4 still targets `start + 4T`, not “one period after request 3 actually woke.”

#### Subtle semantic difference

In `run_fixed_rate`, `t_arrival` is the ideal scheduled time.

In `run_mailbox`, the loop first obtains scheduled `t_arrival`, then stamps:

```python
t_obs = time.monotonic()
```

and stores `t_obs` as the frame's actual arrival.

So the two modes do not use precisely the same arrival semantics when the emitter wakes late. This should be made explicit or unified before results from the modes are directly compared.

### Asyncio mental model for `feeder.py`

The event loop runs one coroutine at a time until that coroutine reaches an `await`.

Think of two people sharing one marker:

- emitter uses the marker until it has to wait for the next timestamp;
- consumer uses it until it has to wait for a frame or HTTP reply;
- whenever one waits, it hands the marker back to the event loop.

#### `run_fixed_rate`

```text
at each scheduled time:
    create a new send task
continue to next scheduled time without waiting for reply
after all requests have been created:
    wait until every task finishes
```

Important: this can create a large number of queued tasks if inference is slower than arrivals.

#### `run_closed_loop`

```text
send one
wait for it to finish
send next
```

The model determines the input rate. This measures saturation throughput, not deadline behavior.

#### `run_mailbox`

`asyncio.gather(emit(), consume())` starts two coroutines.

Emitter:

```text
sleep until tick
publish newest frame
record any replaced frame
repeat
close mailbox
```

Consumer:

```text
wait for a frame
remove it from mailbox
send it to model
wait for model response
repeat
```

Example with model service time `2.5T`:

```text
t=0    emitter puts frame 0
       consumer takes frame 0 and waits for model

t=1T   emitter puts frame 1

t=2T   emitter puts frame 2
       frame 1 is still waiting, so frame 1 is overwritten

t=2.5T model finishes frame 0
       consumer takes frame 2
```

The emitter continues because the consumer's engine request is an `await`, not a blocking call on the event-loop thread.

### Asyncio safety condition to remember

The architecture works only while work inside the event-loop thread remains short or awaitable.

Dangerous pattern:

```python
async def supposedly_async():
    do_cpu_or_gpu_work_for_two_seconds_without_await()
```

Despite `async def`, that blocks the event loop for two seconds. The emitter cannot run.

That is why a blocking in-process LeRobot call needs a worker thread/process, while async HTTP to llama-server can remain on the loop.

## Emerging target architecture

### The final product should drive the abstraction

The target is not merely “Stack A plus Stack B.” It is a shared-GPU edge microdatacenter in which several independently deployed workloads contend for one accelerator.

Examples:

- multiple Docker containers hosting VLM/VLA services;
- two robot policies sharing one GPU;
- one satellite VLM plus detection/compression tenants;
- controlled and uncontrolled co-tenants;
- different power modes and clock policies.

Containers provide packaging and process boundaries, but they do not automatically provide GPU isolation. Separate containers can still:

- create separate CUDA contexts;
- contend for compute and memory bandwidth;
- reserve GPU/unified memory;
- perturb clocks and power;
- block one another in driver/runtime layers.

That contention is part of the experiment, not an implementation accident.

### The system needs two distinct layers

#### 1. Experiment control plane

Responsible for:

- validating the matrix;
- starting/stopping containers or processes;
- waiting for readiness;
- setting/restoring device state;
- starting telemetry;
- applying workload schedules;
- collecting raw data;
- marking contamination;
- guaranteeing cleanup.

This should run outside the tenant under test.

#### 2. System under test

Contains:

- tenant inference services;
- their runtime/model/container image;
- co-located non-model services;
- the actual transport and GPU contention.

Keeping these layers separate prevents the harness from accidentally becoming part of the service architecture it claims to measure.

### Proposed component boundaries

```text
ExperimentController
│
├── CellSpec
│   ├── workload
│   ├── scheduling policy
│   ├── tenants
│   ├── device policy
│   └── stop/admission policy
│
├── WorkloadAdapter
│   ├── prepare inputs
│   ├── define output/quality semantics
│   └── attach workload metadata
│
├── SchedulingPolicy
│   ├── latest-only mailbox
│   ├── bounded/FIFO stream
│   ├── action-chunk refresh
│   └── closed-loop saturation
│
├── RuntimeAdapter
│   ├── start/readiness/stop
│   ├── infer
│   ├── owned process/container IDs
│   └── engine-reported stages
│
├── TenantManager
│   ├── Docker/process launch
│   ├── controlled co-runners
│   └── failure cleanup
│
├── DeviceContext
│   ├── telemetry
│   ├── nvpmodel/clocks
│   └── restore
│
├── Recorder
│   └── append-only raw events
│
└── Reporter
    └── offline derived metrics
```

### Scheduling policies should be first-class

The final system should not assume one queue policy for every workload.

#### Space-image triage

Likely policy:

```text
latest-only frame mailbox
```

A stale unclaimed frame may have less value than the newest observation.

#### VLA action policy

Possible policy:

```text
new observation arrives at control rate;
new action chunk must be ready before committed actions run out;
policy may use latest observation, queue one request, or intentionally skip.
```

This is not automatically identical to image replacement. The deadline may be the remaining action horizon, not simply the next observation.

#### Event workloads

Some events cannot be overwritten and may require bounded FIFO plus backpressure or admission control.

#### Throughput characterization

Closed-loop saturation remains useful, but should be explicitly labeled as a capacity experiment.

### Common record schema

The current `RequestRecord` is already moving in the right direction:

- generic arrival/send/completion timestamps;
- terminal outcome;
- engine/model/power metadata;
- optional token fields;
- optional action count;
- optional named engine stages.

The next schema should add or clarify:

- schema version;
- workload ID and tenant ID;
- scheduling-policy name/version;
- container image digest;
- actual versus scheduled arrival;
- explicit completion timestamp independent of token naming;
- drop/overwrite reason;
- co-runner identities;
- contamination status at cell level;
- request body/input identity without storing huge payloads.

### How to incorporate the staff-engineer comments

#### Comment: one source of truth

Action:

- make the controller the only lifecycle entry point;
- make Stack A and Stack B plugins/adapters;
- retain old CLIs as thin compatibility wrappers or deprecate them.

#### Comment: metrics with different meanings

Action:

- store scheduling policy in the manifest;
- rename paced `throughput_rps` to `completion_rate_rps`;
- reserve `service_capacity_rps` for closed-loop/saturation runs;
- include schema and metric-definition versions.

#### Comment: future multi-tenant system

Action:

- promote co-runners from a string field to explicit managed tenants;
- record process/container IDs and image digests;
- define readiness, health, and teardown contracts;
- distinguish controlled tenants from unexpected contamination.

#### Comment: branch capabilities

Action:

- port analysis from `main` only after tests capture deadline-branch behavior;
- ensure stage/tail analysis excludes overwritten frames but miss metrics include them;
- restore clock-window analysis using the new telemetry files.

#### Comment: keep scheduling explicit

Action:

- do not create one generic `run()` whose behavior changes implicitly by stack;
- require `schedule = mailbox | fixed_queue | closed_loop | action_chunk`;
- report schedule in every manifest and comparison.

## Provisional pre-merge recommendations

These are draft review requests for the author. They should be refined after Rounds 2-5.

### Must resolve before merging to `main`

#### 1. Document the integration architecture

Add a short decision record naming:

- `controller.py` as the future lifecycle path;
- scheduling as an explicit policy;
- Stack A and Stack B as workload/runtime adapters;
- the plan for deprecating or wrapping `run_a1.py`, `run_b1.py`, and `runner.py`.

Reason: without an explicit migration plan, merging creates two authoritative ways to run experiments.

#### 2. Preserve main's analysis capabilities

Port with tests:

- `DeviceTrace`;
- `stages_ms`;
- `tail_by_stage`;
- request-window GPU clock analysis;
- memory-clock sampling if the Thor source remains valid;
- a revised period sweep.

Reason: the deadline branch answers whether frames miss; main helps answer why.

#### 3. Add schema and scheduling-policy metadata

The manifest should identify:

- record schema version;
- scheduling policy and version;
- metric definition version if summaries are committed/shared.

Reason: “latency run” currently has two meanings.

#### 4. Fix or rename `frames_delivered`

`controller.py:182` sets:

```python
frames_delivered = len(run.records)
```

But `run.records` includes overwritten frames, which were never delivered to the model.

Use separate counts such as:

- `frames_emitted`;
- `records_written`;
- `frames_served`;
- `frames_overwritten`;
- `frames_error`.

Reason: the current field name can mislead readers and downstream tools.

#### 5. Fix the cell-duration estimate

Do not price warm-ups at the sensor cadence. Make estimate components explicit and add boundary tests for `--max-minutes`.

Reason: long-cadence cells can be skipped even when they fit.

#### 6. Preserve old-run compatibility

The merged reporter must load:

- old v1 `run.json` runs;
- new `manifest.json` cells;
- old `power.csv`;
- new telemetry files.

Use fixture/golden tests with existing run formats.

Reason: old raw runs are expensive scientific artifacts.

#### 7. Add a Stack B mailbox test before claiming unification

Use a slow fake action policy and prove:

```text
emitted = served + overwritten + errors
```

Then reproduce one existing SmolVLA/pi0.5 result through the unified path.

Reason: a generic interface is not validated until both concrete stacks use it.

### Strongly recommended before expensive new cells

#### 8. Separate functional probe from latency probe

Do not interleave every timed one-token request with a four-token diagnostic request.

Reason: the diagnostic changes GPU state and request sequence.

#### 9. Define alone-cell admission and contamination policy

Process presence alone is too strict; logging contamination without a decision is too weak.

Define:

- preflight quiet window;
- allowed processes;
- utilization threshold;
- during-run invalidation;
- abort versus finish-and-mark-invalid behavior.

#### 10. Quantify observer effect

A/B test:

- per-request `gpu_co_tenants()` on/off;
- telemetry at 100 Hz, 10 Hz, and off.

Reason: the harness spawns/reads while inference is measured.

#### 11. Treat co-runners as managed processes/containers

Require:

- ready handshake;
- per-iteration log;
- owned PIDs/container IDs;
- cleanup on failure;
- explicit manifest identity.

#### 12. Add power-mode transaction tests

Verify:

- set requested mode;
- confirm it took effect;
- restore original mode after success/error/interrupt;
- detect inherited wrong mode on next startup.

### Not necessarily pre-merge blockers

- replacing asyncio with ZeroMQ;
- splitting emitter into another process;
- optimizing sysfs reads;
- introducing a general plugin framework;
- supporting every future runtime.

These should be driven by measured jitter, observer-effect evidence, or the first concrete multi-container experiment.

## Learning ledger

### Concepts now understood

#### Configuration flow

```text
load(file) -> parse all config
_cell(raw) -> validate/build one CellConfig
config.cell(id) -> select one existing cell
```

#### Hashing

SHA-256 fingerprints exact bytes for reproducibility; it does not generate the run ID and does not capture the whole environment.

#### Warm-up

Warm-up removes initialization transients from steady-state measurement, but cold start must be measured separately.

#### Asyncio

Asyncio gives one thread cooperative concurrency. Coroutines switch at `await`. It is useful because the emitter must continue while inference is waiting on HTTP.

#### Mailbox

The mailbox has capacity one. A new unclaimed frame replaces the old one. Served-late and overwritten are different failure modes.

#### Python method forms

- instance method: uses `self`;
- class method: uses `cls`, often alternate constructor;
- property: computed value accessed like a field;
- static method: class-namespaced function without `self`/`cls`;
- module function: ordinary standalone behavior;
- async function: coroutine;
- nested function: local closure;
- generator: lazily yields values.

#### Branch discernment

A branch can improve one layer and regress another. Review should produce a capability union and semantic decision, not a winner/loser label.

### Important distinctions to retain

- scheduled arrival versus actual observed arrival;
- cold-start versus steady-state latency;
- completion rate versus service capacity;
- late completion versus overwrite;
- gross rail energy versus isolated model energy;
- process presence versus active GPU contention;
- container boundary versus GPU isolation;
- concurrency versus parallelism;
- scientific validity versus stylistic cleanliness.

### Questions I should now be able to answer

1. Why would a synchronous emitter invalidate the experiment?
2. When is lock-free async state safe?
3. Why does a hash belong in a manifest?
4. Why are warm-up frames both necessary and potentially biasing?
5. Why can the same metric name mean different things under different scheduling policies?
6. Why is Docker packaging not GPU isolation?
7. Why should a controller be generic while scheduling remains explicit?

## Open questions for later rounds

### Round 2

- Does per-request `nvidia-smi` materially perturb latency?
- Should deadline start at scheduled arrival or actual observation?
- What exact guarantee should `--max-minutes` provide?
- Does telemetry affect emitter jitter or inference latency?

### Round 3

- Should synchronized stage attribution and optimized overlapped execution be separate modes?
- Which preprocessing/serialization costs belong inside the deployment latency?
- Is 100 Hz telemetry the right accuracy/overhead trade-off?
- Should runtime adapters run in-process, over HTTP/gRPC, or inside managed containers?

### Round 4

- How should energy be attributed across waiting, served, overwritten, and concurrent tenants?
- Which Gate 2 claims are reproducible from committed raw data?
- How should confidence bounds determine frame counts?
- What provenance mismatches should make `compare.py` refuse a comparison?

### Round 5

- What scheduling contract best represents VLA action chunks?
- What is the minimum common runtime interface across llama.cpp, LeRobot, and future vLLM?
- How should unexpected and controlled tenants be represented?
- Which changes are true merge blockers versus follow-up milestones?

### Round 6

- Does every process a cell starts get stopped on every failure path (MPS daemon, load generators)?
- Is "bursty time-slicing vs bursty MPS" a fair comparison when neighbours' achieved load differs?
- When do visualizer bundles split frames wrongly, and which of their numbers are modeled?
- What can the ncu summary support before the "typical calls" fix?
