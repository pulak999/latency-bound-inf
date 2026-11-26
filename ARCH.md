# ARCH.md

## Overview

A measurement rig. A simulated world feeds a real device (the Thor) at a fixed rate; the device runs a model through
some inference engine; a harness records when each answer arrives against its deadline. Mapped to slide 6:

| Axis | Meaning | Stack A | Stack B |
|---|---|---|---|
| W | input trace | satellite tiles, one per T seconds | LIBERO observations at 30 Hz |
| M | model x engine | Gemma 3 4B on llama.cpp / vLLM / TRT-LLM / transformers | pi0.5, SmolVLA on LeRobot / openpi / EagleVLA-Edge |
| NM | non-model load | labmates' jobs, then own detector / encoder / scrubber | VIO stand-in, preprocessing, second policy |
| D | deployment | power modes 70 W to MAXN | same, later a remote "connected" cell |

Current scope uses W (synthetic inputs) and M (both stacks). NM and D are held fixed: no co-located load, 120 W. No simulator yet.

## Components

- **Engines** (outside git, under `/data/projects/pulak/src` and `envs`). Each exposes a local endpoint:
  llama-server (HTTP, OpenAI-compatible, port 8081), vLLM (HTTP, `/metrics`), LeRobot policy server (gRPC),
  openpi `serve_policy.py` (websocket, port 8000), EagleVLA-Edge (llama.cpp fork).
- **Harness** (`harness/`, in git):
  - *Synthetic source* — seeded generator of fixed-shape inputs: images for Stack A; camera images plus robot state for Stack B.
  - *Controller* (`harness/controller.py`, Stack A) — reads a TOML config from `configs/`, and for each
    cell starts the samplers and the engine, sends the warm-up frames, runs the mailbox feeder, and writes
    `manifest.json`, `records.jsonl` and the telemetry files into the cell's own directory.
  - *Feeder* (`harness/feeder.py`) — three schedules. Mailbox (Stack A cells): frame i fires at `t0 + i*T`
    into a depth-1 mailbox (`harness/mailbox.py`); the model takes the newest frame when free, and a frame
    replaced before it is taken is recorded as `overwritten`. Fixed rate (Stack B, older Stack A runs): every
    request is sent at its tick and queues in the engine. Closed loop: back to back, for throughput.
  - *Client adapter* — one per engine; sends a request, returns client timestamps and engine-reported timings.
  - *Recorder* — appends one JSONL record per request (fields below).
  - *Device logger* (`harness/device.py`) — `tegrastats --interval 100`, GPU and board power plus GPU clock
    from sysfs every 10 ms (`power.csv`), and GPU load from `nvidia-smi` (`gpu_util.csv`, controller only).
  - *Reporter* — offline: p50, p99, p99/p50, miss rate with Wilson CI, latency CDF, throughput in requests/s, energy per request from rail power.
  - *Runner* (`harness/runner.py`) — the older path, still used by Stack B (`run_b1.py`) and by
    `run_a1.py` for Stack A throughput runs: command line, run directory, power and tegrastats capture,
    then the report. Its runs write `run.json` instead of `manifest.json`; the reporter reads both.
  - *Engine adapters* (`harness/engines/`): `llama_server` (HTTP, streaming) and
    `lerobot_policy` (in-process; one worker thread so one request is on the GPU at a time).
- **Setup scripts** (`setup/`, in git): reproducible build and download steps per plan section.
- **Environments:** Stack A's client runs in the repo's uv env. Stack B runs in
  `/data/projects/pulak/envs/lerobot`, with the harness installed into it in editable mode.

## Data flow

```
images / sim obs --> feeder --(every T s)--> client adapter --> engine on GPU
                                                  |                  |
                                                  v                  v
                                     runs/<run>/records.jsonl   tegrastats.log
                                                  \                 /
                                                   +--> reporter --> CDF, p99, miss rate, J/request
```

## Per-request record

`req_id, t_arrival, t_send, t_first_token, t_last_token, deadline, missed, outcome, tile_id, n_prompt_tokens, n_output_tokens, engine, model, quant, power_mode, co_tenants`
(`outcome` is `ok`, `overwritten` or `error`; an overwritten frame has no `t_send` and counts as a miss)
plus engine-side timings where the engine reports them: prompt and decode ms for LLM engines,
named stages (`engine_stages_ms`: preprocess, inference, postprocess) for robot policies, and
`n_actions` for the chunk length. A policy's chunk arrives at once, so `t_first_token` is None
and `t_last_token` is when the chunk was ready.
