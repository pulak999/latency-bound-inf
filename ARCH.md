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
  - *Feeder* — latency mode fires request i at `t0 + i*T` on a monotonic clock, independent of replies.
    Throughput mode sends back to back until the engine saturates.
  - *Client adapter* — one per engine; sends a request, returns client timestamps and engine-reported timings.
  - *Recorder* — appends one JSONL record per request (fields below).
  - *Device logger* — runs `tegrastats --interval 100 --logfile` for the run's duration.
  - *Reporter* — offline: p50, p99, p99/p50, miss rate with Wilson CI, latency CDF, throughput in requests/s, energy per request from rail power.
- **Setup scripts** (`setup/`, in git): reproducible build and download steps per plan section.

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

`req_id, t_send, t_first_token, t_last_token, n_prompt_tokens, n_output_tokens, deadline, missed, engine, model, quant, power_mode, co_tenants`
plus engine-side timings where the engine reports them.
