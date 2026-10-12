"""Stack A run: Gemma 3 4B (QAT Q4_0) on llama-server. Synthetic image in, one label out.

    uv run python -m harness.run_a1 latency --period 30 --n-requests 100
    uv run python -m harness.run_a1 throughput --duration 300

Each run gets its own directory under /data/projects/pulak/runs/ holding
run.json (config), records.jsonl, power.csv, tegrastats.log, llama-server.log,
and the report's summary.json and latency_cdf.png.
"""

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

from harness.device import PowerSampler, Tegrastats, gpu_co_tenants, power_mode
from harness.engines import llama_server
from harness.feeder import run_closed_loop, run_fixed_rate
from harness.records import RequestRecord, write_records
from harness.report import report
from harness.synthetic import GEMMA3_IMAGE_SIZE, image_pool

RUNS_DIR = Path("/data/projects/pulak/runs")
MODEL_DIR = Path("/data/projects/pulak/models/gemma-3-4b-it-qat-q4_0")
SERVER = llama_server.LlamaServerConfig(
    model=MODEL_DIR / "gemma-3-4b-it-q4_0.gguf",
    mmproj=MODEL_DIR / "mmproj-model-f16-4B.gguf",
)

PROMPT = "Classify this satellite scene with one word from: forest, water, urban, cloud, field."
N_OUTPUT_TOKENS = 4  # fixed output length, so latency reflects the system and not the answer
IMAGE_POOL_SIZE = 64  # distinct images, cycled; request i uses image i % 64
N_WARMUP = 3  # first requests pay one-off CUDA setup; sent but not recorded
REQUEST_TIMEOUT_S = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)

    latency = modes.add_parser("latency", help="fixed-rate input; report latency and misses")
    latency.add_argument(
        "--period",
        dest="period_s",
        type=float,
        default=30.0,
        help="seconds between inputs; also the deadline (default 30)",
    )
    latency.add_argument("--n-requests", type=int, default=100)

    throughput = modes.add_parser("throughput", help="back-to-back requests; report requests/s")
    throughput.add_argument(
        "--duration",
        dest="duration_s",
        type=float,
        default=300.0,
        help="seconds to keep sending (default 300)",
    )
    return parser.parse_args()


async def measure(args: argparse.Namespace, server: llama_server.LlamaServer, mode: str):
    """Warm up, then send requests on the chosen schedule. Returns one record per request."""
    # Encode every request body before timing starts.
    bodies = [
        llama_server.encode_request(png, PROMPT, N_OUTPUT_TOKENS)
        for png in image_pool(IMAGE_POOL_SIZE)
    ]
    our_pids = {server.process.pid}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        for body in bodies[:N_WARMUP]:
            await llama_server.send(client, SERVER.url, body)

        async def send_one(req_id: int, t_arrival: float, deadline: float | None):
            co_tenants = asyncio.create_task(gpu_co_tenants(exclude_pids=our_pids))
            t_send = time.monotonic()
            try:
                reply = await llama_server.send(
                    client, SERVER.url, bodies[req_id % IMAGE_POOL_SIZE]
                )
                error = None
            except httpx.HTTPError as exc:
                reply, error = llama_server.NO_REPLY, repr(exc)

            return RequestRecord(
                req_id=req_id,
                t_arrival=t_arrival,
                t_send=t_send,
                t_first_token=reply.t_first_token,
                t_last_token=reply.t_last_token,
                deadline=deadline,
                n_prompt_tokens=reply.n_prompt_tokens,
                n_output_tokens=reply.n_output_tokens,
                output_text=reply.text,
                engine_prompt_ms=reply.engine_prompt_ms,
                engine_decode_ms=reply.engine_decode_ms,
                engine=f"llama.cpp@{server.commit()[:7]}",
                model="gemma-3-4b-it",
                quant="QAT Q4_0",
                power_mode=mode,
                co_tenants=await co_tenants,
                error=error,
            )

        if args.mode == "latency":
            return await run_fixed_rate(send_one, args.period_s, args.n_requests)
        return await run_closed_loop(send_one, args.duration_s)


def harness_commit() -> str:
    """This repo's commit, with '-dirty' if there are uncommitted changes."""
    return subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    run_dir = RUNS_DIR / f"a1-{args.mode}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    mode = power_mode()

    with llama_server.LlamaServer(SERVER, logfile=run_dir / "llama-server.log") as server:
        config = vars(args) | {
            "stack": "A1",
            "engine_commit": server.commit(),
            "harness_commit": harness_commit(),
            "model": str(SERVER.model),
            "mmproj": str(SERVER.mmproj),
            "prompt": PROMPT,
            "n_output_tokens": N_OUTPUT_TOKENS,
            "image_size": GEMMA3_IMAGE_SIZE,
            "image_pool_size": IMAGE_POOL_SIZE,
            "n_warmup": N_WARMUP,
            "power_mode": mode,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        (run_dir / "run.json").write_text(json.dumps(config, indent=2) + "\n")

        with Tegrastats(run_dir / "tegrastats.log"), PowerSampler() as power:
            records = asyncio.run(measure(args, server, mode))

    write_records(run_dir / "records.jsonl", records)
    power.write_csv(run_dir / "power.csv")
    print(json.dumps(report(run_dir), indent=2))
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
