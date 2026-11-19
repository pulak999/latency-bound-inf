"""Profile a few served frames with nsys and save them as a bundle.

    uv run python -m visualizer.capture CONFIG --name alone [--frames 5] [--period 3]
        [--cell CELL_ID]             neighbours and GPU sharing from that cell of CONFIG
        [--server-arg=--flash-attn=off ...]
        [--notes "why this capture"]

The measured server runs on the host under `nsys launch`, so model loading and
warm-up frames are not traced; tracing starts just before the first measured
frame and stops after the last. Neighbours, if the cell has any, run in their
own containers exactly as in a cell (harness/tenancy.py) and are not traced.
Every frame sends the same tile, so frames are comparable.

Writes <profiles dir>/<YYYYmmdd-HHMMSS>-<name>/ with profile.nsys-rep,
profile.sqlite, frames.jsonl, meta.json, server.log and bundle.json.
GPU metrics (SM activity, DRAM bandwidth over time) need root: pass
--gpu-metrics and run under sudo.
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

from harness import config as config_module
from harness.config import CellConfig
from harness.engines.llama_server import encode_request, send, server_command
from harness.runner import harness_commit
from harness.tenancy import Tenants, server_config
from harness.tiles import load_tiles
from visualizer import bundle as bundle_module

PROFILES_DIR = Path("/data/projects/rohan/profiles")
GPU_CLOCK = Path("/sys/class/devfreq/gpu-gpc-0/cur_freq")


def nsys(*args: str) -> str:
    result = subprocess.run(["nsys", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nsys {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result.stdout


def wait_until_healthy(url: str, process: subprocess.Popen, timeout_s: float = 180) -> None:
    give_up_at = time.monotonic() + timeout_s
    while time.monotonic() < give_up_at:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited early; see server.log")
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server not healthy after {timeout_s} s")


async def send_frames(url: str, body: bytes, n: int, period_s: float) -> list[dict]:
    """n requests, one every period_s (or back to back if 0). One dict per frame."""
    rows = []
    async with httpx.AsyncClient(timeout=600) as client:
        start = time.monotonic()
        for i in range(n):
            await asyncio.sleep(max(0.0, start + i * period_s - time.monotonic()))
            t_send = time.monotonic()
            reply = await send(client, url, body)
            rows.append(
                {
                    "i": i,
                    "latency_ms": (reply.t_last_token - t_send) * 1e3,
                    "prompt_ms": reply.engine_prompt_ms,
                    "n_prompt_tokens": reply.n_prompt_tokens,
                    "text": reply.text,
                }
            )
    return rows


def gpu_clock_mhz() -> float | None:
    try:
        return int(GPU_CLOCK.read_text()) / 1e6
    except (OSError, ValueError):
        return None


def capture(args: argparse.Namespace) -> Path:
    config = config_module.load(args.config)
    cell = (
        config.cell(args.cell)
        if args.cell
        else CellConfig(id=args.name, cadence_s=args.period, frames=args.frames)
    )
    bundle_dir = args.out / f"{datetime.now():%Y%m%d-%H%M%S}-{args.name}"
    bundle_dir.mkdir(parents=True)
    tiles, tile_source = load_tiles(config.tiles)
    body = encode_request(tiles[0].png, config.model.prompt, config.model.max_tokens)
    session = f"lbi-{bundle_dir.name}"

    meta = {
        "name": args.name,
        "notes": args.notes,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "harness_commit": harness_commit(),
        "config": str(args.config),
        "cell": cell.id if args.cell else None,
        "tenants": cell.tenants,
        "load": cell.load,
        "sharing": cell.sharing,
        "server_args": args.server_arg,
        "frames": args.frames,
        "warmup_frames": args.warmup,
        "period_s": args.period,
        "tile": tiles[0].tile_id,
        "tiles": tile_source,
        "measured_tenant": "host process under nsys",
        "nsys_version": nsys("--version").strip(),
        "gpu_metrics": args.gpu_metrics,
    }

    with Tenants(config, cell, bundle_dir) as tenants:
        meta["tenancy"] = tenants.describe()
        server = server_config(config, config.llama_server.port, None, tenants.tenant0_env())
        trace = ["--trace=cuda,nvtx", "--cuda-graph-trace=node"]
        if args.gpu_metrics:
            trace += ["--gpu-metrics-devices=all"]
        command = ["nsys", "launch", f"--session-new={session}", *trace]
        command += server_command(server) + args.server_arg
        with (bundle_dir / "server.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=os.environ | server.env
            )
            try:
                wait_until_healthy(server.url, process)
                asyncio.run(send_frames(server.url, body, args.warmup, 0))
                meta["gpu_clock_mhz_before"] = gpu_clock_mhz()
                nsys("start", f"--session={session}", f"--output={bundle_dir / 'profile'}")
                frames = asyncio.run(send_frames(server.url, body, args.frames, args.period))
                nsys("stop", f"--session={session}")
                meta["gpu_clock_mhz_after"] = gpu_clock_mhz()
            finally:
                subprocess.run(
                    ["nsys", "shutdown", f"--session={session}", "--kill=sigterm"],
                    capture_output=True,
                )
                process.wait(timeout=60)

    (bundle_dir / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    nsys(
        "export", "--type=sqlite", "--force-overwrite=true",
        f"--output={bundle_dir / 'profile.sqlite'}", str(bundle_dir / "profile.nsys-rep"),
    )  # fmt: skip
    bundle_module.build(bundle_dir)
    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", required=True, help="short label, e.g. alone or fa-off")
    parser.add_argument("--cell", help="take neighbours and sharing from this cell of CONFIG")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--period", type=float, default=3.0, help="seconds between frames")
    parser.add_argument("--server-arg", action="append", default=[], help="extra llama-server arg")
    parser.add_argument("--notes", default="")
    parser.add_argument("--gpu-metrics", action="store_true", help="needs root")
    parser.add_argument("--out", type=Path, default=PROFILES_DIR)
    args = parser.parse_args()
    bundle_dir = capture(args)
    print(f"bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
