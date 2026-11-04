"""Run the cells of a config: one model, one fixed-rate frame stream, one log per cell.

    uv run python -m harness.controller configs/stack_a.toml [--cells alone-T20 ...]
                                        [--engine llama|dummy] [--max-minutes N]

For each cell the controller starts the samplers, starts the engine, sends the
warm-up frames, runs the emitter through the mailbox for the cell's frames,
stops everything, writes manifest.json, records.jsonl, power.csv, gpu_util.csv
and tegrastats.log into runs/<cell id>-<timestamp>/, and reports.

Power modes and co-runners are later extensions. The config loader refuses a
cell that asks for either, so a run never silently lacks what it was meant to have.
"""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from harness import config as config_module
from harness.config import CellConfig, Config
from harness.device import (
    GpuUtilization,
    PowerSampler,
    Tegrastats,
    device_snapshot,
    gpu_co_tenants,
    power_mode,
)
from harness.engines import llama_server
from harness.engines.base import NO_REPLY, Engine
from harness.engines.dummy import DummyEngine
from harness.feeder import MailboxRun, run_mailbox
from harness.mailbox import Frame
from harness.records import RequestRecord, write_records
from harness.report import report
from harness.runner import harness_commit
from harness.tiles import Tile, load_tiles

ENGINE_START_ALLOWANCE_S = 120  # model load, counted against --max-minutes


def make_engine(kind: str, config: Config, cell_dir: Path) -> Engine:
    if kind == "dummy":
        return DummyEngine(config.dummy, label=config.model.labels[0])
    if config.llama_server is None:
        raise ValueError(f"{config.path} has no [llama_server] section")
    settings = config.llama_server
    server = llama_server.LlamaServerConfig(
        model=settings.model,
        mmproj=settings.mmproj,
        port=settings.port,
        server_bin=settings.server_bin,
        request_timeout_s=settings.request_timeout_s,
    )
    return llama_server.LlamaServer(server, logfile=cell_dir / "llama-server.log")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def measure(
    engine: Engine,
    config: Config,
    cell: CellConfig,
    tiles: list[Tile],
    run: MailboxRun,
    stop_at: float | None,
) -> None:
    """Warm up, then feed the cell's frames through the mailbox. Fills `run` in place."""
    bodies = [engine.prepare(t.png, config.model.prompt, config.model.max_tokens) for t in tiles]
    static = {
        "engine": f"{engine.name}@{engine.version[:7]}",
        "model": config.model.name,
        "quant": config.model.quant,
        "power_mode": power_mode(),
    }

    async with engine.session() as send:
        for i in range(config.harness.warmup_frames):
            await send(bodies[i % len(bodies)])

        async def send_frame(frame: Frame) -> RequestRecord:
            co_tenants = asyncio.create_task(gpu_co_tenants(exclude_pids=engine.pids()))
            t_send = time.monotonic()
            try:
                reply, error = await send(bodies[frame.frame_id % len(bodies)]), None
            except Exception as exc:  # noqa: BLE001 - a failed request is a data point
                reply, error = NO_REPLY, repr(exc)
            return RequestRecord(
                req_id=frame.frame_id,
                t_arrival=frame.t_obs,
                t_send=t_send,
                t_first_token=reply.t_first_token,
                t_last_token=reply.t_last_token,
                deadline=frame.deadline,
                n_prompt_tokens=reply.n_prompt_tokens,
                n_output_tokens=reply.n_output_tokens,
                output_text=reply.text,
                engine_prompt_ms=reply.engine_prompt_ms,
                engine_decode_ms=reply.engine_decode_ms,
                co_tenants=await co_tenants,
                error=error,
                tile_id=frame.tile_id,
                **static,
            )

        await run_mailbox(
            send_frame,
            period_s=cell.cadence_s,
            n_frames=cell.frames,
            tile_ids=[t.tile_id for t in tiles],
            overwritten_fields=static,
            start_delay_s=config.harness.start_delay_s,
            max_late_fraction=config.harness.emitter_max_late_fraction,
            stop_at=stop_at,
            run=run,
        )


def run_cell(
    config: Config,
    cell: CellConfig,
    engine_kind: str,
    stop_at: float | None,
    model_hash: str | None,
) -> Path:
    cell_dir = config.harness.runs_dir / f"{cell.id}-{datetime.now():%Y%m%d-%H%M%S}"
    cell_dir.mkdir(parents=True)
    tiles, tile_source = load_tiles(config.tiles)
    mode_before = power_mode()

    manifest = {
        "cell_id": cell.id,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "harness_commit": harness_commit(),
        "engine_kind": engine_kind,
        "model": config.model.name,
        "quant": config.model.quant,
        "model_file_sha256": model_hash,
        "prompt": config.model.prompt,
        "max_tokens": config.model.max_tokens,
        "labels": list(config.model.labels),
        "tiles": tile_source,
        "n_tiles": len(tiles),
        "cadence_s": cell.cadence_s,
        "co_runner": cell.co_runner,
        "power_mode_requested": cell.power_mode,
        "power_mode_before": mode_before,
        "power_mode_during": mode_before,  # the controller cannot change it yet
        "warmup_frames": config.harness.warmup_frames,
        "frames_requested": cell.frames,
        "wall_start": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": device_snapshot(),
    }
    if engine_kind == "dummy":
        manifest["dummy"] = asdict(config.dummy)  # gate 1 replays these draws
    sampler = PowerSampler(config.harness.sampler_interval_s)
    manifest["sensors"] = sampler.sensors
    run = MailboxRun()
    try:
        with Tegrastats(cell_dir / "tegrastats.log"), GpuUtilization(cell_dir / "gpu_util.csv"):
            with sampler, make_engine(engine_kind, config, cell_dir) as engine:
                manifest["engine"] = {"name": engine.name, "version": engine.version}
                asyncio.run(measure(engine, config, cell, tiles, run, stop_at))
    except BaseException as exc:
        manifest["aborted"] = repr(exc)
        raise
    finally:
        manifest["wall_end"] = datetime.now().astimezone().isoformat(timespec="seconds")
        manifest["frames_emitted"] = run.frames_emitted
        manifest["frames_delivered"] = len(run.records)
        manifest["emitter_faults"] = [
            {"frame_id": f.frame_id, "late_s": round(f.late_s, 6)} for f in run.emitter_faults
        ]
        manifest["power_mode_after"] = power_mode()
        write_records(cell_dir / "records.jsonl", run.records)
        sampler.write_csv(cell_dir / "power.csv")
        (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if run.records:
        print(json.dumps(report(cell_dir), indent=2))
    print(f"cell directory: {cell_dir}")
    return cell_dir


def cell_estimate_s(config: Config, cell: CellConfig) -> float:
    return (config.harness.warmup_frames + cell.frames) * cell.cadence_s + ENGINE_START_ALLOWANCE_S


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--engine", choices=["llama", "dummy"], default="llama")
    parser.add_argument("--cells", nargs="+", metavar="ID", help="cell ids to run (default all)")
    parser.add_argument(
        "--max-minutes",
        type=float,
        help="stop starting cells, and cut the running one short, after this long",
    )
    args = parser.parse_args()

    config = config_module.load(args.config)
    cells = [config.cell(i) for i in args.cells] if args.cells else config.cells
    if not cells:
        parser.error(f"{args.config} defines no cells")

    stop_at = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
    model_hash = None
    if args.engine == "llama" and config.llama_server is not None:
        model_hash = file_sha256(config.llama_server.model)

    for cell in cells:
        if stop_at is not None and time.monotonic() + cell_estimate_s(config, cell) > stop_at:
            print(f"skipping {cell.id}: would exceed --max-minutes")
            continue
        run_cell(config, cell, args.engine, stop_at, model_hash)


if __name__ == "__main__":
    main()
