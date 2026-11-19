"""A background tenant's client: keeps one llama-server busy in a set pattern until stopped.

    python -m harness.background CONFIG --port P --load steady --out FILE [--tile-offset K]

Runs in its own process, one per background tenant, so its requests never share
an event loop with the measured tenant's feeder. Three load patterns, the first
two with the same mean rate (one request per `period_s`):

- `steady`: requests at random (Poisson) times, one per `period_s` on average.
- `bursty`: bursts of `burst_size` requests back to back, bursts at random
  times, one burst per `burst_size * period_s` on average.
- `saturating`: back to back, as fast as the server answers.

Arrivals are random, not periodic, on purpose: a periodic tenant at the same
period as the measured one locks to a fixed phase, and the cell then measures
that one phase instead of interference (the first smoke cell did exactly this:
every frame overlapped the background request by the same 500 ms).

A request is never sent while the previous one is still running (the server has
one slot). If a request overruns its tick, the next one goes out at once.
Each request is one JSON line in `--out`, written as it finishes, so a cell can
compute how busy each background tenant actually was. Stops on SIGTERM.
"""

import argparse
import asyncio
import json
import random
import signal
import time
from pathlib import Path

import httpx

from harness import config as config_module
from harness.engines.llama_server import encode_request, send
from harness.feeder import sleep_until
from harness.tiles import load_tiles

LOADS = ("steady", "bursty", "saturating")


def send_times(load: str, start: float, period_s: float, burst_size: int, seed: int):
    """When each request is due: an endless sequence of monotonic times (None = right away)."""
    if load not in LOADS:
        raise ValueError(f"load must be one of {LOADS}, not {load!r}")
    rng = random.Random(seed)
    t = start
    i = 0
    while True:
        if load == "saturating":
            yield None
        elif load == "steady":
            t += rng.expovariate(1 / period_s)
            yield t
        elif i % burst_size == 0:  # bursty: the first request of a burst waits for its time
            t += rng.expovariate(1 / (burst_size * period_s))
            yield t
        else:
            yield None
        i += 1


async def run(args: argparse.Namespace, stop: asyncio.Event) -> None:
    config = config_module.load(args.config)
    tiles, _ = load_tiles(config.tiles)
    tiles = tiles[args.tile_offset :] + tiles[: args.tile_offset]
    bodies = [encode_request(t.png, config.model.prompt, config.model.max_tokens) for t in tiles]
    url = f"http://127.0.0.1:{args.port}"
    times = send_times(args.load, time.monotonic(), args.period_s, args.burst_size, args.seed)
    async with httpx.AsyncClient(timeout=config.llama_server.request_timeout_s) as client:
        with args.out.open("w") as log:
            for i, due in enumerate(times):
                if due is not None:
                    await asyncio.wait(
                        [asyncio.create_task(sleep_until(due)), asyncio.create_task(stop.wait())],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                if stop.is_set():
                    return
                t_send = time.monotonic()
                try:
                    reply, error = await send(client, url, bodies[i % len(bodies)]), None
                    t_done, prompt_ms = reply.t_last_token, reply.engine_prompt_ms
                except httpx.HTTPError as exc:
                    t_done, prompt_ms, error = time.monotonic(), None, repr(exc)
                line = {
                    "i": i,
                    "t_due": due,
                    "t_send": t_send,
                    "t_done": t_done,
                    "engine_prompt_ms": prompt_ms,
                    "error": error,
                }
                log.write(json.dumps(line) + "\n")
                log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--load", choices=LOADS, required=True)
    parser.add_argument("--period-s", type=float, default=3.0)
    parser.add_argument("--burst-size", type=int, default=5)
    parser.add_argument("--tile-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="seed for the random arrival times")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    async def run_until_stopped() -> None:
        stop = asyncio.Event()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
        await run(args, stop)

    asyncio.run(run_until_stopped())


if __name__ == "__main__":
    main()
