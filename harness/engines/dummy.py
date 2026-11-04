"""A stand-in model that sleeps instead of inferring.

Used to validate the harness itself (DESIGN.md section 4, gate 1): the sleep
distribution is known, so the overwrite count and miss rate it should produce
can be predicted and compared with what the harness recorded.

Sleeps are drawn in the order requests are served, from a seeded generator,
so `harness.gate1` can regenerate the same draws and replay them through a
model of the mailbox.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np

from harness.engines.base import Reply


@dataclass(frozen=True)
class SleepDistribution:
    kind: str  # "uniform" or "constant"
    low_s: float
    high_s: float
    seed: int

    def draws(self, n: int) -> list[float]:
        """The first n sleep durations this distribution produces."""
        rng = np.random.default_rng(self.seed)
        if self.kind == "constant":
            return [self.low_s] * n
        if self.kind == "uniform":
            return rng.uniform(self.low_s, self.high_s, size=n).tolist()
        raise ValueError(f"unknown sleep distribution {self.kind!r}")


class DummyEngine:
    name = "dummy"
    version = "sleep"

    def __init__(self, distribution: SleepDistribution, label: str = "keep"):
        self.distribution = distribution
        self.label = label
        self._rng = np.random.default_rng(distribution.seed)
        self.sleeps: list[float] = []  # every draw, in service order, including warm-up

    def __enter__(self) -> "DummyEngine":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def pids(self) -> set[int]:
        return {os.getpid()}

    def prepare(self, image_png: bytes, prompt: str, max_tokens: int) -> object:
        return None  # nothing to encode; the image is never looked at

    def _next_sleep(self) -> float:
        if self.distribution.kind == "constant":
            sleep = self.distribution.low_s
        else:
            sleep = float(self._rng.uniform(self.distribution.low_s, self.distribution.high_s))
        self.sleeps.append(sleep)
        return sleep

    @asynccontextmanager
    async def session(self):
        async def send(body: object) -> Reply:
            sleep = self._next_sleep()
            await asyncio.sleep(sleep)
            t = time.monotonic()
            return Reply(
                t_first_token=t,
                t_last_token=t,
                text=self.label,
                n_prompt_tokens=0,
                n_output_tokens=1,
                engine_prompt_ms=sleep * 1000,
                engine_decode_ms=0.0,
            )

        yield send
