"""A depth-1 slot between the emitter and the model.

A new frame replaces the one not yet taken. The emitter learns at once that it
overwrote a frame, and that frame counts as a miss with outcome `overwritten`.
This is what a real sensor pipeline does with a stale frame: it does not queue
it behind the next one.

Everything lives in one process on one asyncio loop, so put() and take() need
no locks and the mailbox adds no measurable delay. A ZeroMQ PUB/SUB pair with
CONFLATE=1 has the same semantics across processes; swap it in here if the
emitter's tick jitter ever shows the single process is not enough.
"""

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    frame_id: int
    tile_id: str
    t_obs: float  # monotonic seconds, when the emitter published it
    deadline: float  # t_obs plus the cadence; applied to the log afterwards


class Mailbox:
    def __init__(self) -> None:
        self._frame: Frame | None = None
        self._closed = False
        self._changed = asyncio.Event()

    def put(self, frame: Frame) -> Frame | None:
        """Publish a frame. Returns the frame it replaced, if one was still waiting."""
        replaced = self._frame
        self._frame = frame
        self._changed.set()
        return replaced

    def close(self) -> None:
        """No more frames will come. take() returns None once the slot is empty."""
        self._closed = True
        self._changed.set()

    async def take(self) -> Frame | None:
        """Wait for the newest frame and remove it from the slot."""
        while self._frame is None:
            if self._closed:
                return None
            self._changed.clear()
            await self._changed.wait()
        frame, self._frame = self._frame, None
        return frame
