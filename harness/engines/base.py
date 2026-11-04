"""What every Stack A engine offers the controller.

An engine turns an image and a prompt into a request body once, up front, and
then answers that body as many times as asked. The controller holds the engine
open for a whole cell with `with engine:` and sends through `engine.session()`.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Reply:
    t_first_token: float | None
    t_last_token: float | None
    text: str
    n_prompt_tokens: int | None
    n_output_tokens: int | None
    engine_prompt_ms: float | None
    engine_decode_ms: float | None


# What a failed request records: no tokens, no times.
NO_REPLY = Reply(None, None, "", None, None, None, None)

Send = Callable[[object], Awaitable[Reply]]


class Engine(Protocol):
    name: str  # "llama.cpp", "dummy", ...
    version: str  # build commit or package version

    def __enter__(self) -> "Engine": ...
    def __exit__(self, *exc) -> None: ...

    def pids(self) -> set[int]:
        """Processes this engine owns, so they are not counted as co-tenants."""
        ...

    def prepare(self, image_png: bytes, prompt: str, max_tokens: int) -> object:
        """Build the request body for one image. Called before timing starts."""
        ...

    def session(self) -> AbstractAsyncContextManager[Send]:
        """An async context whose value sends one prepared body and returns its Reply."""
        ...


AsyncSession = AsyncIterator[Send]
