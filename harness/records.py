"""The per-request record and its JSONL file.

One line per request. A run directory holds `records.jsonl`; the report reads
only this file and the power samples, never the engine's own logs.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# What became of a request. `overwritten` is a frame the mailbox replaced before
# the model took it: it was never sent and has no timings after t_arrival.
OUTCOMES = ("ok", "overwritten", "error")


@dataclass
class RequestRecord:
    req_id: int

    # When the input arrived (the feeder's schedule) and when the request left.
    # Latency is measured from arrival, so time spent waiting behind an earlier
    # request counts against the deadline, as it would on the real system.
    # t_send is None for a frame that was overwritten before it was sent.
    t_arrival: float
    t_send: float | None

    # When the first and last output tokens reached the client. None on error.
    # For a robot policy the whole action chunk arrives at once: t_last_token is
    # when it was ready and t_first_token is None.
    t_first_token: float | None
    t_last_token: float | None

    # Arrival of the next input. None in throughput mode, which has no deadline.
    deadline: float | None

    n_prompt_tokens: int | None
    n_output_tokens: int | None
    output_text: str

    # The engine's own timing of the same request. The gap between this and the
    # client-side latency is transport, queueing and Python overhead.
    # LLM engines report prompt and decode time; other engines report named
    # stages in `engine_stages_ms` instead (e.g. preprocess, inference, postprocess).
    engine_prompt_ms: float | None
    engine_decode_ms: float | None

    # What was running and how the device was configured.
    engine: str
    model: str
    quant: str
    power_mode: str
    co_tenants: list[dict] = field(default_factory=list)

    error: str | None = None

    # Robot policies: actions in the returned chunk, and the engine's stage timings.
    # Both default to None so records written before these fields existed still load.
    n_actions: int | None = None
    engine_stages_ms: dict[str, float] | None = None

    # One of OUTCOMES, and which input this was (a tile file name, or a synthetic seed).
    outcome: str = "ok"
    tile_id: str | None = None

    def __post_init__(self) -> None:
        if self.error is not None and self.outcome == "ok":
            self.outcome = "error"
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, not {self.outcome!r}")

    @classmethod
    def overwritten(
        cls, req_id: int, t_arrival: float, deadline: float | None, tile_id: str | None, **static
    ) -> "RequestRecord":
        """The record of a frame the mailbox replaced before it was sent.

        `static` is the run's engine, model, quant and power_mode.
        """
        return cls(
            req_id=req_id,
            t_arrival=t_arrival,
            t_send=None,
            t_first_token=None,
            t_last_token=None,
            deadline=deadline,
            n_prompt_tokens=None,
            n_output_tokens=None,
            output_text="",
            engine_prompt_ms=None,
            engine_decode_ms=None,
            outcome="overwritten",
            tile_id=tile_id,
            **static,
        )

    @property
    def engine_total_ms(self) -> float | None:
        """Time the engine itself spent on this request, however it reports it."""
        if self.engine_stages_ms:
            return sum(self.engine_stages_ms.values())
        if self.engine_prompt_ms is not None and self.engine_decode_ms is not None:
            return self.engine_prompt_ms + self.engine_decode_ms
        return None

    @property
    def latency_s(self) -> float | None:
        """Arrival to last token, in seconds."""
        if self.t_last_token is None:
            return None
        return self.t_last_token - self.t_arrival

    @property
    def waiting_s(self) -> float | None:
        """Arrival to send: time spent in the mailbox behind an earlier request."""
        if self.t_send is None:
            return None
        return self.t_send - self.t_arrival

    @property
    def missed(self) -> bool | None:
        """True if the answer came after the deadline or never came."""
        if self.deadline is None:
            return None
        return self.t_last_token is None or self.t_last_token > self.deadline


def write_records(path: Path, records: list[RequestRecord]) -> None:
    with path.open("w") as f:
        for record in records:
            line = asdict(record) | {"latency_s": record.latency_s, "missed": record.missed}
            f.write(json.dumps(line) + "\n")


def read_records(path: Path) -> list[RequestRecord]:
    records = []
    with path.open() as f:
        for line in f:
            fields = json.loads(line)
            # Derived values are written for people reading the file; drop them on load.
            fields.pop("latency_s", None)
            fields.pop("missed", None)
            records.append(RequestRecord(**fields))
    return records
