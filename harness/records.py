"""The per-request record and its JSONL file.

One line per request. A run directory holds `records.jsonl`; the report reads
only this file and the power samples, never the engine's own logs.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RequestRecord:
    req_id: int

    # When the input arrived (the feeder's schedule) and when the request left.
    # Latency is measured from arrival, so time spent waiting behind an earlier
    # request counts against the deadline, as it would on the real system.
    t_arrival: float
    t_send: float

    # When the first and last output tokens reached the client. None on error.
    t_first_token: float | None
    t_last_token: float | None

    # Arrival of the next input. None in throughput mode, which has no deadline.
    deadline: float | None

    n_prompt_tokens: int | None
    n_output_tokens: int | None
    output_text: str

    # The engine's own timing of the same request. The gap between this and the
    # client-side latency is transport, queueing and Python overhead.
    engine_prompt_ms: float | None
    engine_decode_ms: float | None

    # What was running and how the device was configured.
    engine: str
    model: str
    quant: str
    power_mode: str
    co_tenants: list[dict] = field(default_factory=list)

    error: str | None = None

    @property
    def latency_s(self) -> float | None:
        """Arrival to last token, in seconds."""
        if self.t_last_token is None:
            return None
        return self.t_last_token - self.t_arrival

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
