"""llama.cpp's llama-server: start it, and time one image-to-label request.

The server runs one slot with continuous batching off, so each request is
served alone and the latency we record is the engine's, not a batch's.
"""

import base64
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

LLAMA_CPP = Path("/data/projects/pulak/src/llama.cpp")
SERVER_BIN = LLAMA_CPP / "build/bin/llama-server"


@dataclass
class LlamaServerConfig:
    model: Path
    mmproj: Path
    port: int = 8081

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class LlamaServer:
    """Starts llama-server for a `with` block and stops it afterwards."""

    def __init__(self, config: LlamaServerConfig, logfile: Path):
        self.config = config
        self.logfile = logfile
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "LlamaServer":
        command = [
            str(SERVER_BIN),
            "--model",
            str(self.config.model),
            "--mmproj",
            str(self.config.mmproj),
            "--n-gpu-layers",
            "99",
            "--port",
            str(self.config.port),
            "--parallel",
            "1",
            "--no-cont-batching",
        ]
        self._log = self.logfile.open("w")
        self.process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT)
        self._wait_until_healthy()
        return self

    def __exit__(self, *exc) -> None:
        self.process.terminate()
        self.process.wait()
        self._log.close()

    def _wait_until_healthy(self, timeout_s: float = 120) -> None:
        give_up_at = time.monotonic() + timeout_s
        while time.monotonic() < give_up_at:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.logfile}")
            try:
                if httpx.get(f"{self.config.url}/health", timeout=1).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"llama-server not healthy after {timeout_s} s; see {self.logfile}")

    def commit(self) -> str:
        """The llama.cpp git commit the binary was built from."""
        return (LLAMA_CPP / "build/COMMIT").read_text().strip()


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


def encode_request(image_png: bytes, prompt: str, n_output_tokens: int) -> bytes:
    """OpenAI-style chat request with one image, forced to exactly n_output_tokens tokens.

    Returns the serialized JSON body. Call this before timing starts: a full-size
    image is several megabytes of base64, and encoding it is not the engine's cost.
    """
    image_url = "data:image/png;base64," + base64.b64encode(image_png).decode()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": n_output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Fixed output length: keep generating past the end-of-sequence token,
        # so every request decodes the same number of tokens.
        "ignore_eos": True,
        # Don't reuse the previous request's prompt cache; every request pays full prefill.
        "cache_prompt": False,
    }
    return json.dumps(body).encode()


def parse_stream_line(line: str) -> dict | None:
    """Decode one server-sent-events line. Returns None for blank lines and the end marker."""
    if not line.startswith("data: "):
        return None
    payload = line.removeprefix("data: ")
    if payload == "[DONE]":
        return None
    return json.loads(payload)


async def send(client: httpx.AsyncClient, url: str, body: bytes) -> Reply:
    """Send one streaming request and time its first and last tokens on arrival."""
    t_first_token = None
    t_last_token = None
    text_parts = []
    usage = {}
    timings = {}

    async with client.stream(
        "POST",
        f"{url}/v1/chat/completions",
        content=body,
        headers={"Content-Type": "application/json"},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            chunk = parse_stream_line(line)
            if chunk is None:
                continue
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    t_last_token = time.monotonic()
                    if t_first_token is None:
                        t_first_token = t_last_token
                    text_parts.append(content)
            usage = chunk.get("usage", usage)
            timings = chunk.get("timings", timings)

    if t_last_token is None:  # no content came back; fall back to end of stream
        t_last_token = time.monotonic()

    return Reply(
        t_first_token=t_first_token,
        t_last_token=t_last_token,
        text="".join(text_parts),
        n_prompt_tokens=usage.get("prompt_tokens"),
        n_output_tokens=usage.get("completion_tokens"),
        engine_prompt_ms=timings.get("prompt_ms"),
        engine_decode_ms=timings.get("predicted_ms"),
    )
