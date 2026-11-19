"""llama.cpp's llama-server: start it, and time one image-to-label request.

The server runs one slot with continuous batching off, so each request is
served alone and the latency we record is the engine's, not a batch's.

It runs either as a host process or, with a `Container`, inside its own Docker
container. A container mounts the same binary, CUDA libraries and model from
the host, so the only difference between the two is the container itself.
"""

import base64
import json
import os
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from harness.engines.base import NO_REPLY, Reply

__all__ = ["NO_REPLY", "Reply", "LlamaServer", "LlamaServerConfig", "encode_request", "send"]

LLAMA_CPP = Path("/data/projects/pulak/src/llama.cpp")
SERVER_BIN = LLAMA_CPP / "build/bin/llama-server"


@dataclass
class Container:
    """Run the server in a container of this image, under this name."""

    image: str
    name: str
    cuda_dir: Path  # the host's CUDA toolkit, mounted read-only

    # Directories mounted read-write at the same path, e.g. the MPS pipe directory.
    writable: tuple[Path, ...] = ()


@dataclass
class LlamaServerConfig:
    model: Path
    mmproj: Path
    port: int = 8081
    server_bin: Path = SERVER_BIN
    request_timeout_s: float = 300.0
    container: Container | None = None

    # Extra environment for the server, e.g. CUDA_MPS_PIPE_DIRECTORY.
    env: dict[str, str] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def docker_command(config: LlamaServerConfig, command: list[str]) -> list[str]:
    """`command` wrapped in `docker run`, with the build, CUDA and model mounted from the host.

    Host network (no port mapping in the request path), host IPC and the
    caller's user id (both needed for MPS clients to reach a host MPS daemon).
    """
    container = config.container
    build_dir = config.server_bin.parent.parent
    read_only = {build_dir, container.cuda_dir, config.model.parent, config.mmproj.parent}
    libraries = [config.server_bin.parent, container.cuda_dir / "targets/sbsa-linux/lib"]
    env = config.env | {"LD_LIBRARY_PATH": ":".join(str(p) for p in libraries)}
    args = ["docker", "run", "--rm", "--name", container.name, "--runtime", "nvidia"]
    args += ["--gpus", "all", "--network", "host", "--ipc", "host"]
    args += ["--user", f"{os.getuid()}:{os.getgid()}"]
    for path in sorted(read_only):
        args += ["-v", f"{path}:{path}:ro"]
    for path in container.writable:
        args += ["-v", f"{path}:{path}"]
    for key, value in sorted(env.items()):
        args += ["-e", f"{key}={value}"]
    return [*args, container.image, *command]


class LlamaServer:
    """Starts llama-server for a `with` block and stops it afterwards."""

    name = "llama.cpp"

    def __init__(self, config: LlamaServerConfig, logfile: Path):
        self.config = config
        self.logfile = logfile
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "LlamaServer":
        command = [
            str(self.config.server_bin),
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
        if self.config.container is not None:
            command = docker_command(self.config, command)
            env = None  # the container gets its environment through `docker run -e`
        else:
            env = os.environ | self.config.env
        self._log = self.logfile.open("w")
        self.process = subprocess.Popen(
            command, stdout=self._log, stderr=subprocess.STDOUT, env=env
        )
        try:
            self._wait_until_healthy()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self.config.container is not None:
            # Stopping the `docker run` client would leave the container running.
            subprocess.run(
                ["docker", "stop", "--time", "10", self.config.container.name], capture_output=True
            )
        else:
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
        """The llama.cpp git commit the binary was built from (setup/a1_llamacpp.sh writes it)."""
        commit_file = self.config.server_bin.parent.parent / "COMMIT"
        return commit_file.read_text().strip() if commit_file.exists() else "unknown"

    @property
    def version(self) -> str:
        return self.commit()

    def pids(self) -> set[int]:
        return {self.process.pid} if self.process else set()

    def prepare(self, image_png: bytes, prompt: str, max_tokens: int) -> bytes:
        return encode_request(image_png, prompt, max_tokens)

    @asynccontextmanager
    async def session(self):
        async with httpx.AsyncClient(timeout=self.config.request_timeout_s) as client:

            async def send_body(body: bytes) -> Reply:
                return await send(client, self.config.url, body)

            yield send_body


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
