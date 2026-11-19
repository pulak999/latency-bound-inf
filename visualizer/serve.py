"""Serve the viewer and the saved bundles on this machine.

    uv run python -m visualizer.serve [--profiles DIR] [--port 8765]
    then open http://127.0.0.1:8765

Routes:
    GET  /                          the viewer (visualizer/viewer/)
    GET  /api/bundles               every bundle under --profiles: meta, frame summary, notes
    GET  /api/bundles/<id>          that bundle's bundle.json
    PUT  /api/bundles/<id>/notes    replace that bundle's notes.md (plain text)

Listens on 127.0.0.1 only. Notes live next to the raw files, so a bundle and
what we concluded from it stay together.
"""

import argparse
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from visualizer.capture import PROFILES_DIR

VIEWER_DIR = Path(__file__).parent / "viewer"
BUNDLE_ID = re.compile(r"^[\w.-]+$")
MAX_NOTES_BYTES = 200_000


def summarize(bundle_dir: Path) -> dict:
    """What the library list shows for one bundle, without sending the whole bundle."""
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    frames = bundle["frames"][1:] or bundle["frames"]  # the first traced frame can be warmer
    notes = bundle_dir / "notes.md"

    def median(key: str) -> float | None:
        values = sorted(f[key] for f in frames if f.get(key) is not None)
        return values[len(values) // 2] if values else None

    return {
        "id": bundle_dir.name,
        "meta": bundle["meta"],
        "gpu": bundle["gpu"]["name"],
        "frames": len(bundle["frames"]),
        "client_latency_ms": median("client_latency_ms"),
        "gpu_span_ms": median("gpu_span_ms"),
        "idle_ms": median("idle_ms"),
        "has_ncu": bool(bundle.get("ncu")),
        "has_gpu_metrics": bool(bundle.get("gpu_metrics")),
        "notes": notes.read_text() if notes.exists() else "",
        "warnings": bundle.get("warnings", []),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, profiles: Path, **kwargs):
        self.profiles = profiles
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bundle_dir(self, bundle_id: str) -> Path | None:
        if not BUNDLE_ID.match(bundle_id):
            return None
        path = self.profiles / bundle_id
        return path if (path / "bundle.json").exists() else None

    def do_GET(self) -> None:
        if self.path == "/api/bundles":
            dirs = sorted(self.profiles.glob("*/bundle.json"), reverse=True)
            return self._json([summarize(p.parent) for p in dirs])
        if self.path.startswith("/api/bundles/"):
            bundle_dir = self._bundle_dir(self.path.removeprefix("/api/bundles/"))
            if bundle_dir is None:
                return self._json({"error": "no such bundle"}, 404)
            body = (bundle_dir / "bundle.json").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        return super().do_GET()

    def do_PUT(self) -> None:
        match = re.fullmatch(r"/api/bundles/([\w.-]+)/notes", self.path)
        bundle_dir = self._bundle_dir(match.group(1)) if match else None
        if bundle_dir is None:
            return self._json({"error": "no such bundle"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_NOTES_BYTES:
            return self._json({"error": "notes too long"}, 413)
        (bundle_dir / "notes.md").write_bytes(self.rfile.read(length))
        return self._json({"saved": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", type=Path, default=PROFILES_DIR)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, profiles=args.profiles))
    print(f"viewer: http://127.0.0.1:{args.port}  (bundles from {args.profiles})")
    server.serve_forever()


if __name__ == "__main__":
    main()
