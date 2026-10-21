"""A tiny stand-in for the Ollama HTTP API, so tests run without Ollama or GPUs.

Implements /api/tags, /api/embed and /api/chat (streaming) closely enough for
the official `ollama` Python client. Embeddings are hashed bag-of-words
vectors: crude, but texts that share words get similar vectors, which is all
the tests need.
"""

import hashlib
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 128


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeOllama:
    def __init__(self, models=("fake-embed", "fake-chat"), answer="PackageFlow uses PostgreSQL [1]. See also [2]."):
        self.models = set(models)
        self.answer = answer
        self.calls: list[tuple[str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/api/tags"):
                    self._json(200, {"models": [{"name": m, "model": m, "size": 1, "digest": "x"} for m in outer.models]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                outer.calls.append((self.path, req))
                model = req.get("model")
                if model not in outer.models:
                    self._json(404, {"error": f"model '{model}' not found"})
                    return
                if self.path.startswith("/api/embed"):
                    inp = req.get("input")
                    inp = [inp] if isinstance(inp, str) else inp
                    self._json(200, {"model": model, "embeddings": [fake_vector(t) for t in inp]})
                elif self.path.startswith("/api/chat"):
                    words = re.findall(r"\S+\s*", outer.answer)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    if req.get("stream", True):
                        for w in words:
                            line = {"model": model, "created_at": "2026-01-01T00:00:00Z",
                                    "message": {"role": "assistant", "content": w}, "done": False}
                            self.wfile.write((json.dumps(line) + "\n").encode())
                    final = {"model": model, "created_at": "2026-01-01T00:00:00Z",
                             "message": {"role": "assistant", "content": "" if req.get("stream", True) else outer.answer},
                             "done": True, "done_reason": "stop"}
                    self.wfile.write((json.dumps(final) + "\n").encode())
                else:
                    self._json(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
