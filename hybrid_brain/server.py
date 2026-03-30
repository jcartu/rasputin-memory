"""
hybrid_brain.server — Thin HTTP server wrapper around HybridSearch.

This module is intentionally minimal. All search logic lives in
:mod:`hybrid_brain.search`. The server just wires HTTP ↔ HybridSearch.

Usage (direct)::

    from hybrid_brain.server import serve
    serve(port=7777)

Or via CLI::

    python -m hybrid_brain.server --port 7777

The legacy entry point ``tools/hybrid_brain.py`` continues to work
unchanged and still starts the original full-featured server.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .search import HybridSearch

_search: HybridSearch | None = None


def _get_search() -> HybridSearch:
    global _search
    if _search is None:
        _search = HybridSearch()
    return _search


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # silence access log
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/health":
            self._json({"status": "ok", "version": "library"})
            return

        if parsed.path == "/search":
            query = params.get("q", [""])[0].strip()
            limit = int(params.get("limit", ["10"])[0])
            if not query:
                self._error(400, "Missing ?q=")
                return
            try:
                results = _get_search().query(query, limit=limit)
                self._json({"query": query, "results": results})
            except Exception as e:
                self._error(500, str(e))
            return

        self._error(404, "Not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._error(400, "Invalid JSON")
            return

        if self.path == "/commit":
            # Delegate to legacy commit_memory if available; otherwise stub
            text = data.get("text", "")
            if not text:
                self._error(400, "Missing 'text'")
                return
            try:
                from tools.hybrid_brain import commit_memory
                result = commit_memory(text, source=data.get("source", "api"))
                self._json(result)
            except ImportError:
                self._json({"status": "stub", "message": "commit not available in library-only mode"})
            return

        self._error(404, "Not found")

    def _json(self, data: object, status: int = 200) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, status=code)


def serve(port: int = 7777) -> None:
    """Start the lightweight library HTTP server on *port*."""
    server = HTTPServer(("", port), _Handler)
    print(f"[hybrid_brain.server] Listening on port {port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[hybrid_brain.server] Stopped.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Brain library server")
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()
    serve(args.port)
