from __future__ import annotations

import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from brain import _state
from brain import amac
from brain import commit
from brain import embedding
from brain import graph
from brain import proactive
from brain import search


class SimpleRateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.history: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str = "default") -> bool:
        now = time.time()
        with self._lock:
            entries = self.history.get(key, [])
            entries = [timestamp for timestamp in entries if now - timestamp < 60]
            if len(entries) >= self.calls_per_minute:
                self.history[key] = entries
                return False
            entries.append(now)
            self.history[key] = entries
            # Evict stale keys to prevent unbounded memory growth
            if len(self.history) > 10000:
                cutoff = now - 60
                self.history = {k: v for k, v in self.history.items() if v and v[-1] > cutoff}
            return True


_rate_limiters = {
    "/commit": SimpleRateLimiter(calls_per_minute=30),
    "/search": SimpleRateLimiter(calls_per_minute=120),
}


class HybridHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        _state.logger.debug("%s %s", self.client_address[0], format % args)

    def _check_auth(self) -> bool:
        if not _state.MEMORY_API_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {_state.MEMORY_API_TOKEN}":
            return True
        self._send_json({"error": "Unauthorized"}, 401)
        return False

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _rate_limit_key(self) -> str:
        client_ip = self.client_address[0] if getattr(self, "client_address", None) else "unknown"
        return f"{self.path.split('?')[0]}:{client_ip}"

    def _enforce_rate_limit(self, endpoint: str) -> bool:
        limiter = _rate_limiters.get(endpoint)
        if not limiter:
            return True
        if limiter.allow(self._rate_limit_key()):
            return True
        self._send_json({"error": "Too Many Requests"}, 429)
        return False

    def do_GET(self) -> None:
        try:
            self._handle_get()
        except Exception as error:
            _state.logger.error("Unhandled GET error: %s", error)
            try:
                self._send_json({"error": "Internal server error"}, 500)
            except Exception:
                pass

    def _handle_get(self) -> None:
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/search":
            if not self._enforce_rate_limit("/search"):
                return
            query = params.get("q", [""])[0]
            limit = min(max(int(params.get("limit", ["10"])[0]), 1), 100)
            source = params.get("source", [None])[0]
            expand = params.get("expand", ["true"])[0].lower() != "false"

            if not query:
                self._send_json({"error": "Missing q parameter"}, 400)
                return

            result = search.hybrid_search(query, limit=limit, source_filter=source, expand=expand)
            self._send_json(result)

        elif parsed.path == "/graph":
            query = params.get("q", [""])[0]
            limit = min(max(int(params.get("limit", ["10"])[0]), 1), 100)
            hops = min(max(int(params.get("hops", ["2"])[0]), 1), 4)

            if not query:
                self._send_json({"error": "Missing q parameter"}, 400)
                return

            results = graph.graph_search(query, hops=hops, limit=limit)
            self._send_json({"query": query, "results": results})

        elif parsed.path == "/stats":
            try:
                info = _state.qdrant.get_collection(_state.COLLECTION)
                qdrant_count = info.points_count
            except Exception:
                qdrant_count = -1

            try:
                redis_client = _state.get_redis()
                node_count = redis_client.execute_command(
                    "GRAPH.QUERY", _state.GRAPH_NAME, "MATCH (n) RETURN count(n)"
                )[1][0][0]
                edge_count = redis_client.execute_command(
                    "GRAPH.QUERY", _state.GRAPH_NAME, "MATCH ()-[e]->() RETURN count(e)"
                )[1][0][0]
            except Exception:
                node_count = -1
                edge_count = -1

            self._send_json(
                {
                    "qdrant": {"collection": _state.COLLECTION, "points": qdrant_count},
                    "graph": {"nodes": node_count, "edges": edge_count},
                    "status": "ok",
                }
            )

        elif parsed.path == "/health":
            health = {
                "status": "ok",
                "engine": "hybrid-brain",
                "version": "0.3.0",
                "components": {
                    "qdrant": "unknown",
                    "falkordb": "unknown",
                    "ollama_embed": "unknown",
                    "reranker": "up" if embedding.is_reranker_available() else "down",
                    "bm25": "up" if _state.BM25_AVAILABLE else "down",
                },
            }
            try:
                info = _state.qdrant.get_collection(_state.COLLECTION)
                health["components"]["qdrant"] = f"up ({info.points_count} pts)"
            except Exception:
                health["components"]["qdrant"] = "down"
                health["status"] = "degraded"
            try:
                redis_client = _state.get_redis()
                redis_client.ping()
                health["components"]["falkordb"] = "up"
            except Exception:
                health["components"]["falkordb"] = "down"
                health["status"] = "degraded"
            try:
                test_resp = _state.requests.post(
                    _state.EMBED_URL,
                    json={"model": _state.EMBED_MODEL, "input": "health check"},
                    timeout=5,
                )
                if test_resp.status_code == 200 and "embeddings" in test_resp.json():
                    health["components"]["ollama_embed"] = "up"
                else:
                    health["components"]["ollama_embed"] = "error"
                    health["status"] = "degraded"
            except Exception:
                health["components"]["ollama_embed"] = "down"
                health["status"] = "degraded"
            self._send_json(health)

        elif parsed.path == "/amac/metrics":
            with _state._amac_metrics_lock:
                snapshot = dict(_state._amac_metrics)
            total = snapshot["accepted"] + snapshot["rejected"]
            avg_score = (
                round(snapshot["score_sum"] / snapshot["score_count"], 2) if snapshot["score_count"] > 0 else None
            )
            rejection_rate = round(snapshot["rejected"] / total * 100, 1) if total > 0 else 0
            self._send_json(
                {
                    "accepted": snapshot["accepted"],
                    "rejected": snapshot["rejected"],
                    "bypassed": snapshot["bypassed"],
                    "timeout_accepts": snapshot["timeout_accepts"],
                    "total": total,
                    "avg_composite_score": avg_score,
                    "rejection_rate_pct": rejection_rate,
                    "threshold": _state.AMAC_THRESHOLD,
                }
            )

        elif parsed.path == "/contradictions":
            limit = min(max(int(params.get("limit", ["50"])[0]), 1), 500)
            rows = commit.list_contradictions(limit=limit)
            self._send_json({"count": len(rows), "results": rows})

        else:
            self._send_json({"error": f"Unknown path: {parsed.path}"}, 404)

    def do_POST(self) -> None:
        try:
            self._handle_post()
        except Exception as error:
            _state.logger.error("Unhandled POST error: %s", error)
            try:
                self._send_json({"error": "Internal server error"}, 500)
            except Exception:
                pass

    def _handle_post(self) -> None:
        if not self._check_auth():
            return
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > _state.MAX_BODY_SIZE:
            self._send_json({"error": "Request body too large (max 1MB)"}, 413)
            return
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        parsed = urlparse(self.path)

        if parsed.path == "/search":
            if not self._enforce_rate_limit("/search"):
                return
            query = data.get("q", data.get("query", ""))
            limit = min(max(int(data.get("limit", 10)), 1), 100)
            source = data.get("source", None)
            expand_raw = data.get("expand", True)
            if isinstance(expand_raw, str):
                expand = expand_raw.lower() != "false"
            else:
                expand = bool(expand_raw)

            if not query:
                self._send_json({"error": "Missing query"}, 400)
                return

            result = search.hybrid_search(query, limit=limit, source_filter=source, expand=expand)
            self._send_json(result)

        elif parsed.path == "/proactive":
            messages = data.get("messages", [])
            context = data.get("context", "")
            max_results = data.get("max_results", 3)
            selected_entities = data.get("entities", [])
            min_days_since_access = int(data.get("min_days_since_access", 7))

            if context and not messages:
                messages = [context]

            if not messages:
                self._send_json({"error": "Missing messages or context"}, 400)
                return

            suggestions = proactive.proactive_surface(
                messages,
                max_results=max_results,
                active_entities=selected_entities,
                min_days_since_access=max(0, min_days_since_access),
            )
            self._send_json(
                {
                    "suggestions": suggestions,
                    "count": len(suggestions),
                }
            )

        elif parsed.path == "/commit":
            if not self._enforce_rate_limit("/commit"):
                return
            text = data.get("text", "")
            source = data.get("source", "conversation")
            importance = data.get("importance", 60)
            metadata = data.get("metadata", None)
            force = bool(data.get("force", False))

            if not text:
                self._send_json({"error": "Missing text"}, 400)
                return

            if len(text) < 20:
                self._send_json({"error": "Text too short (minimum 20 characters)"}, 400)
                return

            if len(text) > 8000:
                self._send_json({"error": "Text too long (maximum 8000 characters)"}, 400)
                return

            try:
                importance = int(importance)
            except (ValueError, TypeError):
                importance = 60
            importance = max(0, min(100, importance))

            allowed, reason, scores = amac.amac_gate(text, source=source, force=force)
            if not allowed:
                self._send_json(
                    {
                        "ok": False,
                        "rejected": True,
                        "reason": "amac_below_threshold",
                        "scores": scores,
                        "threshold": _state.AMAC_THRESHOLD,
                    },
                    200,
                )
                return

            if scores and "composite" in scores:
                amac_composite = float(scores.get("composite", 0))
                importance = int(0.4 * importance + 0.6 * amac_composite * 10)

            result = commit.commit_memory(text, source=source, importance=importance, metadata=metadata)
            if scores:
                result["amac"] = {"reason": reason, "scores": scores}
            status = 200 if result.get("ok") else 500
            self._send_json(result, status)

        elif parsed.path == "/feedback":
            point_id = data.get("point_id")
            helpful = data.get("helpful", True)
            if isinstance(helpful, str):
                helpful = helpful.lower() not in ("false", "0", "no")
            else:
                helpful = bool(helpful)

            if point_id is None:
                self._send_json({"error": "Missing point_id"}, 400)
                return

            result = commit.apply_relevance_feedback(point_id=point_id, helpful=helpful)
            status = 200 if result.get("ok") else 404
            self._send_json(result, status)

        else:
            self._send_json({"error": f"Unknown path: {parsed.path}"}, 404)


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True


def serve(port: int = _state.SERVER_PORT) -> None:
    server = ReusableHTTPServer(("127.0.0.1", port), HybridHandler)

    def shutdown_handler(signum: int, _frame: Any) -> None:
        _state.logger.info("Received signal %s, shutting down gracefully", signum)
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    _state.logger.info("Serving on http://127.0.0.1:%s", port)
    _state.logger.info(
        "Qdrant: %s (%s pts)",
        _state.COLLECTION,
        _state.qdrant.get_collection(_state.COLLECTION).points_count,
    )
    try:
        redis_client = _state.get_redis()
        node_count = redis_client.execute_command("GRAPH.QUERY", _state.GRAPH_NAME, "MATCH (n) RETURN count(n)")[1][0][
            0
        ]
        edge_count = redis_client.execute_command("GRAPH.QUERY", _state.GRAPH_NAME, "MATCH ()-[e]->() RETURN count(e)")[
            1
        ][0][0]
        _state.logger.info("FalkorDB: %s nodes, %s edges", node_count, edge_count)
    except Exception:
        _state.logger.warning("FalkorDB unavailable (graph search disabled)")
    if not _state.MEMORY_API_TOKEN:
        _state.logger.warning("⚠️  No MEMORY_API_TOKEN set — API authentication is DISABLED")
    server.serve_forever()
