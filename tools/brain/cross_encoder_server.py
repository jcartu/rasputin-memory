#!/usr/bin/env python3
"""Cross-encoder reranking server. Run on a GPU machine.

Usage:
    python cross_encoder_server.py [--port 9090] [--host 0.0.0.0] [--device cuda]

Exposes POST /rerank accepting {"pairs": [["query", "doc"], ...]}
Returns  {"scores": [float, ...]}
"""

import argparse
import logging
import math
import os
import time

from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_LENGTH = 512
BATCH_SIZE = 64

app = Flask(__name__)
_model = None
_device = "cuda"


def load_model(device: str):
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import CrossEncoder

    logger.info("Loading %s on %s ...", MODEL_NAME, device)
    t0 = time.monotonic()
    _model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH, device=device)
    _model.predict([["warmup query", "warmup document"]], batch_size=1)
    logger.info("Model ready in %.1fs", time.monotonic() - t0)
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": _device})


@app.route("/rerank", methods=["POST"])
def rerank():
    body = request.get_json(force=True, silent=True) or {}
    pairs = body.get("pairs", [])
    if not pairs:
        return jsonify({"scores": []})

    t0 = time.monotonic()
    try:
        model = load_model(_device)
        raw_scores = model.predict(pairs, batch_size=BATCH_SIZE)
        scores = [0.0 if math.isnan(float(s)) else round(float(s), 6) for s in raw_scores]
    except Exception as e:
        logger.error("predict failed: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

    elapsed = time.monotonic() - t0
    logger.info(
        "Reranked %d pairs in %.3fs (%.0f pairs/s)", len(pairs), elapsed, len(pairs) / elapsed if elapsed > 0 else 0
    )
    return jsonify({"scores": scores})


def main():
    global _device
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    _device = args.device

    load_model(_device)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
