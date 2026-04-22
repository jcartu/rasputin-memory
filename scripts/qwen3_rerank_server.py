#!/usr/bin/env python3
"""Qwen3-Reranker-0.6B FastAPI server for Rasputin Memory.

Protocol matches tools/brain/cross_encoder.py:
  POST /rerank  {"pairs": [[query, passage], ...]}  ->  {"scores": [f, ...]}
  GET  /health  ->  {"status": "ok", "device": "cuda:0", "model": "..."}

Deploys on CUDA_VISIBLE_DEVICES GPU (set via env before launch).
Default port 9091 (matches config/rasputin.toml expected URL).
"""
from __future__ import annotations

import logging
import os
from typing import List

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qwen3-rerank")

MODEL_PATH = os.environ.get("RERANK_MODEL_PATH", "/home/josh/models/Qwen3-Reranker-0.6B")
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "8192"))
PORT = int(os.environ.get("RERANK_PORT", "9091"))
HOST = os.environ.get("RERANK_HOST", "127.0.0.1")

app = FastAPI(title="Qwen3-Reranker", version="1.0")

model: CrossEncoder | None = None
device: str = "cpu"


class RerankRequest(BaseModel):
    pairs: List[List[str]]


class RerankResponse(BaseModel):
    scores: List[float]


@app.on_event("startup")
def _load() -> None:
    global model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s ...", MODEL_PATH, device)
    model = CrossEncoder(MODEL_PATH, max_length=MAX_LENGTH, device=device)
    if device == "cuda":
        logger.info("  memory allocated: %.2f GB", torch.cuda.memory_allocated() / 1e9)
    logger.info("Ready.")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if model is not None else "loading",
        "device": device,
        "model": MODEL_PATH,
    }


@app.get("/")
def root() -> dict:
    return {"service": "qwen3-rerank", "model": MODEL_PATH}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if model is None:
        return RerankResponse(scores=[])
    if not req.pairs:
        return RerankResponse(scores=[])
    scores = model.predict(req.pairs, batch_size=32, show_progress_bar=False)
    return RerankResponse(scores=[float(s) for s in scores])


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
