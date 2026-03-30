#!/usr/bin/env python3
"""Embedding server on RTX 5090 (GPU1) - serves on port 8004
Replaces the GPU0 server to free PRO 6000 for inference"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # GPU index or UUID. Set via env var. MUST be before torch import.
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA_VISIBLE_DEVICES=1 set at top of file before torch import

app = FastAPI()

logger.info("Loading nomic-ai/nomic-embed-text-v1.5 on GPU1 (RTX 5090) with FP16...")
device = "cuda:0"  # cuda:0 because CUDA_VISIBLE_DEVICES=1 remaps it
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', device=device, trust_remote_code=True)
model.half()
logger.info("Model loaded on RTX 5090 (FP16)")

_ = model.encode(["warmup"] * 32, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
logger.info("Warmup complete")

class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]]
class EmbedRequestAlt(BaseModel):
    texts: Union[str, List[str]]

class EmbedRequestFlex(BaseModel):
    texts: Union[str, List[str], None] = None
    inputs: Union[str, List[str], None] = None

@app.post("/embed")
async def embed(req: EmbedRequestFlex):
    raw = req.texts or req.inputs
    if raw is None:
        return {"error": "provide 'texts' or 'inputs'"}
    texts = raw if isinstance(raw, list) else [raw]
    start = time.time()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64, normalize_embeddings=True)
    elapsed = time.time() - start
    logger.info(f"Embedded {len(texts)} texts in {elapsed:.3f}s ({len(texts)/elapsed:.0f} texts/s)")
    return {"embeddings": embeddings.tolist(), "model": "nomic-embed-text-v1.5", "device": "RTX 5090 (cuda:2)"}

@app.post("/v1/embeddings")
async def openai_embed(req: EmbedRequest):
    texts = req.inputs if isinstance(req.inputs, list) else [req.inputs]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64, normalize_embeddings=True)
    return {"object": "list", "data": [{"object": "embedding", "embedding": e.tolist(), "index": i} for i, e in enumerate(embeddings)], "model": "nomic-embed-text-v1.5"}

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": "RTX 5090", "model": "nomic-embed-text-v1.5"}

@app.get("/")
async def root():
    """Root handler — stops Uptime Kuma / healthcheck 404 spam."""
    return {"status": "ok", "service": "embed-gpu1", "model": "nomic-embed-text-v1.5"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003, log_level="info")
