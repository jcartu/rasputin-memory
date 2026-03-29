#!/usr/bin/env python3
"""
BGE Reranker Server — Cross-encoder neural reranking on GPU.

Exposes a FastAPI HTTP API for reranking search results.
Uses BAAI/bge-reranker-v2-m3 by default.

Usage:
    python3 reranker_server.py                # Start on port 8006
    python3 reranker_server.py --port 8007    # Custom port
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
PORT = int(os.environ.get("RERANKER_PORT", "8006"))

app = FastAPI(title="Reranker Server", version="1.0")

# Global model/tokenizer
model = None
tokenizer = None
device = None

class RerankRequest(BaseModel):
    query: str
    passages: List[str]

class RerankResponse(BaseModel):
    scores: List[float]

def load_model():
    """Load cross-encoder reranker model on GPU."""
    global model, tokenizer, device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    logger.info(f"✓ Model loaded on {device}")
    if torch.cuda.is_available():
        logger.info(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

@app.on_event("startup")
async def startup():
    load_model()

@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    """
    Rerank passages against a query.
    Returns normalized scores (higher = more relevant)
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not req.passages:
        return RerankResponse(scores=[])

    try:
        pairs = [[req.query, p] for p in req.passages]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        return RerankResponse(scores=scores)

    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    mem_gb = 0
    if torch.cuda.is_available():
        try:
            mem_gb = torch.cuda.memory_allocated(device) / 1e9
        except Exception:
            mem_gb = -1
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(device),
        "memory_allocated_gb": mem_gb,
    }

@app.get("/")
async def root():
    """Root handler for health checks."""
    return {"status": "ok", "service": "reranker", "model": MODEL_NAME}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
