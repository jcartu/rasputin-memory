#!/usr/bin/env python3
"""Cross-encoder / reranker server. Supports both classic cross-encoders
(ms-marco-MiniLM) and foundation-model rerankers (Qwen3-Reranker).

Usage:
    CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 python cross_encoder_server.py
    CROSS_ENCODER_MODEL=Qwen/Qwen3-Reranker-0.6B python cross_encoder_server.py
"""

import argparse
import logging
import math
import os
import time

import torch
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_LENGTH = int(os.environ.get("CROSS_ENCODER_MAX_LENGTH", "8192"))
BATCH_SIZE = int(os.environ.get("CROSS_ENCODER_BATCH_SIZE", "32"))
RERANKER_INSTRUCTION = os.environ.get(
    "RERANKER_INSTRUCTION",
    "Given a query about a person's life, retrieve relevant memory snippets that answer the query",
)

app = Flask(__name__)
_predictor = None
_device = "cuda"


class ClassicCEPredictor:
    def __init__(self, model_name, device, max_length):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=min(max_length, 512), device=device)
        self.model.predict([["warmup", "warmup"]], batch_size=1)

    def predict(self, pairs, batch_size=32):
        raw = self.model.predict(pairs, batch_size=batch_size)
        return [0.0 if math.isnan(float(s)) else round(float(s), 6) for s in raw]


class Qwen3RerankerPredictor:
    def __init__(self, model_name, device, max_length):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        try:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
                )
                .to(device)
                .eval()
            )
        except (ImportError, ValueError):
            logger.info("flash_attention_2 unavailable, using sdpa")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.max_length = max_length
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self._warmup()

    def _warmup(self):
        self.predict([["warmup query", "warmup document"]], batch_size=1)

    def _format(self, query, doc):
        return f"<Instruct>: {RERANKER_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def predict(self, pairs, batch_size=32):
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            texts = [self._format(q, d) for q, d in batch]
            inputs = self.tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            )
            for j, ids in enumerate(inputs["input_ids"]):
                inputs["input_ids"][j] = self.prefix_tokens + ids + self.suffix_tokens
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits[:, -1, :]
            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]
            stacked = torch.stack([false_logits, true_logits], dim=1)
            probs = torch.nn.functional.log_softmax(stacked, dim=1)
            scores = probs[:, 1].exp().tolist()
            all_scores.extend([round(s, 6) for s in scores])
        return all_scores


def load_predictor(device):
    global _predictor
    if _predictor is not None:
        return _predictor
    is_qwen = "qwen" in MODEL_NAME.lower() or "reranker" in MODEL_NAME.lower()
    logger.info("Loading %s (%s) on %s ...", MODEL_NAME, "qwen3" if is_qwen else "classic", device)
    t0 = time.monotonic()
    if is_qwen:
        _predictor = Qwen3RerankerPredictor(MODEL_NAME, device, MAX_LENGTH)
    else:
        _predictor = ClassicCEPredictor(MODEL_NAME, device, MAX_LENGTH)
    logger.info("Model ready in %.1fs", time.monotonic() - t0)
    return _predictor


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
        pred = load_predictor(_device)
        scores = pred.predict(pairs, batch_size=BATCH_SIZE)
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
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    _device = args.device
    load_predictor(_device)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
