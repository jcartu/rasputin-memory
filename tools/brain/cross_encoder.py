from __future__ import annotations

import json
import logging
import math
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_NAME = os.environ.get("CROSS_ENCODER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
_MAX_LENGTH = int(os.environ.get("CROSS_ENCODER_MAX_LENGTH", "8192"))
_BATCH_SIZE = int(os.environ.get("CROSS_ENCODER_BATCH_SIZE", "32"))
_REMOTE_URL = os.environ.get("CROSS_ENCODER_URL", "")
_REMOTE_TIMEOUT = int(os.environ.get("CROSS_ENCODER_TIMEOUT", "30"))
_RERANKER_INSTRUCTION = os.environ.get(
    "RERANKER_INSTRUCTION",
    "Given a query about a person's life, retrieve relevant memory snippets that answer the query",
)

_model = None
_remote_ok: bool | None = None


def _is_qwen_model() -> bool:
    lower = _MODEL_NAME.lower()
    return "qwen" in lower or "reranker" in lower


def _check_remote() -> bool:
    global _remote_ok
    if not _REMOTE_URL:
        _remote_ok = False
        return False
    if _remote_ok is not None:
        return _remote_ok
    try:
        url = _REMOTE_URL.rstrip("/").rsplit("/", 1)[0] + "/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        _remote_ok = data.get("status") == "ok"
        if _remote_ok:
            logger.info("Remote cross-encoder at %s (device=%s)", _REMOTE_URL, data.get("device"))
        return bool(_remote_ok)
    except Exception as e:
        logger.warning("Remote cross-encoder unavailable (%s), falling back to local", e)
        _remote_ok = False
        return False


def _predict_remote(pairs: list[list[str]]) -> list[float]:
    body = json.dumps({"pairs": pairs}).encode()
    req = urllib.request.Request(_REMOTE_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=_REMOTE_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    return data["scores"]


class _Qwen3LocalPredictor:
    def __init__(self, model_name: str, device: str, max_length: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.prefix = (
            '<|im_start|>system\\nJudge whether the Document meets the requirements based on the Query '
            'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\\n'
            '<|im_start|>user\\n'
        )
        self.suffix = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        chosen_device = device
        if chosen_device != "cpu":
            try:
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                    )
                    .to(chosen_device)
                    .eval()
                )
            except Exception:
                logger.warning("GPU load failed, falling back to CPU")
                chosen_device = "cpu"

        if chosen_device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            ).to("cpu").eval()

    def _format(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {_RERANKER_INSTRUCTION}\\n"
            f"<Query>: {query}\\n"
            f"<Document>: {doc}"
        )

    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> list[float]:
        if not pairs:
            return []
        all_scores: list[float] = []
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
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]
            stacked = self.torch.stack([false_logits, true_logits], dim=1)
            probs = self.torch.nn.functional.log_softmax(stacked, dim=1)
            scores = probs[:, 1].exp().tolist()
            all_scores.extend(float(score) for score in scores)
        return all_scores


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        device = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else "cuda"
        logger.info("Loading cross-encoder: %s (device=%s)", _MODEL_NAME, device)

        if _is_qwen_model():
            _model = _Qwen3LocalPredictor(_MODEL_NAME, device=device, max_length=_MAX_LENGTH)
        else:
            from sentence_transformers import CrossEncoder

            try:
                _model = CrossEncoder(_MODEL_NAME, max_length=min(_MAX_LENGTH, 512), device=device)
            except Exception:
                if device != "cpu":
                    logger.warning("GPU load failed, falling back to CPU")
                    _model = CrossEncoder(_MODEL_NAME, max_length=min(_MAX_LENGTH, 512), device="cpu")
                else:
                    raise

        logger.info("Cross-encoder loaded")
        return _model
    except ImportError:
        logger.warning("Cross-encoder dependencies not installed, cross-encoder disabled")
        return None
    except Exception as e:
        logger.error("Failed to load cross-encoder: %s", e)
        return None


def is_available() -> bool:
    if _check_remote():
        return True
    return _load_model() is not None


def _build_pairs(query: str, results: list[dict[str, Any]]) -> list[list[str]]:
    pairs = []
    for r in results:
        doc_text = (r.get("text") or "")[:1000]
        date = r.get("date") or ""
        if date and len(date) >= 10:
            doc_text = f"[Date: {date[:10]}] {doc_text}"
        source = r.get("source") or ""
        if source:
            doc_text = f"[Source: {source}] {doc_text}"
        pairs.append([query, doc_text])
    return pairs


def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    if not results:
        return results[:top_k] if top_k else results

    pairs = _build_pairs(query, results)
    raw_scores = None

    if _check_remote():
        try:
            raw_scores = _predict_remote(pairs)
        except Exception as e:
            logger.warning("Remote cross-encoder failed (%s), falling back to local", e)

    if raw_scores is None:
        model = _load_model()
        if model is None:
            return results[:top_k] if top_k else results
        try:
            raw_scores = model.predict(pairs, batch_size=_BATCH_SIZE)
        except Exception as e:
            logger.error("Cross-encoder predict failed: %s", e)
            return results[:top_k] if top_k else results

    for r, raw_score in zip(results, raw_scores):
        score = float(raw_score)
        if math.isnan(score):
            score = 0.0
        ce_score = score if 0.0 <= score <= 1.0 else 1.0 / (1.0 + math.exp(-score))
        r["ce_score"] = round(ce_score, 6)
        r["ce_raw"] = round(score, 4)

    results.sort(key=lambda x: x.get("ce_score", 0), reverse=True)
    return results[:top_k] if top_k else results


def rerank_with_recency(
    query: str,
    results: list[dict[str, Any]],
    top_k: int | None = None,
    recency_alpha: float = 0.2,
) -> list[dict[str, Any]]:
    results = rerank(query, results, top_k=None)
    now = datetime.now(timezone.utc)

    for r in results:
        ce = r.get("ce_score", 0.5)
        recency = 0.5
        date_str = r.get("date") or ""
        if date_str:
            try:
                if date_str.endswith("Z"):
                    date_str = date_str[:-1] + "+00:00"
                dt = datetime.fromisoformat(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                days_ago = max((now - dt).total_seconds() / 86400, 0)
                recency = max(0.1, min(1.0, 1.0 - (days_ago / 365)))
            except (ValueError, TypeError):
                pass

        recency_boost = 1.0 + recency_alpha * (recency - 0.5)
        r["recency"] = round(recency, 4)
        r["recency_boost"] = round(recency_boost, 4)
        r["final_score"] = round(ce * recency_boost, 6)

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return results[:top_k] if top_k else results
