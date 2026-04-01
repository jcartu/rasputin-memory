from __future__ import annotations

SOURCE_IMPORTANCE: dict[str, float] = {
    "conversation": 0.95,
    "chatgpt": 0.90,
    "perplexity": 0.90,
    "email": 0.90,
    "telegram": 0.75,
    "whatsapp": 0.70,
    "social_intel": 0.65,
    "consolidator": 0.50,
    "auto-extract": 0.40,
    "auto_extract": 0.40,
    "fact_extractor": 0.40,
    "web_page": 0.35,
}


def get_source_weight(source: str) -> float:
    normalized = (source or "").strip().lower()
    if normalized in SOURCE_IMPORTANCE:
        return SOURCE_IMPORTANCE[normalized]
    if "social_intel" in normalized:
        return SOURCE_IMPORTANCE["social_intel"]
    return 0.5
