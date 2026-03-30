GOLD_SOURCES = {
    "conversation": 0.95,
    "chatgpt": 0.9,
    "perplexity": 0.9,
    "email": 0.9,
}

SILVER_SOURCES = {
    "telegram": 0.75,
    "whatsapp": 0.7,
    "social_intel": 0.65,
}

BRONZE_SOURCES = {
    "consolidator": 0.5,
    "auto-extract": 0.4,
    "auto_extract": 0.4,
    "fact_extractor": 0.4,
    "web_page": 0.35,
}


def get_source_weight(source: str) -> float:
    normalized = (source or "").strip().lower()
    if normalized in GOLD_SOURCES:
        return GOLD_SOURCES[normalized]
    if normalized in SILVER_SOURCES:
        return SILVER_SOURCES[normalized]
    if normalized in BRONZE_SOURCES:
        return BRONZE_SOURCES[normalized]
    if "social_intel" in normalized:
        return SILVER_SOURCES["social_intel"]
    return 0.5
