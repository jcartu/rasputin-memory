import json
import os
import re

ENTITY_GRAPH_PATH = os.environ.get(
    "ENTITY_GRAPH_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "entity_graph.json"),
)

_SKIP_WORDS = {
    "The",
    "This",
    "That",
    "What",
    "When",
    "Where",
    "How",
    "Why",
    "Can",
    "Could",
    "Would",
    "Should",
    "Will",
    "Do",
    "Does",
    "Did",
    "Is",
    "Are",
    "Was",
    "Were",
    "Have",
    "Has",
    "Had",
    "I",
    "My",
    "We",
    "Our",
    "Yes",
    "No",
    "Ok",
    "Hey",
    "Hi",
    "Hello",
    "Set",
    "Make",
    "Run",
    "Get",
    "Let",
    "Put",
    "Give",
    "Take",
    "Check",
    "Look",
    "Find",
    "Show",
    "Tell",
    "Ask",
    "Try",
    "Write",
    "Read",
    "Send",
    "Add",
    "Use",
    "Also",
    "Just",
    "Sure",
    "Yeah",
    "Please",
    "Thanks",
    "Good",
    "Great",
    "Nice",
    "Cool",
    "Right",
    "Well",
    "But",
    "And",
    "Or",
    "So",
    "If",
    "For",
    "Not",
    "It",
    "A",
    "An",
    "To",
    "In",
    "On",
    "At",
    "By",
    "Up",
    "Of",
    "With",
    "About",
    "From",
    "Go",
    "Be",
    "All",
    "Now",
    "Very",
    "Much",
    "Really",
    "Think",
    "Know",
    "Want",
    "Need",
    "Like",
}

_SEMANTIC_EXPANSIONS: dict[str, str] = {}


def lookup_entity_graph(name: str, entity_graph_path: str | None = None) -> str:
    graph_path = entity_graph_path or ENTITY_GRAPH_PATH
    try:
        with open(graph_path) as f:
            graph = json.load(f)
    except Exception:
        return ""

    name_lower = name.lower()
    for person, data in graph.get("people", {}).items():
        if name_lower in person.lower():
            return f"{data.get('role', '')} {data.get('context', '')}".strip()
    for company, data in graph.get("companies", {}).items():
        if name_lower in company.lower():
            return f"{data.get('type', '')} {data.get('context', '')}".strip()
    return ""


def _topic_rephrase(query: str) -> str:
    stripped = re.sub(
        r"^(what|when|where|who|how|why|did|do|does|is|are|was|were|has|have|had|can|could|would|should|will|tell me about|remind me about|what happened with)\s+",
        "",
        query.lower(),
        flags=re.IGNORECASE,
    ).strip()
    return stripped


def _extract_entities(query: str) -> list[str]:
    entities: list[str] = []
    for word in query.split():
        clean = re.sub(r"[^\w]", "", word)
        if clean and clean[0].isupper() and clean not in _SKIP_WORDS and len(clean) > 1:
            entities.append(clean)
    seen = set()
    unique = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            unique.append(entity)
    return unique


def _append_unique(queries: list[str], candidate: str) -> None:
    candidate = candidate.strip()
    if candidate and candidate not in queries:
        queries.append(candidate)


def expand_queries(query: str, max_expansions: int = 5) -> list[str]:
    queries = [query]
    query_lower = query.lower()
    topic = _topic_rephrase(query)

    for entity in _extract_entities(query)[:4]:
        graph_context = lookup_entity_graph(entity)
        if graph_context:
            _append_unique(queries, f"{entity} {graph_context}")
        else:
            _append_unique(queries, entity)

    if topic != query_lower and len(topic) > 10:
        _append_unique(queries, topic)

    if any(word in query_lower for word in ["email", "wrote", "sent", "received", "inbox", "from", "to"]):
        _append_unique(queries, f"email {topic or query}")
    if any(
        word in query_lower for word in ["searched", "researched", "looked up", "looked into", "perplexity", "google"]
    ):
        _append_unique(queries, f"perplexity search {topic or query}")
    if any(word in query_lower for word in ["chatgpt", "conversation", "discussed", "talked about", "told"]):
        _append_unique(queries, f"chatgpt conversation {topic or query}")

    if any(word in query_lower for word in ["last week", "recently", "yesterday", "today", "this month"]):
        _append_unique(queries, f"recent {topic or query}")
    if any(word in query_lower for word in ["last year", "months ago", "a while back", "back in"]):
        _append_unique(queries, f"older {topic or query}")

    for keyword, expansion in _SEMANTIC_EXPANSIONS.items():
        if keyword in query_lower:
            _append_unique(queries, expansion)

    cap = max(1, max_expansions)
    return queries[:cap]
