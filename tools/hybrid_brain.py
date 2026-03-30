#!/usr/bin/env python3

import argparse
from typing import Optional

from brain import _state
from brain import amac
from brain import commit
from brain import embedding
from brain import entities
from brain import graph
from brain import proactive
from brain import scoring
from brain import search
from brain import server

for _state_name in _state.__all__:
    globals()[_state_name] = getattr(_state, _state_name)

qdrant = _state.qdrant
requests = _state.requests
redis = _state.redis

KNOWN_ENTITIES_CACHE_TTL_SECONDS = entities.KNOWN_ENTITIES_CACHE_TTL_SECONDS
KNOWN_PERSONS = entities.KNOWN_PERSONS
KNOWN_ORGS = entities.KNOWN_ORGS
KNOWN_PROJECTS = entities.KNOWN_PROJECTS
_load_known_entities = entities._load_known_entities
extract_entities_fast = entities.extract_entities_fast

is_reranker_available = embedding.is_reranker_available
get_embedding = embedding.get_embedding
check_duplicate = embedding.check_duplicate


def get_embedding_safe(
    text: str,
    default_action: str = "fail",
    prefix: str = _state.EMBED_PREFIX_QUERY,
) -> Optional[list[float]]:
    try:
        return get_embedding(text, prefix=prefix)
    except Exception as error:
        _state.logger.error("Embedding failed (%s): %s", default_action, error)
        if default_action == "empty":
            return [0.0] * 768
        if default_action == "skip":
            return None
        raise


neural_rerank = search._neural_rerank
_inc_metric = amac._inc_metric
AMAC_PROMPT_TEMPLATE = amac.AMAC_PROMPT_TEMPLATE
amac_score = amac.amac_score
amac_gate = amac.amac_gate

_parse_date = scoring._parse_date
apply_temporal_decay = scoring.apply_temporal_decay
apply_multifactor_scoring = scoring.apply_multifactor_scoring
get_source_weight = scoring.get_source_weight

_decode = graph._decode
_safe_graph_label = graph._safe_graph_label
write_to_graph = graph.write_to_graph
graph_search = graph.graph_search
enrich_with_graph = graph.enrich_with_graph

check_contradictions = commit.check_contradictions
commit_memory = commit.commit_memory
list_contradictions = commit.list_contradictions
apply_relevance_feedback = commit.apply_relevance_feedback

expand_queries = search.expand_queries
bm25_rerank = search.bm25_rerank
STOP_WORDS = search.STOP_WORDS
qdrant_search = search.qdrant_search
_update_access_tracking = search._update_access_tracking
hybrid_search = search.hybrid_search

proactive_surface = proactive.proactive_surface

SimpleRateLimiter = server.SimpleRateLimiter
HybridHandler = server.HybridHandler
ReusableHTTPServer = server.ReusableHTTPServer
serve = server.serve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Brain Search Server")
    parser.add_argument("--port", type=int, default=_state.SERVER_PORT)
    args = parser.parse_args()
    serve(args.port)
