from __future__ import annotations

from typing import Any, Iterator


def scroll_all(
    qdrant_client: Any,
    collection: str,
    batch_size: int = 100,
    scroll_filter: Any = None,
    with_payload: bool = True,
    with_vectors: bool = False,
    start_offset: Any = None,
) -> Iterator[Any]:
    offset = start_offset
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        if not points:
            break
        for point in points:
            yield point
        if offset is None:
            break
