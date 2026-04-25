from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator

from brain.schema import Entity, EntityUnitJoin, MemoryLink, model_validate_compat


class SqliteStore:
    def __init__(self, sqlite_path: str, *, entity_match_threshold: float = 0.45) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.entity_match_threshold = entity_match_threshold
        self._init_lock = threading.Lock()
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.sqlite_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        with self._init_lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memory_links (
                        id TEXT PRIMARY KEY,
                        bank_id TEXT NOT NULL,
                        from_unit_id TEXT NOT NULL,
                        to_unit_id TEXT NOT NULL,
                        link_type TEXT NOT NULL CHECK (
                            link_type IN (
                                'entity_cooccurrence',
                                'semantic',
                                'causes',
                                'caused_by',
                                'enables',
                                'prevents',
                                'temporal',
                                'proof'
                            )
                        ),
                        weight REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_links_forward
                    ON memory_links (bank_id, from_unit_id, link_type, weight DESC);

                    CREATE INDEX IF NOT EXISTS idx_links_reverse
                    ON memory_links (bank_id, to_unit_id, link_type);

                    CREATE TABLE IF NOT EXISTS entities (
                        id TEXT PRIMARY KEY,
                        bank_id TEXT NOT NULL,
                        canonical_name TEXT NOT NULL,
                        aliases TEXT,
                        entity_type TEXT,
                        first_mentioned_at TIMESTAMP,
                        last_mentioned_at TIMESTAMP,
                        mention_count INTEGER DEFAULT 0,
                        UNIQUE(bank_id, canonical_name)
                    );

                    CREATE INDEX IF NOT EXISTS idx_entities_bank_name
                    ON entities (bank_id, canonical_name);

                    CREATE TABLE IF NOT EXISTS entity_units (
                        entity_id TEXT NOT NULL,
                        unit_id TEXT NOT NULL,
                        role TEXT,
                        bank_id TEXT NOT NULL,
                        FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_entity_units_entity_unit
                    ON entity_units (entity_id, unit_id);

                    CREATE INDEX IF NOT EXISTS idx_entity_units_bank_unit
                    ON entity_units (bank_id, unit_id);

                    CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_units_unique
                    ON entity_units (entity_id, unit_id, bank_id, role);
                    """
                )

                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(canonical_name, aliases, tokenize='trigram')"
                )
                conn.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
                        INSERT INTO entities_fts(rowid, canonical_name, aliases)
                        VALUES (new.rowid, new.canonical_name, COALESCE(new.aliases, ''));
                    END;

                    CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
                        INSERT INTO entities_fts(entities_fts, rowid, canonical_name, aliases)
                        VALUES ('delete', old.rowid, old.canonical_name, COALESCE(old.aliases, ''));
                    END;

                    CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
                        INSERT INTO entities_fts(entities_fts, rowid, canonical_name, aliases)
                        VALUES ('delete', old.rowid, old.canonical_name, COALESCE(old.aliases, ''));
                        INSERT INTO entities_fts(rowid, canonical_name, aliases)
                        VALUES (new.rowid, new.canonical_name, COALESCE(new.aliases, ''));
                    END;
                    """
                )
                conn.execute(
                    """
                    INSERT INTO entities_fts(rowid, canonical_name, aliases)
                    SELECT rowid, canonical_name, COALESCE(aliases, '')
                    FROM entities
                    WHERE rowid NOT IN (SELECT rowid FROM entities_fts)
                    """
                )

    def add_link(self, link: MemoryLink) -> None:
        self.batch_add_links([link])

    def batch_add_links(self, links: list[MemoryLink]) -> None:
        if not links:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO memory_links(
                    id, bank_id, from_unit_id, to_unit_id, link_type, weight, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?)
                """,
                [
                    (
                        link.id,
                        link.bank_id,
                        link.from_unit_id,
                        link.to_unit_id,
                        link.link_type,
                        link.weight,
                        link.created_at.isoformat() if link.created_at else None,
                        json.dumps(link.metadata),
                    )
                    for link in links
                ],
            )

    def upsert_entity(self, entity: Entity) -> None:
        timestamp = datetime.now(timezone.utc)
        first_mentioned_at = entity.first_mentioned_at or entity.last_mentioned_at or timestamp
        last_mentioned_at = entity.last_mentioned_at or entity.first_mentioned_at or timestamp
        mention_increment = entity.mention_count or 1

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entities(
                    id, bank_id, canonical_name, aliases, entity_type,
                    first_mentioned_at, last_mentioned_at, mention_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(bank_id, canonical_name) DO UPDATE SET
                    aliases = excluded.aliases,
                    entity_type = excluded.entity_type,
                    first_mentioned_at = COALESCE(
                        MIN(entities.first_mentioned_at, excluded.first_mentioned_at),
                        entities.first_mentioned_at,
                        excluded.first_mentioned_at
                    ),
                    last_mentioned_at = COALESCE(
                        MAX(entities.last_mentioned_at, excluded.last_mentioned_at),
                        entities.last_mentioned_at,
                        excluded.last_mentioned_at
                    ),
                    mention_count = entities.mention_count + excluded.mention_count
                """,
                (
                    entity.id,
                    entity.bank_id,
                    entity.canonical_name,
                    json.dumps(entity.aliases),
                    entity.entity_type,
                    first_mentioned_at.isoformat() if first_mentioned_at else None,
                    last_mentioned_at.isoformat() if last_mentioned_at else None,
                    mention_increment,
                ),
            )

    def get_entity(self, canonical_name: str, bank_id: str) -> Entity | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, bank_id, canonical_name, aliases, entity_type,
                       first_mentioned_at, last_mentioned_at, mention_count
                FROM entities
                WHERE bank_id = ? AND canonical_name = ?
                """,
                (bank_id, canonical_name),
            ).fetchone()
        return self._entity_from_row(row)

    def resolve_entity(self, name: str, bank_id: str) -> Entity | None:
        search_term = name.strip()
        if not search_term:
            return None

        candidates = self._fts_candidates(search_term, bank_id)
        if not candidates:
            with self._connect() as conn:
                candidates = conn.execute(
                    """
                    SELECT id, bank_id, canonical_name, aliases, entity_type,
                           first_mentioned_at, last_mentioned_at, mention_count
                    FROM entities
                    WHERE bank_id = ? AND (
                        canonical_name = ? OR canonical_name LIKE ? OR aliases LIKE ?
                    )
                    ORDER BY mention_count DESC
                    LIMIT 10
                    """,
                    (bank_id, search_term, f"%{search_term}%", f"%{search_term}%"),
                ).fetchall()

        best_score = 0.0
        best_entity: Entity | None = None
        lowered_search = search_term.lower()
        for row in candidates:
            entity = self._entity_from_row(row)
            if entity is None:
                continue
            aliases = entity.aliases or []
            score = max(
                [SequenceMatcher(None, lowered_search, entity.canonical_name.lower()).ratio()]
                + [SequenceMatcher(None, lowered_search, alias.lower()).ratio() for alias in aliases]
            )
            if lowered_search == entity.canonical_name.lower() or lowered_search in {alias.lower() for alias in aliases}:
                score = 1.0
            if score > best_score:
                best_score = score
                best_entity = entity

        if best_entity is None or best_score < self.entity_match_threshold:
            return None
        return best_entity

    def add_entity_unit(self, join: EntityUnitJoin) -> None:
        self.batch_add_entity_units([join])

    def batch_add_entity_units(self, joins: list[EntityUnitJoin]) -> int:
        if not joins:
            return 0
        with self._connect() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO entity_units(entity_id, unit_id, role, bank_id)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        join.entity_id,
                        join.unit_id,
                        join.role or "other",
                        join.bank_id,
                    )
                    for join in joins
                ],
            )
            return cursor.rowcount if cursor.rowcount is not None else 0

    def expand_links(
        self,
        unit_ids: list[str],
        bank_id: str,
        link_types: list[str],
        *,
        limit_per_source: int = 10,
    ) -> list[MemoryLink]:
        if not unit_ids or not link_types:
            return []

        placeholders = ", ".join("?" for _ in link_types)
        results: list[MemoryLink] = []
        query = f"""
            SELECT id, bank_id, from_unit_id, to_unit_id, link_type, weight, created_at, metadata
            FROM memory_links
            WHERE bank_id = ? AND from_unit_id = ? AND link_type IN ({placeholders})
            ORDER BY weight DESC
            LIMIT ?
        """
        with self._connect() as conn:
            for unit_id in unit_ids:
                rows = conn.execute(query, (bank_id, unit_id, *link_types, limit_per_source)).fetchall()
                results.extend(self._link_from_row(row) for row in rows)
        return results

    def explain_expand_links_query(self, bank_id: str, unit_id: str, link_types: list[str]) -> list[str]:
        placeholders = ", ".join("?" for _ in link_types)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                EXPLAIN QUERY PLAN
                SELECT id
                FROM memory_links
                WHERE bank_id = ? AND from_unit_id = ? AND link_type IN ({placeholders})
                ORDER BY weight DESC
                LIMIT 10
                """,
                (bank_id, unit_id, *link_types),
            ).fetchall()
        return [str(row[3]) for row in rows]

    def count_rows(self, table: str, *, bank_id: str | None = None) -> int:
        with self._connect() as conn:
            if bank_id is None:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            else:
                row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE bank_id = ?", (bank_id,)).fetchone()
        return int(row[0]) if row else 0

    def delete_bank(self, bank_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_links WHERE bank_id = ?", (bank_id,))
            conn.execute("DELETE FROM entity_units WHERE bank_id = ?", (bank_id,))
            conn.execute("DELETE FROM entities WHERE bank_id = ?", (bank_id,))

    def _fts_candidates(self, search_term: str, bank_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.bank_id, e.canonical_name, e.aliases, e.entity_type,
                       e.first_mentioned_at, e.last_mentioned_at, e.mention_count
                FROM entities_fts f
                JOIN entities e ON e.rowid = f.rowid
                WHERE e.bank_id = ? AND entities_fts MATCH ?
                ORDER BY bm25(entities_fts), e.mention_count DESC
                LIMIT 20
                """,
                (bank_id, search_term),
            ).fetchall()
        return rows

    def _entity_from_row(self, row: sqlite3.Row | None) -> Entity | None:
        if row is None:
            return None
        return model_validate_compat(
            Entity,
            {
                "id": row["id"],
                "bank_id": row["bank_id"],
                "canonical_name": row["canonical_name"],
                "aliases": json.loads(row["aliases"] or "[]"),
                "entity_type": row["entity_type"],
                "first_mentioned_at": row["first_mentioned_at"],
                "last_mentioned_at": row["last_mentioned_at"],
                "mention_count": row["mention_count"],
            },
        )

    def _link_from_row(self, row: sqlite3.Row) -> MemoryLink:
        return model_validate_compat(
            MemoryLink,
            {
                "id": row["id"],
                "bank_id": row["bank_id"],
                "from_unit_id": row["from_unit_id"],
                "to_unit_id": row["to_unit_id"],
                "link_type": row["link_type"],
                "weight": row["weight"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"] or "{}"),
            },
        )
