"""SQLite FTS5 sidecar for BM25 keyword search.

Creates an in-memory SQLite database with FTS5 full-text index alongside
Qdrant. Provides keyword search that dense embeddings miss — e.g. finding
"Xenoblade 2" or "Nintendo Switch" when the query is "What console does
Nate own?"

Usage:
    from bm25_sidecar import BM25Index
    idx = BM25Index()
    idx.add("conv-26", point_id=123, text="Nate plays Xenoblade 2", chunk_type="fact")
    results = idx.search("conv-26", "console Nate", limit=10)
"""

import sqlite3


class BM25Index:
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                collection, point_id UNINDEXED, chunk_type UNINDEXED, text,
                tokenize='porter unicode61'
            )
        """)
        self.conn.commit()
        self._count = 0

    def add(self, collection, point_id, text, chunk_type="fact"):
        self.conn.execute(
            "INSERT INTO chunks(collection, point_id, chunk_type, text) VALUES (?, ?, ?, ?)",
            (collection, str(point_id), chunk_type, text),
        )
        self._count += 1
        if self._count % 500 == 0:
            self.conn.commit()

    def commit(self):
        self.conn.commit()

    def search(self, collection, query, limit=10):
        query_escaped = query.replace('"', '""')
        terms = query_escaped.split()
        if not terms:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in terms if len(t) > 2)
        if not fts_query:
            return []

        try:
            rows = self.conn.execute(
                """
                SELECT point_id, chunk_type, text, rank
                FROM chunks
                WHERE collection = ? AND chunks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (collection, fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            {
                "point_id": int(row[0]) if row[0].isdigit() else row[0],
                "chunk_type": row[1],
                "text": row[2],
                "score": -float(row[3]),
                "source": "bm25",
            }
            for row in rows
        ]

    def count(self, collection=None):
        if collection:
            return self.conn.execute("SELECT COUNT(*) FROM chunks WHERE collection = ?", (collection,)).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self):
        self.conn.close()
