#!/usr/bin/env python3
"""
Migrate Qdrant second brain → FalkorDB graph layer.
Uses local Qwen 72B (Ollama) for entity extraction. $0 API cost.

Usage:
    python3 migrate_to_graph.py                  # Start/resume migration
    python3 migrate_to_graph.py --batch 5        # 5 chunks per NER prompt
    python3 migrate_to_graph.py --dry-run        # Preview without writing
    python3 migrate_to_graph.py --reset          # Reset progress, start over
    python3 migrate_to_graph.py --status         # Show migration progress
"""

import json
import os
import sys
import time
import argparse
import hashlib
import requests
from pathlib import Path
from datetime import datetime

from falkordb import FalkorDB

# Config
FALKOR_HOST = "localhost"
FALKOR_PORT = 6380
GRAPH_NAME = "brain"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "second_brain"
OLLAMA_URL = "http://${OLLAMA_URL:-localhost:11434}/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"  # 4x faster than 72B for NER, comparable quality

STATE_FILE = Path(__file__).parent / "migration_state.json"
LOG_FILE = Path(__file__).parent / "migration.log"

NER_PROMPT = """Extract named entities from the following text. Return ONLY valid JSON, no explanation.

Categories:
- persons: people's names (full names preferred)
- organizations: companies, regulators, brands, teams
- projects: software projects, initiatives, campaigns  
- topics: key themes (2-5 words each, lowercase)
- locations: cities, countries, regions

Text:
{text}

JSON:"""

NER_BATCH_PROMPT = """Extract named entities from each text chunk below. Return a JSON array with one object per chunk. Each object has: persons, organizations, projects, topics, locations.

{chunks}

Return ONLY a valid JSON array, no explanation:"""


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"offset": None, "processed": 0, "errors": 0, "started": datetime.now().isoformat()}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_graph():
    db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    return db.select_graph(GRAPH_NAME)


def scroll_qdrant(offset=None, limit=50):
    """Scroll through Qdrant collection."""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}/points/scroll"
    payload = {"limit": limit, "with_payload": True, "with_vector": False}
    if offset is not None:
        payload["offset"] = offset
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()["result"]
    return data["points"], data.get("next_page_offset")


def extract_entities_single(text, timeout=120):
    """Extract entities from a single chunk via Ollama."""
    prompt = NER_PROMPT.format(text=text[:2000])  # Cap input size
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512}
    }, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json()["response"].strip()
    # Try to parse JSON from response
    try:
        # Handle markdown-wrapped JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
    return None


def extract_entities_batch(texts, timeout=180):
    """Extract entities from multiple chunks in one prompt."""
    chunks_text = "\n\n".join(
        f"--- Chunk {i+1} ---\n{t[:1500]}" for i, t in enumerate(texts)
    )
    prompt = NER_BATCH_PROMPT.format(chunks=chunks_text)
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048}
    }, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json()["response"].strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
    return None


def escape_cypher(s):
    """Escape string for Cypher query."""
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')


def ingest_to_graph(graph, point_id, payload, entities):
    """Create/merge nodes and relationships in FalkorDB."""
    text = payload.get("text", "")[:500]
    source = payload.get("source", "unknown")
    date = payload.get("date", payload.get("timestamp", ""))
    if isinstance(date, (int, float)):
        date = datetime.fromtimestamp(date).strftime("%Y-%m-%d")
    elif isinstance(date, str) and len(date) > 10:
        date = date[:10]

    # Create Memory node
    mem_id = str(point_id)
    graph.query(f"""
        MERGE (m:Memory {{id: '{escape_cypher(mem_id)}'}})
        SET m.text = '{escape_cypher(text)}',
            m.source = '{escape_cypher(source)}',
            m.date = '{escape_cypher(date)}'
    """)

    # Process each entity type
    for person in (entities.get("persons") or [])[:10]:
        if not person or len(person) < 2:
            continue
        name = escape_cypher(person.strip())
        graph.query(f"""
            MERGE (p:Person {{name: '{name}'}})
            MERGE (m:Memory {{id: '{escape_cypher(mem_id)}'}})
            MERGE (m)-[:MENTIONS]->(p)
        """)

    for org in (entities.get("organizations") or [])[:10]:
        if not org or len(org) < 2:
            continue
        name = escape_cypher(org.strip())
        graph.query(f"""
            MERGE (o:Organization {{name: '{name}'}})
            MERGE (m:Memory {{id: '{escape_cypher(mem_id)}'}})
            MERGE (m)-[:MENTIONS]->(o)
        """)

    for proj in (entities.get("projects") or [])[:10]:
        if not proj or len(proj) < 2:
            continue
        name = escape_cypher(proj.strip())
        graph.query(f"""
            MERGE (pr:Project {{name: '{name}'}})
            MERGE (m:Memory {{id: '{escape_cypher(mem_id)}'}})
            MERGE (m)-[:MENTIONS]->(pr)
        """)

    for topic in (entities.get("topics") or [])[:10]:
        if not topic or len(topic) < 2:
            continue
        name = escape_cypher(topic.strip().lower())
        graph.query(f"""
            MERGE (t:Topic {{name: '{name}'}})
            MERGE (m:Memory {{id: '{escape_cypher(mem_id)}'}})
            MERGE (m)-[:ABOUT]->(t)
        """)

    for loc in (entities.get("locations") or [])[:5]:
        if not loc or len(loc) < 2:
            continue
        name = escape_cypher(loc.strip())
        graph.query(f"""
            MERGE (l:Location {{name: '{name}'}})
            MERGE (m:Memory {{id: '{escape_cypher(mem_id)}'}})
            MERGE (m)-[:MENTIONS]->(l)
        """)


def show_status():
    state = load_state()
    print(f"\n📊 Migration Status")
    print(f"  Processed: {state.get('processed', 0):,}")
    print(f"  Errors: {state.get('errors', 0)}")
    print(f"  Started: {state.get('started', 'N/A')}")
    print(f"  Last offset: {state.get('offset', 'None')}")

    # Get total from Qdrant
    try:
        resp = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}", timeout=5)
        total = resp.json()["result"]["points_count"]
        pct = (state.get("processed", 0) / total * 100) if total else 0
        print(f"  Total in Qdrant: {total:,}")
        print(f"  Progress: {pct:.1f}%")
        if state.get("processed", 0) > 0:
            # Estimate remaining time
            elapsed = state.get("_elapsed_seconds", 0)
            if elapsed > 0:
                rate = state["processed"] / elapsed
                remaining = (total - state["processed"]) / rate
                print(f"  Est. remaining: {remaining/3600:.1f} hours")
    except:
        pass

    # Graph stats
    try:
        from schema import stats
        stats()
    except:
        pass


def run_migration(batch_size=1, dry_run=False):
    state = load_state()
    offset = state.get("offset")
    processed = state.get("processed", 0)
    errors = state.get("errors", 0)
    start_time = time.time()

    if not dry_run:
        graph = get_graph()
    else:
        graph = None

    # Get total count
    try:
        resp = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}", timeout=5)
        total = resp.json()["result"]["points_count"]
        log(f"Total memories in Qdrant: {total:,}")
    except:
        total = 0
        log("Could not get total count from Qdrant")

    log(f"Starting migration (batch={batch_size}, dry_run={dry_run})")
    log(f"Resuming from offset={offset}, already processed={processed:,}")

    consecutive_errors = 0
    batch_texts = []
    batch_points = []

    while True:
        try:
            points, next_offset = scroll_qdrant(offset=offset, limit=50)
        except Exception as e:
            log(f"❌ Qdrant scroll error: {e}")
            time.sleep(5)
            continue

        if not points:
            log("✅ Migration complete — no more points to process")
            break

        for point in points:
            text = point.get("payload", {}).get("text", "")
            if not text or len(text.strip()) < 50:
                processed += 1
                continue

            if batch_size > 1:
                batch_texts.append(text)
                batch_points.append(point)

                if len(batch_texts) >= batch_size:
                    try:
                        results = extract_entities_batch(batch_texts)
                        if results:
                            for i, (pt, ent) in enumerate(zip(batch_points, results)):
                                if ent and not dry_run:
                                    ingest_to_graph(graph, pt["id"], pt.get("payload", {}), ent)
                                processed += 1
                            consecutive_errors = 0
                        else:
                            # Fallback to single
                            for pt, txt in zip(batch_points, batch_texts):
                                try:
                                    ent = extract_entities_single(txt)
                                    if ent and not dry_run:
                                        ingest_to_graph(graph, pt["id"], pt.get("payload", {}), ent)
                                    processed += 1
                                except:
                                    errors += 1
                                    processed += 1
                    except Exception as e:
                        log(f"❌ Batch NER error: {e}")
                        errors += len(batch_texts)
                        processed += len(batch_texts)
                        consecutive_errors += 1

                    batch_texts = []
                    batch_points = []
            else:
                try:
                    entities = extract_entities_single(text)
                    if entities and not dry_run:
                        ingest_to_graph(graph, point["id"], point.get("payload", {}), entities)
                    elif dry_run and entities:
                        log(f"  [DRY] {str(point['id'])[:8]}... → {json.dumps(entities)[:200]}")
                    processed += 1
                    consecutive_errors = 0
                except Exception as e:
                    errors += 1
                    processed += 1
                    consecutive_errors += 1
                    log(f"❌ NER error on {point['id']}: {e}")

            # Log progress
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1) if processed > state.get("processed", 0) else 0
                pct = (processed / total * 100) if total else 0
                eta_h = ((total - processed) / rate / 3600) if rate > 0 else 0
                log(f"📊 Progress: {processed:,}/{total:,} ({pct:.1f}%) | {rate:.1f}/s | ETA: {eta_h:.1f}h | Errors: {errors}")

            if consecutive_errors >= 10:
                log("⚠️ 10 consecutive errors — pausing 60s")
                time.sleep(60)
                consecutive_errors = 0

        # Save state after each Qdrant page
        offset = next_offset
        state.update({
            "offset": offset,
            "processed": processed,
            "errors": errors,
            "_elapsed_seconds": time.time() - start_time
        })
        save_state(state)

        if offset is None:
            log("✅ Migration complete — reached end of Qdrant collection")
            break

    # Final save
    state["completed"] = datetime.now().isoformat()
    save_state(state)
    elapsed = time.time() - start_time
    log(f"\n🏁 Done. Processed: {processed:,} | Errors: {errors} | Time: {elapsed/3600:.1f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Qdrant → FalkorDB graph")
    parser.add_argument("--batch", type=int, default=1, help="Chunks per NER prompt (1=single, 3-5=batch)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to graph")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start over")
    parser.add_argument("--status", action="store_true", help="Show migration progress")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("Migration state reset.")
    else:
        run_migration(batch_size=args.batch, dry_run=args.dry_run)
