#!/usr/bin/env python3
"""
BrainBox — Hebbian Procedural Memory System.

Tracks co-occurrence patterns in agent behavior:
- File access co-occurrence (files used together)
- Command sequences (what follows what)
- Error→fix patterns (what resolved what)

Uses SQLite for persistence. Implements Hebbian learning:
"Neurons that fire together, wire together."

Usage:
    from brainbox import BrainBox
    bb = BrainBox()
    bb.record_access("config.yaml")
    bb.record_access("deploy.sh")       # strengthens config.yaml <-> deploy.sh
    bb.record_command("git pull")
    bb.record_command("npm install")     # strengthens git_pull -> npm_install
    bb.record_error_fix("ECONNREFUSED", "pm2 restart proxy")
    
    bb.suggest_files("config.yaml")      # returns ["deploy.sh", ...]
    bb.suggest_next_command("git pull")   # returns ["npm install", ...]
    bb.suggest_fix("ECONNREFUSED")       # returns ["pm2 restart proxy", ...]

CLI:
    python3 brainbox.py record-access "file.py"
    python3 brainbox.py record-command "git status"
    python3 brainbox.py record-error-fix "error msg" "fix command"
    python3 brainbox.py suggest-files "file.py"
    python3 brainbox.py suggest-commands "git pull"
    python3 brainbox.py suggest-fix "error msg"
    python3 brainbox.py stats
"""

import sqlite3
import os
import time
import json
import argparse
from typing import List, Tuple

DB_PATH = os.environ.get(
    "BRAINBOX_DB",
    os.path.join(os.path.dirname(__file__), "brainbox.db"),
)

# Hebbian parameters
LEARNING_RATE = 0.1
DECAY_RATE = 0.01
CO_OCCURRENCE_WINDOW = 60  # seconds — accesses within this window are "co-occurring"
SEQUENCE_WINDOW = 30       # seconds — commands within this window are "sequential"


class BrainBox:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self):
        c = self.conn
        # File co-occurrence weights
        c.execute("""
            CREATE TABLE IF NOT EXISTS file_cooccurrence (
                file_a TEXT NOT NULL,
                file_b TEXT NOT NULL,
                weight REAL DEFAULT 0.1,
                last_fired REAL DEFAULT 0,
                fire_count INTEGER DEFAULT 1,
                PRIMARY KEY (file_a, file_b)
            )
        """)
        # Command sequence weights
        c.execute("""
            CREATE TABLE IF NOT EXISTS command_sequence (
                cmd_a TEXT NOT NULL,
                cmd_b TEXT NOT NULL,
                weight REAL DEFAULT 0.1,
                last_fired REAL DEFAULT 0,
                fire_count INTEGER DEFAULT 1,
                PRIMARY KEY (cmd_a, cmd_b)
            )
        """)
        # Error -> fix associations
        c.execute("""
            CREATE TABLE IF NOT EXISTS error_fix (
                error_pattern TEXT NOT NULL,
                fix_command TEXT NOT NULL,
                weight REAL DEFAULT 0.1,
                last_fired REAL DEFAULT 0,
                fire_count INTEGER DEFAULT 1,
                success_count INTEGER DEFAULT 0,
                PRIMARY KEY (error_pattern, fix_command)
            )
        """)
        # Recent access log (for co-occurrence detection)
        c.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item TEXT NOT NULL,
                item_type TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        c.commit()

    def _hebbian_strengthen(self, table: str, key_a_col: str, key_b_col: str, a: str, b: str):
        """Strengthen a connection using Hebbian learning rule."""
        now = time.time()
        # Ensure a < b for file pairs to avoid duplicates
        if table == "file_cooccurrence" and a > b:
            a, b = b, a

        existing = self.conn.execute(
            f"SELECT weight, fire_count FROM {table} WHERE {key_a_col}=? AND {key_b_col}=?",
            (a, b),
        ).fetchone()

        if existing:
            old_weight, count = existing
            # Hebbian: strengthen proportionally, with diminishing returns
            new_weight = old_weight + LEARNING_RATE * (1.0 - old_weight)
            self.conn.execute(
                f"UPDATE {table} SET weight=?, last_fired=?, fire_count=? WHERE {key_a_col}=? AND {key_b_col}=?",
                (new_weight, now, count + 1, a, b),
            )
        else:
            self.conn.execute(
                f"INSERT INTO {table} ({key_a_col}, {key_b_col}, weight, last_fired, fire_count) VALUES (?,?,?,?,?)",
                (a, b, LEARNING_RATE, now, 1),
            )
        self.conn.commit()

    def record_access(self, filepath: str):
        """Record a file access. Strengthens co-occurrence with recent accesses."""
        now = time.time()
        # Find recent accesses within window
        recent = self.conn.execute(
            "SELECT DISTINCT item FROM access_log WHERE item_type='file' AND timestamp > ? AND item != ?",
            (now - CO_OCCURRENCE_WINDOW, filepath),
        ).fetchall()

        # Strengthen co-occurrence with each recent file
        for (other,) in recent:
            self._hebbian_strengthen("file_cooccurrence", "file_a", "file_b", filepath, other)

        # Log this access
        self.conn.execute(
            "INSERT INTO access_log (item, item_type, timestamp) VALUES (?,?,?)",
            (filepath, "file", now),
        )
        self.conn.commit()

    def record_command(self, command: str):
        """Record a command execution. Strengthens sequence with recent commands."""
        now = time.time()
        recent = self.conn.execute(
            "SELECT item FROM access_log WHERE item_type='command' AND timestamp > ? ORDER BY timestamp DESC LIMIT 3",
            (now - SEQUENCE_WINDOW,),
        ).fetchall()

        for (prev_cmd,) in recent:
            self._hebbian_strengthen("command_sequence", "cmd_a", "cmd_b", prev_cmd, command)

        self.conn.execute(
            "INSERT INTO access_log (item, item_type, timestamp) VALUES (?,?,?)",
            (command, "command", now),
        )
        self.conn.commit()

    def record_error_fix(self, error: str, fix: str, success: bool = True):
        """Record an error→fix association."""
        self._hebbian_strengthen("error_fix", "error_pattern", "fix_command", error, fix)
        if success:
            self.conn.execute(
                "UPDATE error_fix SET success_count = success_count + 1 WHERE error_pattern=? AND fix_command=?",
                (error, fix),
            )
            self.conn.commit()

    def suggest_files(self, filepath: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Suggest related files based on co-occurrence."""
        rows = self.conn.execute(
            """
            SELECT file_b as related, weight FROM file_cooccurrence WHERE file_a=?
            UNION
            SELECT file_a as related, weight FROM file_cooccurrence WHERE file_b=?
            ORDER BY weight DESC LIMIT ?
            """,
            (filepath, filepath, limit),
        ).fetchall()
        return [(r[0], round(r[1], 4)) for r in rows]

    def suggest_next_command(self, command: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Suggest likely next commands based on sequences."""
        rows = self.conn.execute(
            "SELECT cmd_b, weight FROM command_sequence WHERE cmd_a=? ORDER BY weight DESC LIMIT ?",
            (command, limit),
        ).fetchall()
        return [(r[0], round(r[1], 4)) for r in rows]

    def suggest_fix(self, error: str, limit: int = 5) -> List[Tuple[str, float, int]]:
        """Suggest fixes for an error pattern."""
        rows = self.conn.execute(
            "SELECT fix_command, weight, success_count FROM error_fix WHERE error_pattern=? ORDER BY weight DESC LIMIT ?",
            (error, limit),
        ).fetchall()
        return [(r[0], round(r[1], 4), r[2]) for r in rows]

    def decay_all(self):
        """Apply time-based decay to all weights (run periodically)."""
        self.conn.execute(
            f"UPDATE file_cooccurrence SET weight = weight * (1.0 - {DECAY_RATE}) WHERE weight > 0.01"
        )
        self.conn.execute(
            f"UPDATE command_sequence SET weight = weight * (1.0 - {DECAY_RATE}) WHERE weight > 0.01"
        )
        self.conn.execute(
            f"UPDATE error_fix SET weight = weight * (1.0 - {DECAY_RATE}) WHERE weight > 0.01"
        )
        # Clean up tiny weights
        for table in ["file_cooccurrence", "command_sequence", "error_fix"]:
            self.conn.execute(f"DELETE FROM {table} WHERE weight < 0.01")
        # Clean up old access log (keep last 24h)
        cutoff = time.time() - 86400
        self.conn.execute("DELETE FROM access_log WHERE timestamp < ?", (cutoff,))
        self.conn.commit()

    def stats(self) -> dict:
        """Return statistics about the memory."""
        result = {}
        for table in ["file_cooccurrence", "command_sequence", "error_fix"]:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            avg_w = self.conn.execute(f"SELECT AVG(weight) FROM {table}").fetchone()[0]
            max_w = self.conn.execute(f"SELECT MAX(weight) FROM {table}").fetchone()[0]
            result[table] = {
                "connections": count,
                "avg_weight": round(avg_w or 0, 4),
                "max_weight": round(max_w or 0, 4),
            }
        log_count = self.conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
        result["access_log_entries"] = log_count
        return result

    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="BrainBox Hebbian Procedural Memory")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("record-access")
    p.add_argument("file")

    p = sub.add_parser("record-command")
    p.add_argument("command")

    p = sub.add_parser("record-error-fix")
    p.add_argument("error")
    p.add_argument("fix")

    p = sub.add_parser("suggest-files")
    p.add_argument("file")

    p = sub.add_parser("suggest-commands")
    p.add_argument("command")

    p = sub.add_parser("suggest-fix")
    p.add_argument("error")

    sub.add_parser("stats")
    sub.add_parser("test")

    args = parser.parse_args()
    bb = BrainBox()

    if args.cmd == "record-access":
        bb.record_access(args.file)
        print(f"Recorded access: {args.file}")
    elif args.cmd == "record-command":
        bb.record_command(args.command)
        print(f"Recorded command: {args.command}")
    elif args.cmd == "record-error-fix":
        bb.record_error_fix(args.error, args.fix)
        print(f"Recorded: '{args.error}' -> '{args.fix}'")
    elif args.cmd == "suggest-files":
        results = bb.suggest_files(args.file)
        for f, w in results:
            print(f"  {w:.4f}  {f}")
    elif args.cmd == "suggest-commands":
        results = bb.suggest_next_command(args.command)
        for c, w in results:
            print(f"  {w:.4f}  {c}")
    elif args.cmd == "suggest-fix":
        results = bb.suggest_fix(args.error)
        for c, w, s in results:
            print(f"  {w:.4f}  (success:{s})  {c}")
    elif args.cmd == "stats":
        print(json.dumps(bb.stats(), indent=2))
    elif args.cmd == "test":
        print("Running BrainBox self-test...")
        # Simulate a workflow
        bb.record_access("config.yaml")
        time.sleep(0.1)
        bb.record_access("deploy.sh")
        time.sleep(0.1)
        bb.record_access("config.yaml")
        time.sleep(0.1)
        bb.record_access("Dockerfile")

        bb.record_command("git pull")
        time.sleep(0.1)
        bb.record_command("npm install")
        time.sleep(0.1)
        bb.record_command("npm test")

        bb.record_error_fix("ECONNREFUSED port 8889", "pm2 restart proxy-service")
        bb.record_error_fix("ECONNREFUSED port 8889", "pm2 restart proxy-service")
        bb.record_error_fix("CUDA OOM", "kill orphan llama processes")

        print("\nFile suggestions for 'config.yaml':")
        for f, w in bb.suggest_files("config.yaml"):
            print(f"  {w:.4f}  {f}")

        print("\nNext command after 'git pull':")
        for c, w in bb.suggest_next_command("git pull"):
            print(f"  {w:.4f}  {c}")

        print("\nFixes for 'ECONNREFUSED port 8889':")
        for c, w, s in bb.suggest_fix("ECONNREFUSED port 8889"):
            print(f"  {w:.4f}  (success:{s})  {c}")

        print("\nStats:")
        print(json.dumps(bb.stats(), indent=2))
        print("\n✅ BrainBox self-test passed!")
    else:
        parser.print_help()

    bb.close()


if __name__ == "__main__":
    main()
