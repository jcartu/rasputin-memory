/**
 * OpenClaw-Mem Database Module
 * SQLite-based storage for observations, sessions, and summaries
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';

const DATA_DIR = path.join(os.homedir(), '.openclaw-mem');
const DB_PATH = path.join(DATA_DIR, 'memory.db');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Initialize database
const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');

// Load sqlite-vec extension for vector search
try {
  sqliteVec.load(db);
  console.log('[openclaw-mem] sqlite-vec extension loaded');
} catch (e) {
  console.error('[openclaw-mem] Failed to load sqlite-vec:', e.message);
}

// Create tables (base schema without new columns for backward compatibility)
db.exec(`
  -- Sessions table
  CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_path TEXT,
    session_key TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    ended_at TEXT,
    status TEXT DEFAULT 'active',
    source TEXT
  );

  -- Observations table (tool calls) - base schema
  CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp TEXT DEFAULT (datetime('now')),
    tool_name TEXT NOT NULL,
    tool_input TEXT,
    tool_response TEXT,
    summary TEXT,
    concepts TEXT,
    tokens_discovery INTEGER DEFAULT 0,
    tokens_read INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
  );

  -- User prompts table (for tracking user inputs)
  CREATE TABLE IF NOT EXISTS user_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    content TEXT,
    timestamp TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
  );

  -- Summaries table
  CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    content TEXT,
    request TEXT,
    investigated TEXT,
    learned TEXT,
    completed TEXT,
    next_steps TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
  );

  -- Base indexes
  CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);
  CREATE INDEX IF NOT EXISTS idx_observations_timestamp ON observations(timestamp DESC);
  CREATE INDEX IF NOT EXISTS idx_observations_tool ON observations(tool_name);
  CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
  CREATE INDEX IF NOT EXISTS idx_user_prompts_session ON user_prompts(session_id);
`);

// Migrate existing database - add new columns if they don't exist
// Must be done before creating indexes on these columns
const migrations = [
  `ALTER TABLE observations ADD COLUMN type TEXT`,
  `ALTER TABLE observations ADD COLUMN narrative TEXT`,
  `ALTER TABLE observations ADD COLUMN facts TEXT`,
  `ALTER TABLE observations ADD COLUMN files_read TEXT`,
  `ALTER TABLE observations ADD COLUMN files_modified TEXT`,
  `ALTER TABLE summaries ADD COLUMN learned TEXT`
];

for (const migration of migrations) {
  try {
    db.exec(migration);
  } catch (e) {
    // Column already exists, ignore
  }
}

// Create index on type column after migration
try {
  db.exec(`CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(type)`);
} catch (e) {
  // Index might already exist
}

// Create/recreate FTS5 table with extended fields
// Drop old triggers first to avoid conflicts
try {
  db.exec(`DROP TRIGGER IF EXISTS observations_ai`);
  db.exec(`DROP TRIGGER IF EXISTS observations_ad`);
  db.exec(`DROP TRIGGER IF EXISTS observations_au`);
} catch (e) { /* triggers don't exist */ }

// Check if FTS table needs to be recreated with new columns
const ftsInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='observations_fts'`).get();
const needsRecreate = ftsInfo && !ftsInfo.sql.includes('narrative');

if (needsRecreate) {
  try {
    db.exec(`DROP TABLE IF EXISTS observations_fts`);
  } catch (e) { /* table doesn't exist */ }
}

// Create FTS5 table with extended fields
db.exec(`
  CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    tool_name,
    summary,
    concepts,
    narrative,
    facts,
    content='observations',
    content_rowid='id'
  );
`);

// Rebuild FTS index if we recreated the table
if (needsRecreate) {
  try {
    const allObs = db.prepare(`SELECT id, tool_name, summary, concepts, narrative, facts FROM observations`).all();
    const insertFts = db.prepare(`INSERT INTO observations_fts(rowid, tool_name, summary, concepts, narrative, facts) VALUES (?, ?, ?, ?, ?, ?)`);
    for (const obs of allObs) {
      insertFts.run(obs.id, obs.tool_name, obs.summary, obs.concepts, obs.narrative, obs.facts);
    }
  } catch (e) {
    console.error('[openclaw-mem] FTS rebuild error:', e.message);
  }
}

// Create triggers for FTS sync
db.exec(`
  CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
    INSERT INTO observations_fts(rowid, tool_name, summary, concepts, narrative, facts)
    VALUES (new.id, new.tool_name, new.summary, new.concepts, new.narrative, new.facts);
  END;

  CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, tool_name, summary, concepts, narrative, facts)
    VALUES ('delete', old.id, old.tool_name, old.summary, old.concepts, old.narrative, old.facts);
  END;

  CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, tool_name, summary, concepts, narrative, facts)
    VALUES ('delete', old.id, old.tool_name, old.summary, old.concepts, old.narrative, old.facts);
    INSERT INTO observations_fts(rowid, tool_name, summary, concepts, narrative, facts)
    VALUES (new.id, new.tool_name, new.summary, new.concepts, new.narrative, new.facts);
  END;
`);

// Create vec0 virtual table for vector embeddings
// Drop and recreate if dimension mismatch (migration from 768/1024 to 384)
try {
  const vecInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='observation_embeddings'`).get();
  if (vecInfo && !vecInfo.sql.includes('float[384]')) {
    console.log('[openclaw-mem] Recreating vec0 table with 384 dimensions...');
    db.exec(`DROP TABLE IF EXISTS observation_embeddings`);
  }
} catch (e) { /* table doesn't exist yet */ }

try {
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS observation_embeddings USING vec0(
      observation_id INTEGER PRIMARY KEY,
      embedding float[384]
    );
  `);
  console.log('[openclaw-mem] observation_embeddings vec0 table ready');
} catch (e) {
  console.error('[openclaw-mem] Failed to create vec0 table:', e.message);
}

// Prepared statements
const stmts = {
  // Sessions
  createSession: db.prepare(`
    INSERT INTO sessions (id, project_path, session_key, source)
    VALUES (?, ?, ?, ?)
  `),

  getSession: db.prepare(`
    SELECT * FROM sessions WHERE id = ?
  `),

  endSession: db.prepare(`
    UPDATE sessions SET ended_at = datetime('now'), status = 'completed'
    WHERE id = ?
  `),

  getActiveSession: db.prepare(`
    SELECT * FROM sessions WHERE session_key = ? AND status = 'active'
    ORDER BY started_at DESC LIMIT 1
  `),

  // Observations - extended with type, narrative, facts, files tracking
  saveObservation: db.prepare(`
    INSERT INTO observations (session_id, tool_name, tool_input, tool_response, summary, concepts, tokens_discovery, tokens_read, type, narrative, facts, files_read, files_modified)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `),

  getObservation: db.prepare(`
    SELECT * FROM observations WHERE id = ?
  `),

  getObservations: db.prepare(`
    SELECT * FROM observations WHERE id IN (SELECT value FROM json_each(?))
  `),

  updateObservationSummary: db.prepare(`
    UPDATE observations SET summary = ?, concepts = ?, tokens_read = ?
    WHERE id = ?
  `),

  getRecentObservations: db.prepare(`
    SELECT o.*, s.project_path
    FROM observations o
    JOIN sessions s ON o.session_id = s.id
    WHERE s.project_path = ?
    ORDER BY o.timestamp DESC
    LIMIT ?
  `),

  getRecentObservationsAll: db.prepare(`
    SELECT o.*, s.project_path
    FROM observations o
    JOIN sessions s ON o.session_id = s.id
    ORDER BY o.timestamp DESC
    LIMIT ?
  `),

  getObservationsByType: db.prepare(`
    SELECT o.*, s.project_path
    FROM observations o
    JOIN sessions s ON o.session_id = s.id
    WHERE o.type = ?
    ORDER BY o.timestamp DESC
    LIMIT ?
  `),

  searchObservations: db.prepare(`
    SELECT o.*, s.project_path,
           highlight(observations_fts, 1, '<mark>', '</mark>') as summary_highlight
    FROM observations_fts fts
    JOIN observations o ON fts.rowid = o.id
    JOIN sessions s ON o.session_id = s.id
    WHERE observations_fts MATCH ?
    ORDER BY rank
    LIMIT ?
  `),

  // User prompts
  saveUserPrompt: db.prepare(`
    INSERT INTO user_prompts (session_id, content)
    VALUES (?, ?)
  `),

  getRecentUserPrompts: db.prepare(`
    SELECT * FROM user_prompts
    WHERE session_id = ?
    ORDER BY timestamp DESC
    LIMIT ?
  `),

  // Summaries
  saveSummary: db.prepare(`
    INSERT INTO summaries (session_id, content, request, investigated, learned, completed, next_steps)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `),

  getRecentSummaries: db.prepare(`
    SELECT su.*, s.project_path
    FROM summaries su
    JOIN sessions s ON su.session_id = s.id
    WHERE s.project_path = ?
    ORDER BY su.created_at DESC
    LIMIT ?
  `),

  getSummaryBySession: db.prepare(`
    SELECT * FROM summaries
    WHERE session_id = ?
    ORDER BY id DESC
    LIMIT 1
  `),

  getSummaryBySessionKey: db.prepare(`
    SELECT su.*, s.session_key
    FROM summaries su
    JOIN sessions s ON su.session_id = s.id
    WHERE s.session_key = ?
    ORDER BY su.id DESC
    LIMIT 1
  `),

  // Embedding operations
  saveEmbedding: db.prepare(`
    INSERT OR REPLACE INTO observation_embeddings (observation_id, embedding)
    VALUES (?, ?)
  `),

  searchByVector: db.prepare(`
    SELECT observation_id, distance
    FROM observation_embeddings
    WHERE embedding MATCH ?
    AND k = ?
    ORDER BY distance
  `),

  getEmbeddingCount: db.prepare(`
    SELECT COUNT(*) as count FROM observation_embeddings
  `),

  getObservationsWithoutEmbeddings: db.prepare(`
    SELECT o.id, o.summary, o.narrative
    FROM observations o
    LEFT JOIN observation_embeddings oe ON o.id = oe.observation_id
    WHERE oe.observation_id IS NULL
    AND (o.summary IS NOT NULL OR o.narrative IS NOT NULL)
    ORDER BY o.id
    LIMIT ?
  `),

  // Stats
  getStats: db.prepare(`
    SELECT
      (SELECT COUNT(*) FROM sessions) as total_sessions,
      (SELECT COUNT(*) FROM observations) as total_observations,
      (SELECT COUNT(*) FROM summaries) as total_summaries,
      (SELECT COUNT(*) FROM user_prompts) as total_user_prompts,
      (SELECT SUM(tokens_discovery) FROM observations) as total_discovery_tokens,
      (SELECT SUM(tokens_read) FROM observations) as total_read_tokens,
      (SELECT COUNT(*) FROM observation_embeddings) as total_embeddings
  `)
};

// Database API
export const database = {
  // Session operations
  createSession(id, projectPath, sessionKey, source = 'unknown') {
    try {
      stmts.createSession.run(id, projectPath, sessionKey, source);
      return { success: true, id };
    } catch (err) {
      // Session might already exist
      return { success: false, error: err.message };
    }
  },

  getSession(id) {
    return stmts.getSession.get(id);
  },

  getActiveSession(sessionKey) {
    return stmts.getActiveSession.get(sessionKey);
  },

  endSession(id) {
    stmts.endSession.run(id);
  },

  // Observation operations - extended with type, narrative, facts, files tracking
  saveObservation(sessionId, toolName, toolInput, toolResponse, options = {}) {
    const {
      summary = null,
      concepts = null,
      tokensDiscovery = 0,
      tokensRead = 0,
      type = null,
      narrative = null,
      facts = null,
      filesRead = null,
      filesModified = null
    } = options;

    const result = stmts.saveObservation.run(
      sessionId,
      toolName,
      JSON.stringify(toolInput),
      JSON.stringify(toolResponse),
      summary,
      concepts,
      tokensDiscovery,
      tokensRead,
      type,
      narrative,
      typeof facts === 'string' ? facts : JSON.stringify(facts),
      typeof filesRead === 'string' ? filesRead : JSON.stringify(filesRead),
      typeof filesModified === 'string' ? filesModified : JSON.stringify(filesModified)
    );

    return { success: true, id: result.lastInsertRowid };
  },

  getObservation(id) {
    const row = stmts.getObservation.get(id);
    if (row) {
      row.tool_input = JSON.parse(row.tool_input || '{}');
      row.tool_response = JSON.parse(row.tool_response || '{}');
    }
    return row;
  },

  getObservations(ids) {
    const rows = stmts.getObservations.all(JSON.stringify(ids));
    return rows.map(row => ({
      ...row,
      tool_input: JSON.parse(row.tool_input || '{}'),
      tool_response: JSON.parse(row.tool_response || '{}')
    }));
  },

  updateObservationSummary(id, summary, concepts, tokensRead) {
    stmts.updateObservationSummary.run(summary, concepts, tokensRead, id);
  },

  getRecentObservations(projectPath, limit = 50) {
    const rows = projectPath
      ? stmts.getRecentObservations.all(projectPath, limit)
      : stmts.getRecentObservationsAll.all(limit);

    return rows.map(row => ({
      ...row,
      tool_input: JSON.parse(row.tool_input || '{}'),
      tool_response: JSON.parse(row.tool_response || '{}')
    }));
  },

  searchObservations(query, limit = 20) {
    try {
      // Check if query contains CJK characters (Chinese/Japanese/Korean)
      const hasCJK = /[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]/.test(query);

      let rows = [];

      if (!hasCJK) {
        // For non-CJK, use FTS5 search
        // Escape FTS5 special characters by wrapping in double quotes
        const safeQuery = query.includes('.') || query.includes('*') || query.includes('+')
          ? `"${query.replace(/"/g, '""')}"`
          : query;
        rows = stmts.searchObservations.all(safeQuery, limit);
      }

      // If FTS5 returned no results or query has CJK, use LIKE fallback
      if (rows.length === 0) {
        // Split query by spaces and search for each term (AND logic)
        const terms = query.split(/\s+/).filter(t => t.length > 0);

        if (terms.length === 1) {
          // Single term - simple LIKE
          const likeQuery = `%${terms[0]}%`;
          rows = db.prepare(`
            SELECT o.*, s.project_path
            FROM observations o
            JOIN sessions s ON o.session_id = s.id
            WHERE o.summary LIKE ?
               OR o.concepts LIKE ?
               OR o.narrative LIKE ?
               OR o.facts LIKE ?
            ORDER BY o.timestamp DESC
            LIMIT ?
          `).all(likeQuery, likeQuery, likeQuery, likeQuery, limit);
        } else {
          // Multiple terms - search for first term, results should contain all terms
          const firstTerm = `%${terms[0]}%`;
          const candidates = db.prepare(`
            SELECT o.*, s.project_path
            FROM observations o
            JOIN sessions s ON o.session_id = s.id
            WHERE o.summary LIKE ?
               OR o.concepts LIKE ?
               OR o.narrative LIKE ?
               OR o.facts LIKE ?
            ORDER BY o.timestamp DESC
            LIMIT 100
          `).all(firstTerm, firstTerm, firstTerm, firstTerm);

          // Filter to rows containing all terms
          rows = candidates.filter(row => {
            const text = `${row.summary || ''} ${row.concepts || ''} ${row.narrative || ''} ${row.facts || ''}`.toLowerCase();
            return terms.every(term => text.includes(term.toLowerCase()));
          }).slice(0, limit);
        }
      }

      return rows.map(row => ({
        ...row,
        tool_input: JSON.parse(row.tool_input || '{}'),
        tool_response: JSON.parse(row.tool_response || '{}'),
        facts: row.facts ? JSON.parse(row.facts) : null,
        files_read: row.files_read ? JSON.parse(row.files_read) : null,
        files_modified: row.files_modified ? JSON.parse(row.files_modified) : null
      }));
    } catch (err) {
      console.error('[openclaw-mem] Search error:', err.message);
      return [];
    }
  },

  getObservationsByType(type, limit = 20) {
    const rows = stmts.getObservationsByType.all(type, limit);
    return rows.map(row => ({
      ...row,
      tool_input: JSON.parse(row.tool_input || '{}'),
      tool_response: JSON.parse(row.tool_response || '{}'),
      facts: row.facts ? JSON.parse(row.facts) : null,
      files_read: row.files_read ? JSON.parse(row.files_read) : null,
      files_modified: row.files_modified ? JSON.parse(row.files_modified) : null
    }));
  },

  // User prompt operations
  saveUserPrompt(sessionId, content) {
    const result = stmts.saveUserPrompt.run(sessionId, content);
    return { success: true, id: result.lastInsertRowid };
  },

  getRecentUserPrompts(sessionId, limit = 10) {
    return stmts.getRecentUserPrompts.all(sessionId, limit);
  },

  // Summary operations
  saveSummary(sessionId, content, request = null, investigated = null, learned = null, completed = null, nextSteps = null) {
    const result = stmts.saveSummary.run(sessionId, content, request, investigated, learned, completed, nextSteps);
    return { success: true, id: result.lastInsertRowid };
  },

  getRecentSummaries(projectPath, limit = 5) {
    return stmts.getRecentSummaries.all(projectPath, limit);
  },

  getSummaryBySession(sessionId) {
    return stmts.getSummaryBySession.get(sessionId);
  },

  getSummaryBySessionKey(sessionKey) {
    return stmts.getSummaryBySessionKey.get(sessionKey);
  },

  // Embedding operations
  saveEmbedding(observationId, embedding) {
    try {
      // sqlite-vec expects Float32Array directly, not Buffer
      const vec = embedding instanceof Float32Array
        ? embedding
        : new Float32Array(embedding);
      stmts.saveEmbedding.run(BigInt(observationId), vec);
      return { success: true };
    } catch (err) {
      console.error('[openclaw-mem] saveEmbedding error:', err.message);
      return { success: false, error: err.message };
    }
  },

  searchByVector(embedding, limit = 20) {
    try {
      const vec = embedding instanceof Float32Array
        ? embedding
        : new Float32Array(embedding);
      const rows = stmts.searchByVector.all(vec, limit);
      return rows;
    } catch (err) {
      console.error('[openclaw-mem] searchByVector error:', err.message);
      return [];
    }
  },

  getEmbeddingCount() {
    try {
      return stmts.getEmbeddingCount.get().count;
    } catch {
      return 0;
    }
  },

  getObservationsWithoutEmbeddings(limit = 100) {
    try {
      return stmts.getObservationsWithoutEmbeddings.all(limit);
    } catch (err) {
      console.error('[openclaw-mem] getObservationsWithoutEmbeddings error:', err.message);
      return [];
    }
  },

  // Stats
  getStats() {
    return stmts.getStats.get();
  },

  // Close database
  close() {
    db.close();
  }
};

export default database;
