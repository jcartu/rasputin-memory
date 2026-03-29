#!/usr/bin/env node
/**
 * Sync recent messages from OpenClaw session files to database
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import Database from 'better-sqlite3';

const SESSION_FILE = path.join(os.homedir(), '.openclaw/agents/main/sessions/b99a8d14-5b71-4f2d-8fb1-25b48cf4aa68.jsonl');
const DB_PATH = path.join(os.homedir(), '.openclaw-mem/memory.db');

const db = new Database(DB_PATH);
const insert = db.prepare(`
  INSERT INTO observations (session_id, timestamp, tool_name, tool_input, tool_response, summary, concepts, tokens_discovery, tokens_read)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const content = fs.readFileSync(SESSION_FILE, 'utf-8');
const lines = content.trim().split('\n').slice(-30); // Last 30 lines

let saved = 0;
for (const line of lines) {
  try {
    const entry = JSON.parse(line);
    if (entry.type === 'message' && entry.message) {
      const role = entry.message.role;
      let text = '';
      if (Array.isArray(entry.message.content)) {
        text = entry.message.content.map(c => c.text || c).join(' ');
      } else {
        text = entry.message.content || '';
      }

      if (!text.trim()) continue;

      const toolName = role === 'user' ? 'UserMessage' : 'AssistantMessage';
      const summary = text.slice(0, 100) + (text.length > 100 ? '...' : '');

      try {
        insert.run(
          'live-sync',
          entry.timestamp || new Date().toISOString(),
          toolName,
          JSON.stringify({ role }),
          JSON.stringify({ content: text.slice(0, 2000) }),
          summary,
          text.slice(0, 500),
          Math.ceil(text.length / 4),
          Math.ceil(summary.length / 4)
        );
        saved++;
        console.log(`✓ [${role}] ${text.slice(0, 50)}...`);
      } catch (e) {
        // Skip duplicates
      }
    }
  } catch (e) {}
}

console.log(`\n已同步 ${saved} 条消息`);
db.close();
