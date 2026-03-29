#!/usr/bin/env node
/**
 * OpenClaw Session Watcher
 * Watches session JSONL files for new messages and records them in real-time
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import Database from 'better-sqlite3';

const SESSIONS_DIR = path.join(os.homedir(), '.openclaw', 'agents', 'main', 'sessions');
const DB_PATH = path.join(os.homedir(), '.openclaw-mem', 'memory.db');
const POLL_INTERVAL = 2000; // 2 seconds

// Track file positions to only read new content
const filePositions = new Map();
const processedMessages = new Set();

let db;
try {
  db = new Database(DB_PATH);
} catch (err) {
  console.error('Cannot open database:', err.message);
  process.exit(1);
}

// Prepare statement for inserting observations
const insertStmt = db.prepare(`
  INSERT INTO observations (session_id, timestamp, tool_name, tool_input, tool_response, summary, concepts, tokens_discovery, tokens_read)
  VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)
`);

function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(String(text).length / 4);
}

function formatTime() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

console.log('\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m');
console.log('\x1b[36m║\x1b[0m  \x1b[1m📡 OpenClaw Session Watcher\x1b[0m                                \x1b[36m║\x1b[0m');
console.log('\x1b[36m╠══════════════════════════════════════════════════════════════╣\x1b[0m');
console.log('\x1b[36m║\x1b[0m  Sessions: ~/.openclaw/agents/main/sessions/                 \x1b[36m║\x1b[0m');
console.log('\x1b[36m║\x1b[0m  Database: ~/.openclaw-mem/memory.db                         \x1b[36m║\x1b[0m');
console.log('\x1b[36m║\x1b[0m  Press Ctrl+C to stop                                        \x1b[36m║\x1b[0m');
console.log('\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m');
console.log('');

function processNewLines(sessionKey, newContent) {
  const lines = newContent.trim().split('\n').filter(l => l.trim());

  for (const line of lines) {
    try {
      const entry = JSON.parse(line);

      // Skip if already processed (use a hash of content + timestamp)
      const msgId = `${entry.type}-${entry.timestamp || ''}-${(entry.content || entry.message?.content || '').slice(0, 50)}`;
      if (processedMessages.has(msgId)) continue;
      processedMessages.add(msgId);

      // Process message entries
      if (entry.type === 'message' && entry.message) {
        const msg = entry.message;
        const role = msg.role || 'unknown';
        const content = msg.content || '';

        if (!content.trim()) continue;

        const toolName = role === 'user' ? 'UserMessage' : 'AssistantMessage';
        const summary = content.slice(0, 100) + (content.length > 100 ? '...' : '');
        const tokens = estimateTokens(content);

        // Save to database
        try {
          insertStmt.run(
            sessionKey,
            toolName,
            JSON.stringify({ role, sessionKey }),
            JSON.stringify({ content: content.slice(0, 2000) }),
            summary,
            content.slice(0, 500),
            tokens,
            estimateTokens(summary)
          );

          const icon = role === 'user' ? '\x1b[33m👤\x1b[0m' : '\x1b[32m🤖\x1b[0m';
          const preview = content.slice(0, 60).replace(/\n/g, ' ');
          console.log(`\x1b[90m${formatTime()}\x1b[0m ${icon} [${role}] ${preview}${content.length > 60 ? '...' : ''}`);
        } catch (dbErr) {
          // Ignore duplicate errors
        }
      }

      // Process tool call entries
      if (entry.type === 'tool_use' || entry.type === 'tool_result') {
        const toolName = entry.name || entry.tool_name || 'unknown';
        const input = entry.input || entry.tool_input || {};
        const result = entry.result || entry.output || {};

        const summary = `${toolName}: ${JSON.stringify(input).slice(0, 80)}`;

        try {
          insertStmt.run(
            sessionKey,
            toolName,
            JSON.stringify(input),
            JSON.stringify(result).slice(0, 2000),
            summary,
            summary,
            estimateTokens(JSON.stringify(input)),
            estimateTokens(summary)
          );

          console.log(`\x1b[90m${formatTime()}\x1b[0m \x1b[35m🔧\x1b[0m [${toolName}]`);
        } catch (dbErr) {
          // Ignore duplicate errors
        }
      }
    } catch (parseErr) {
      // Skip invalid JSON lines
    }
  }
}

function watchSessions() {
  try {
    if (!fs.existsSync(SESSIONS_DIR)) {
      return;
    }

    const files = fs.readdirSync(SESSIONS_DIR).filter(f => f.endsWith('.jsonl'));

    for (const file of files) {
      const filePath = path.join(SESSIONS_DIR, file);
      const sessionKey = file.replace('.jsonl', '');

      try {
        const stats = fs.statSync(filePath);
        const currentSize = stats.size;
        const lastPosition = filePositions.get(filePath) || 0;

        if (currentSize > lastPosition) {
          // Read new content
          const fd = fs.openSync(filePath, 'r');
          const buffer = Buffer.alloc(currentSize - lastPosition);
          fs.readSync(fd, buffer, 0, buffer.length, lastPosition);
          fs.closeSync(fd);

          const newContent = buffer.toString('utf-8');
          processNewLines(sessionKey, newContent);

          filePositions.set(filePath, currentSize);
        }
      } catch (fileErr) {
        // File might be locked or deleted
      }
    }
  } catch (err) {
    // Directory might not exist yet
  }
}

// Initial scan - just record current positions without processing old content
try {
  if (fs.existsSync(SESSIONS_DIR)) {
    const files = fs.readdirSync(SESSIONS_DIR).filter(f => f.endsWith('.jsonl'));
    for (const file of files) {
      const filePath = path.join(SESSIONS_DIR, file);
      try {
        const stats = fs.statSync(filePath);
        filePositions.set(filePath, stats.size);
      } catch (e) {}
    }
    console.log(`\x1b[90m监控 ${files.length} 个会话文件...\x1b[0m`);
  }
} catch (e) {}

// Start watching
const interval = setInterval(watchSessions, POLL_INTERVAL);

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\x1b[90m监控已停止\x1b[0m');
  clearInterval(interval);
  db.close();
  process.exit(0);
});

console.log('\x1b[32m开始监控新消息...\x1b[0m\n');
