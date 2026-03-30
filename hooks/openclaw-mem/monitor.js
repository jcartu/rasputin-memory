#!/usr/bin/env node
/**
 * OpenClaw-Mem Real-time Monitor
 * Watches for new observations and displays them in real-time
 */

import Database from 'better-sqlite3';
import path from 'node:path';
import os from 'node:os';

const DB_PATH = path.join(os.homedir(), '.openclaw-mem', 'memory.db');
const POLL_INTERVAL = 1000; // 1 second

const db = new Database(DB_PATH, { readonly: true });

// Get the latest observation ID at startup
let lastId = db.prepare('SELECT MAX(id) as maxId FROM observations').get()?.maxId || 0;
let lastSummaryId = db.prepare('SELECT MAX(id) as maxId FROM summaries').get()?.maxId || 0;

console.log('\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m');
console.log('\x1b[36m║\x1b[0m  \x1b[1m🧠 OpenClaw-Mem Real-time Monitor\x1b[0m                           \x1b[36m║\x1b[0m');
console.log('\x1b[36m╠══════════════════════════════════════════════════════════════╣\x1b[0m');
console.log('\x1b[36m║\x1b[0m  Database: ~/.openclaw-mem/memory.db                         \x1b[36m║\x1b[0m');
console.log('\x1b[36m║\x1b[0m  Starting from observation #' + lastId.toString().padEnd(33) + '\x1b[36m║\x1b[0m');
console.log('\x1b[36m║\x1b[0m  Press Ctrl+C to stop                                        \x1b[36m║\x1b[0m');
console.log('\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m');
console.log('');

const TYPE_COLORS = {
  'discovery': '\x1b[34m🔵\x1b[0m',
  'refactor': '\x1b[35m🔄\x1b[0m',
  'bugfix': '\x1b[31m🔴\x1b[0m',
  'feature': '\x1b[32m🟢\x1b[0m',
  'decision': '\x1b[33m⚖️\x1b[0m',
  'session-request': '\x1b[36m📝\x1b[0m',
  'problem-solution': '\x1b[33m💡\x1b[0m'
};

function formatTime(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function truncate(text, max = 60) {
  if (!text) return '';
  const clean = String(text).replace(/\s+/g, ' ').trim();
  if (clean.length <= max) return clean;
  return clean.slice(0, max - 3) + '...';
}

function getTypeIcon(type) {
  return TYPE_COLORS[type] || '\x1b[37m⚪\x1b[0m';
}

function checkNewObservations() {
  try {
    const newObs = db.prepare(`
      SELECT id, timestamp, tool_name, type, summary, narrative
      FROM observations
      WHERE id > ?
      ORDER BY id ASC
    `).all(lastId);

    for (const obs of newObs) {
      const time = formatTime(obs.timestamp);
      const icon = getTypeIcon(obs.type);
      const tool = obs.tool_name || 'unknown';
      const summary = truncate(obs.narrative || obs.summary || `${tool} operation`, 50);

      console.log(`\x1b[90m${time}\x1b[0m ${icon} \x1b[1m#${obs.id}\x1b[0m [\x1b[33m${tool}\x1b[0m] ${summary}`);
      lastId = obs.id;
    }

    // Check for new summaries
    const newSummaries = db.prepare(`
      SELECT id, created_at, request, learned, completed, next_steps
      FROM summaries
      WHERE id > ?
      ORDER BY id ASC
    `).all(lastSummaryId);

    for (const sum of newSummaries) {
      const time = formatTime(sum.created_at);
      console.log('');
      console.log(`\x1b[90m${time}\x1b[0m \x1b[32m📋 Session Summary #${sum.id}\x1b[0m`);
      if (sum.request) console.log(`  \x1b[36m请求:\x1b[0m ${truncate(sum.request, 70)}`);
      if (sum.learned) console.log(`  \x1b[35m学到:\x1b[0m ${truncate(sum.learned, 70)}`);
      if (sum.completed) console.log(`  \x1b[32m完成:\x1b[0m ${truncate(sum.completed, 70)}`);
      if (sum.next_steps) console.log(`  \x1b[33m下步:\x1b[0m ${truncate(sum.next_steps, 70)}`);
      console.log('');
      lastSummaryId = sum.id;
    }

  } catch (err) {
    // Database might be locked, retry next interval
  }
}

// Start polling
const interval = setInterval(checkNewObservations, POLL_INTERVAL);

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\x1b[90m监控已停止\x1b[0m');
  clearInterval(interval);
  db.close();
  process.exit(0);
});

// Initial check
checkNewObservations();
