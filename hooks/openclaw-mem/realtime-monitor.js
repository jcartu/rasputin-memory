#!/usr/bin/env node
/**
 * OpenClaw 实时监控工具
 *
 * 显示：
 * - 用户消息
 * - 发送给 LLM 的请求
 * - LLM 的响应
 * - 工具调用
 * - API 调用
 *
 * 使用: node realtime-monitor.js
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import readline from 'readline';

// 配置
const OPENCLAW_DIR = path.join(os.homedir(), '.openclaw');
const GATEWAY_LOG = path.join(OPENCLAW_DIR, 'logs', 'gateway.log');
const SESSIONS_DIR = path.join(OPENCLAW_DIR, 'agents', 'main', 'sessions');
const API_LOG = path.join(os.homedir(), '.openclaw-mem', 'logs', 'api.log');

// 颜色代码
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  bgBlue: '\x1b[44m',
  bgGreen: '\x1b[42m',
  bgYellow: '\x1b[43m',
  bgRed: '\x1b[41m',
  bgMagenta: '\x1b[45m',
};

function colorize(text, color) {
  return `${colors[color]}${text}${colors.reset}`;
}

function timestamp() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

function truncate(text, maxLen = 200) {
  if (!text) return '';
  const clean = String(text).replace(/\s+/g, ' ').trim();
  if (clean.length <= maxLen) return clean;
  return clean.slice(0, maxLen - 3) + '...';
}

function printHeader() {
  console.clear();
  console.log(colorize('═'.repeat(80), 'cyan'));
  console.log(colorize('  🔍 OpenClaw 实时监控工具', 'bright'));
  console.log(colorize('═'.repeat(80), 'cyan'));
  console.log();
  console.log(colorize('  监控中... (Ctrl+C 退出)', 'dim'));
  console.log();
  console.log(colorize('─'.repeat(80), 'dim'));
  console.log();
}

function printEvent(type, content, extra = '') {
  const ts = colorize(`[${timestamp()}]`, 'dim');
  let icon, label, color;

  switch (type) {
    case 'user':
      icon = '👤';
      label = '用户消息';
      color = 'green';
      break;
    case 'assistant':
      icon = '🤖';
      label = 'AI 响应';
      color = 'blue';
      break;
    case 'tool_call':
      icon = '🔧';
      label = '工具调用';
      color = 'yellow';
      break;
    case 'tool_result':
      icon = '📋';
      label = '工具结果';
      color = 'cyan';
      break;
    case 'api_call':
      icon = '🌐';
      label = 'API 调用';
      color = 'magenta';
      break;
    case 'bootstrap':
      icon = '🚀';
      label = '会话启动';
      color = 'green';
      break;
    case 'hook':
      icon = '🪝';
      label = 'Hook 事件';
      color = 'cyan';
      break;
    case 'error':
      icon = '❌';
      label = '错误';
      color = 'red';
      break;
    case 'info':
      icon = 'ℹ️';
      label = '信息';
      color = 'white';
      break;
    default:
      icon = '•';
      label = type;
      color = 'white';
  }

  console.log(`${ts} ${icon} ${colorize(label, color)}${extra ? ` ${colorize(extra, 'dim')}` : ''}`);
  if (content) {
    const lines = content.split('\n').slice(0, 10);
    lines.forEach(line => {
      console.log(`     ${colorize('│', 'dim')} ${line}`);
    });
    if (content.split('\n').length > 10) {
      console.log(`     ${colorize('│', 'dim')} ${colorize('... (更多内容省略)', 'dim')}`);
    }
  }
  console.log();
}

// 监控最新的 session 文件
let lastSessionFile = null;
let lastSessionSize = 0;
let lastSessionLines = new Set();

function findLatestSession() {
  try {
    const files = fs.readdirSync(SESSIONS_DIR)
      .filter(f => f.endsWith('.jsonl'))
      .map(f => ({
        name: f,
        path: path.join(SESSIONS_DIR, f),
        mtime: fs.statSync(path.join(SESSIONS_DIR, f)).mtime
      }))
      .sort((a, b) => b.mtime - a.mtime);

    return files[0]?.path || null;
  } catch {
    return null;
  }
}

function parseSessionLine(line) {
  try {
    const entry = JSON.parse(line);

    if (entry.type === 'message' && entry.message) {
      const msg = entry.message;

      // 用户消息
      if (msg.role === 'user') {
        let content = '';
        if (Array.isArray(msg.content)) {
          const textPart = msg.content.find(c => c.type === 'text');
          content = textPart?.text || '';
        } else {
          content = msg.content || '';
        }
        if (content && !content.startsWith('/')) {
          return { type: 'user', content: truncate(content, 300) };
        }
      }

      // AI 响应
      if (msg.role === 'assistant') {
        let content = '';
        if (Array.isArray(msg.content)) {
          const textPart = msg.content.find(c => c.type === 'text');
          content = textPart?.text || '';
        } else {
          content = msg.content || '';
        }
        if (content) {
          return { type: 'assistant', content: truncate(content, 300) };
        }
      }

      // 工具调用
      if (msg.role === 'assistant' && msg.tool_calls) {
        for (const call of msg.tool_calls) {
          const toolName = call.function?.name || call.name || 'unknown';
          const toolArgs = call.function?.arguments || call.arguments || '{}';
          let args;
          try {
            args = JSON.parse(toolArgs);
          } catch {
            args = toolArgs;
          }

          let summary = toolName;
          if (args.command) summary += `: ${truncate(args.command, 100)}`;
          else if (args.file_path) summary += `: ${args.file_path}`;
          else if (args.query) summary += `: ${truncate(args.query, 100)}`;
          else if (args.url) summary += `: ${args.url}`;

          return { type: 'tool_call', content: summary, extra: `(${toolName})` };
        }
      }

      // 工具结果
      if (msg.role === 'toolResult' || msg.role === 'tool') {
        const toolName = msg.toolName || msg.name || 'unknown';
        let result = '';
        if (Array.isArray(msg.content)) {
          const textPart = msg.content.find(c => c.type === 'text');
          result = textPart?.text || '';
        } else {
          result = msg.content || '';
        }
        return { type: 'tool_result', content: truncate(result, 200), extra: `(${toolName})` };
      }
    }

    return null;
  } catch {
    return null;
  }
}

function watchSession() {
  const latestSession = findLatestSession();

  if (latestSession !== lastSessionFile) {
    lastSessionFile = latestSession;
    lastSessionSize = 0;
    lastSessionLines.clear();
    if (latestSession) {
      printEvent('info', `监控会话: ${path.basename(latestSession)}`);
    }
  }

  if (!lastSessionFile) return;

  try {
    const content = fs.readFileSync(lastSessionFile, 'utf-8');
    const lines = content.trim().split('\n');

    for (const line of lines) {
      const lineHash = line.slice(0, 100); // 简单去重
      if (lastSessionLines.has(lineHash)) continue;
      lastSessionLines.add(lineHash);

      const parsed = parseSessionLine(line);
      if (parsed) {
        printEvent(parsed.type, parsed.content, parsed.extra);
      }
    }
  } catch {
    // 文件可能正在被写入
  }
}

// 监控 Gateway 日志
let lastGatewaySize = 0;

function watchGatewayLog() {
  try {
    const stats = fs.statSync(GATEWAY_LOG);
    if (stats.size <= lastGatewaySize) return;

    const fd = fs.openSync(GATEWAY_LOG, 'r');
    const buffer = Buffer.alloc(stats.size - lastGatewaySize);
    fs.readSync(fd, buffer, 0, buffer.length, lastGatewaySize);
    fs.closeSync(fd);

    lastGatewaySize = stats.size;

    const newContent = buffer.toString('utf-8');
    const lines = newContent.split('\n');

    for (const line of lines) {
      if (!line.trim()) continue;

      // Hook 事件
      if (line.includes('[openclaw-mem]')) {
        if (line.includes('Agent bootstrap')) {
          printEvent('bootstrap', '新会话开始');
        } else if (line.includes('API server')) {
          printEvent('hook', truncate(line.replace(/.*\[openclaw-mem\]\s*/, ''), 100));
        } else if (line.includes('Tool') || line.includes('tool:post')) {
          printEvent('hook', truncate(line.replace(/.*\[openclaw-mem\]\s*/, ''), 100));
        }
      }

      // 错误
      if (line.toLowerCase().includes('error')) {
        printEvent('error', truncate(line, 200));
      }
    }
  } catch {
    // 日志文件可能不存在
  }
}

// 监控 API 日志
let lastApiSize = 0;

function watchApiLog() {
  try {
    const stats = fs.statSync(API_LOG);
    if (stats.size <= lastApiSize) return;

    const fd = fs.openSync(API_LOG, 'r');
    const buffer = Buffer.alloc(stats.size - lastApiSize);
    fs.readSync(fd, buffer, 0, buffer.length, lastApiSize);
    fs.closeSync(fd);

    lastApiSize = stats.size;

    const newContent = buffer.toString('utf-8');
    const lines = newContent.split('\n');

    for (const line of lines) {
      if (!line.trim()) continue;
      if (line.includes('/search') || line.includes('/get_observations') || line.includes('/timeline')) {
        printEvent('api_call', truncate(line, 150));
      }
    }
  } catch {
    // API 日志可能不存在
  }
}

// 主循环
function main() {
  printHeader();

  // 初始化文件位置
  try {
    lastGatewaySize = fs.statSync(GATEWAY_LOG).size;
  } catch {}
  try {
    lastApiSize = fs.statSync(API_LOG).size;
  } catch {}

  // 开始监控
  setInterval(() => {
    watchSession();
    watchGatewayLog();
    watchApiLog();
  }, 500);

  // 处理退出
  process.on('SIGINT', () => {
    console.log();
    console.log(colorize('监控已停止', 'yellow'));
    process.exit(0);
  });
}

main();
