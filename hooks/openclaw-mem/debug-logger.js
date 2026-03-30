#!/usr/bin/env node
/**
 * OpenClaw-Mem 调试日志工具
 *
 * 显示完整的 API 调用流程：
 * 1. 用户消息
 * 2. AI 调用的 exec 命令
 * 3. API 收到的请求
 * 4. API 返回的响应
 * 5. AI 最终回复
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import http from 'http';

const SESSIONS_DIR = path.join(os.homedir(), '.openclaw', 'agents', 'main', 'sessions');
const LOG_FILE = '/tmp/openclaw-debug.log';

// 颜色
const c = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  bgRed: '\x1b[41m',
  bgGreen: '\x1b[42m',
  bgYellow: '\x1b[43m',
  bgBlue: '\x1b[44m',
};

function log(msg) {
  const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  const line = `[${ts}] ${msg}`;
  console.log(line);
  fs.appendFileSync(LOG_FILE, line.replace(/\x1b\[[0-9;]*m/g, '') + '\n');
}

function logSection(title, color = 'cyan') {
  console.log();
  console.log(`${c[color]}${'═'.repeat(60)}${c.reset}`);
  console.log(`${c[color]}${c.bold}  ${title}${c.reset}`);
  console.log(`${c[color]}${'═'.repeat(60)}${c.reset}`);
}

function logEvent(icon, label, content, color = 'white') {
  const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  console.log();
  console.log(`${c.dim}[${ts}]${c.reset} ${icon} ${c[color]}${c.bold}${label}${c.reset}`);
  if (content) {
    const lines = String(content).split('\n').slice(0, 15);
    lines.forEach(line => {
      console.log(`  ${c.dim}│${c.reset} ${line.slice(0, 120)}`);
    });
    if (String(content).split('\n').length > 15) {
      console.log(`  ${c.dim}│ ... (更多内容省略)${c.reset}`);
    }
  }
  fs.appendFileSync(LOG_FILE, `[${ts}] ${icon} ${label}\n${content || ''}\n\n`);
}

// 监控 Session 文件
let lastSessionFile = null;
let processedLines = new Set();

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

function watchSession() {
  const latestSession = findLatestSession();

  if (latestSession !== lastSessionFile) {
    lastSessionFile = latestSession;
    processedLines.clear();
    if (latestSession) {
      logEvent('📁', '监控会话文件', path.basename(latestSession), 'cyan');
    }
  }

  if (!lastSessionFile) return;

  try {
    const content = fs.readFileSync(lastSessionFile, 'utf-8');
    const lines = content.trim().split('\n');

    for (const line of lines) {
      const lineKey = line.slice(0, 150);
      if (processedLines.has(lineKey)) continue;
      processedLines.add(lineKey);

      try {
        const entry = JSON.parse(line);
        if (entry.type !== 'message') continue;

        const msg = entry.message;
        if (!msg) continue;

        // 用户消息
        if (msg.role === 'user') {
          let text = '';
          if (Array.isArray(msg.content)) {
            const textPart = msg.content.find(c => c.type === 'text');
            text = textPart?.text || '';
          } else {
            text = msg.content || '';
          }
          if (text && !text.startsWith('A new session')) {
            logEvent('👤', '用户消息', text.slice(0, 300), 'green');
          }
        }

        // AI 响应
        if (msg.role === 'assistant' && msg.content) {
          // 检查工具调用
          if (Array.isArray(msg.content)) {
            for (const block of msg.content) {
              if (block.type === 'toolCall') {
                const toolName = block.name;
                const args = block.arguments;

                if (toolName === 'exec') {
                  let argsObj;
                  try {
                    argsObj = typeof args === 'string' ? JSON.parse(args) : args;
                  } catch {
                    argsObj = { command: args };
                  }

                  logEvent('🔧', `工具调用: ${toolName}`,
                    `命令: ${argsObj.command || JSON.stringify(argsObj)}`, 'yellow');

                  // 高亮显示 curl 命令
                  if (argsObj.command && argsObj.command.includes('curl')) {
                    console.log(`  ${c.bgYellow}${c.bold} CURL 命令 ${c.reset}`);
                    console.log(`  ${c.yellow}${argsObj.command}${c.reset}`);
                  }
                }
              }

              if (block.type === 'text' && block.text) {
                logEvent('🤖', 'AI 响应', block.text.slice(0, 400), 'blue');
              }
            }
          }
        }

        // 工具结果
        if (msg.role === 'toolResult') {
          const toolName = msg.toolName || 'unknown';
          let result = '';
          if (Array.isArray(msg.content)) {
            const textPart = msg.content.find(c => c.type === 'text');
            result = textPart?.text || '';
          } else {
            result = msg.content || '';
          }

          if (toolName === 'exec') {
            const isEmpty = !result || result === '(no output)';
            const color = isEmpty ? 'red' : 'green';
            const icon = isEmpty ? '❌' : '✅';
            logEvent(icon, `exec 返回结果`, result || '(空)', color);

            if (isEmpty) {
              console.log(`  ${c.bgRed}${c.bold} 警告: exec 返回空！curl 可能失败了 ${c.reset}`);
            }
          }
        }

      } catch (e) {
        // 解析失败，忽略
      }
    }
  } catch {
    // 文件读取失败
  }
}

// 创建代理 API 服务器来拦截请求
const PROXY_PORT = 18791;
const TARGET_PORT = 18790;

function startProxyServer() {
  const proxy = http.createServer((req, res) => {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      // 记录请求
      logEvent('📥', `API 请求: ${req.method} ${req.url}`,
        body ? `Body: ${body}` : '(无 body)', 'magenta');

      // 转发到真实 API
      const options = {
        hostname: '127.0.0.1',
        port: TARGET_PORT,
        path: req.url,
        method: req.method,
        headers: req.headers
      };

      const proxyReq = http.request(options, (proxyRes) => {
        let responseBody = '';
        proxyRes.on('data', chunk => responseBody += chunk);
        proxyRes.on('end', () => {
          // 记录响应
          const preview = responseBody.slice(0, 500);
          logEvent('📤', `API 响应 (${proxyRes.statusCode})`, preview,
            proxyRes.statusCode === 200 ? 'green' : 'red');

          res.writeHead(proxyRes.statusCode, proxyRes.headers);
          res.end(responseBody);
        });
      });

      proxyReq.on('error', (err) => {
        logEvent('❌', 'API 请求失败', err.message, 'red');
        res.writeHead(500);
        res.end('Proxy error');
      });

      if (body) proxyReq.write(body);
      proxyReq.end();
    });
  });

  proxy.listen(PROXY_PORT, '127.0.0.1', () => {
    log(`${c.green}代理服务器运行在 http://127.0.0.1:${PROXY_PORT}${c.reset}`);
    log(`${c.dim}(转发到 http://127.0.0.1:${TARGET_PORT})${c.reset}`);
  });

  return proxy;
}

// 主函数
function main() {
  // 清空日志文件
  fs.writeFileSync(LOG_FILE, '');

  logSection('OpenClaw-Mem 调试日志工具', 'cyan');
  log('');
  log(`${c.bold}监控内容:${c.reset}`);
  log(`  • Session 文件: ${SESSIONS_DIR}`);
  log(`  • 日志文件: ${LOG_FILE}`);
  log('');
  log(`${c.yellow}提示: 按 Ctrl+C 退出${c.reset}`);
  log('');

  // 开始监控
  setInterval(watchSession, 300);

  logSection('等待事件...', 'dim');
}

main();

process.on('SIGINT', () => {
  console.log();
  log(`${c.yellow}调试日志已停止${c.reset}`);
  log(`${c.dim}完整日志保存在: ${LOG_FILE}${c.reset}`);
  process.exit(0);
});
