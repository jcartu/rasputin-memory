/**
 * OpenClaw-Mem Hook Handler
 *
 * Captures session content and provides memory context injection.
 *
 * Events handled:
 * - command:new - Save session content before reset
 * - gateway:startup - Initialize memory system
 * - agent:bootstrap - Inject historical context
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { summarizeSession, INTERNAL_SUMMARY_PREFIX, callGatewayEmbeddings } from './gateway-llm.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
console.log('[openclaw-mem] >>> HANDLER LOADED AT', new Date().toISOString(), '<<<');
const USE_LLM_EXTRACTION = false; // Disabled: no DEEPSEEK_API_KEY, saves failed spawn attempts
const SUMMARY_MAX_MESSAGES = 200;
const MCP_API_PORT = 18790;
const MEMORY_API_BASE = process.env.MEMORY_API_URL || `http://${process.env.MEMORY_API_HOST || 'localhost:7777'}`;
const HONCHO_BASE = `${process.env.HONCHO_URL || 'http://localhost:7780'}/v3`;
const HONCHO_WORKSPACE = process.env.WORKSPACE_NAME || 'memory';

// Track API server process
let apiServerProcess = null;
let apiServerStarted = false;

// Avoid recursive memory capture for internal LLM runs
function isInternalSessionKey(sessionKey) {
  if (!sessionKey || typeof sessionKey !== 'string') return false;
  return sessionKey.startsWith(INTERNAL_SUMMARY_PREFIX);
}

// Lazy load modules
let database = null;
let contextBuilder = null;
let extractor = null;

async function loadModules() {
  if (database && contextBuilder) return true;

  try {
    const dbModule = await import('./database.js');
    const ctxModule = await import('./context-builder.js');
    database = dbModule.default || dbModule.database;
    contextBuilder = ctxModule.default || ctxModule;

    if (USE_LLM_EXTRACTION) {
      // Try to load extractor (optional, for LLM extraction)
      try {
        const extractorModule = await import('./extractor.js');
        extractor = extractorModule.default || extractorModule;
      } catch (e) {
        console.log('[openclaw-mem] LLM extractor not available, using basic extraction');
      }
    }

    return true;
  } catch (err) {
    console.error('[openclaw-mem] Failed to load modules:', err.message);
    return false;
  }
}

// Generate UUID
function generateId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// Estimate tokens
function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(String(text).length / 4);
}

// Simple hash function for content deduplication
function hashContent(text) {
  if (!text) return '';
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash.toString(16);
}

/**
 * Read session transcript and extract conversation
 */
async function extractSessionContent(sessionFile, maxMessages = 20) {
  try {
    if (!sessionFile) return null;

    // Handle race condition: OpenClaw renames transcript to .reset.* or .deleted.*
    // before the hook fires. If original file is gone, find the renamed version.
    let filePath = sessionFile;
    try {
      await fs.access(filePath);
    } catch {
      const dir = path.dirname(sessionFile);
      const base = path.basename(sessionFile);
      try {
        const files = await fs.readdir(dir);
        const renamed = files.filter(f => f.startsWith(base + '.reset.') || f.startsWith(base + '.deleted.'));
        if (renamed.length > 0) {
          renamed.sort().reverse();
          filePath = path.join(dir, renamed[0]);
          console.log(`[openclaw-mem] Original file gone, using renamed: ${renamed[0]}`);
        } else {
          console.error(`[openclaw-mem] Session file not found: ${base} (no renamed version either)`);
          return null;
        }
      } catch (dirErr) {
        console.error(`[openclaw-mem] Cannot read session dir: ${dirErr.message}`);
        return null;
      }
    }

    const content = await fs.readFile(filePath, 'utf-8');
    const lines = content.trim().split('\n');

    const messages = [];
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'message' && entry.message) {
          const msg = entry.message;
          if ((msg.role === 'user' || msg.role === 'assistant') && msg.content) {
            const text = Array.isArray(msg.content)
              ? msg.content.find(c => c.type === 'text')?.text
              : msg.content;

            if (text && !text.startsWith('/')) {
              messages.push({
                role: msg.role,
                content: text.slice(0, 500) // Truncate long messages
              });
            }
          }
        }
      } catch {
        // Skip invalid lines
      }
    }

    return messages.slice(-maxMessages);
  } catch (err) {
    console.error('[openclaw-mem] Failed to read session file:', err.message);
    return null;
  }
}

/**
 * Extract tool calls and results from a session JSONL file
 * Returns array of {toolName, toolInput, toolResponse} objects
 */
async function extractToolCalls(sessionFile, maxCalls = 50) {
  try {
    if (!sessionFile) return [];

    // Handle race condition: OpenClaw renames transcript to .reset.* or .deleted.*
    let filePath = sessionFile;
    try {
      await fs.access(filePath);
    } catch {
      const dir = path.dirname(sessionFile);
      const base = path.basename(sessionFile);
      try {
        const files = await fs.readdir(dir);
        const renamed = files.filter(f => f.startsWith(base + '.reset.') || f.startsWith(base + '.deleted.'));
        if (renamed.length > 0) {
          renamed.sort().reverse();
          filePath = path.join(dir, renamed[0]);
        } else {
          return [];
        }
      } catch {
        return [];
      }
    }

    const content = await fs.readFile(filePath, 'utf-8');
    const lines = content.trim().split('\n');

    // Build map of toolCallId -> toolCall
    const toolCallMap = new Map();
    const toolResults = [];

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry.type !== 'message') continue;

        const msg = entry.message;
        if (!msg) continue;

        if (msg.role === 'assistant' && Array.isArray(msg.content)) {
          for (const block of msg.content) {
            if (block.type === 'toolCall' && block.id && block.name) {
              toolCallMap.set(block.id, {
                toolName: block.name,
                toolInput: block.arguments || {}
              });
            }
          }
        }

        if (msg.role === 'toolResult' && msg.toolCallId) {
          const resultText = Array.isArray(msg.content)
            ? msg.content.find(c => c.type === 'text')?.text || ''
            : (typeof msg.content === 'string' ? msg.content : '');
          toolResults.push({
            toolCallId: msg.toolCallId,
            toolName: msg.toolName || '',
            toolResponse: resultText.slice(0, 2000)
          });
        }
      } catch {
        // Skip invalid lines
      }
    }

    // Combine toolCalls with their results
    const observations = [];
    for (const result of toolResults) {
      const call = toolCallMap.get(result.toolCallId);
      if (call) {
        observations.push({
          toolName: call.toolName || result.toolName,
          toolInput: call.toolInput,
          toolResponse: result.toolResponse
        });
      }
    }

    return observations.slice(-maxCalls);
  } catch (err) {
    console.error('[openclaw-mem] Failed to extract tool calls:', err.message);
    return [];
  }
}

/**
 * Start the MCP HTTP API server
 */
async function startApiServer() {
  if (apiServerStarted) {
    console.log('[openclaw-mem] API server already started');
    return;
  }

  // Check if server is already running
  try {
    const response = await fetch(`http://127.0.0.1:${MCP_API_PORT}/health`, {
      signal: AbortSignal.timeout(1000)
    });
    if (response.ok) {
      console.log('[openclaw-mem] API server already running on port', MCP_API_PORT);
      apiServerStarted = true;
      return;
    }
  } catch {
    // Server not running, start it
  }

  const apiScript = path.join(__dirname, 'mcp-http-api.js');
  const logDir = path.join(os.homedir(), '.openclaw-mem', 'logs');

  // Ensure log directory exists
  try {
    await fs.mkdir(logDir, { recursive: true });
  } catch {}

  const logFile = path.join(logDir, 'api.log');

  try {
    // Start API server as detached process
    apiServerProcess = spawn('node', [apiScript], {
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, OPENCLAW_MEM_API_PORT: String(MCP_API_PORT) },
      cwd: __dirname
    });

    // Log output to file
    const logStream = await fs.open(logFile, 'a');
    apiServerProcess.stdout?.on('data', (data) => {
      logStream.write(data);
    });
    apiServerProcess.stderr?.on('data', (data) => {
      logStream.write(data);
    });

    apiServerProcess.unref();
    apiServerStarted = true;

    console.log(`[openclaw-mem] ✓ API server started on port ${MCP_API_PORT} (PID: ${apiServerProcess.pid})`);

    // Wait for server to be ready
    for (let i = 0; i < 10; i++) {
      await new Promise(r => setTimeout(r, 200));
      try {
        const response = await fetch(`http://127.0.0.1:${MCP_API_PORT}/health`, {
          signal: AbortSignal.timeout(500)
        });
        if (response.ok) {
          console.log('[openclaw-mem] ✓ API server is ready');
          break;
        }
      } catch {}
    }
  } catch (err) {
    console.error('[openclaw-mem] Failed to start API server:', err.message);
  }
}

/**
 * Handle gateway:startup event
 */
async function handleGatewayStartup(event) {
  console.log('[openclaw-mem] Gateway startup - initializing memory system');

  if (!await loadModules()) return;

  const stats = database.getStats();
  console.log(`[openclaw-mem] Memory stats: ${stats.total_sessions} sessions, ${stats.total_observations} observations`);

  // Start MCP HTTP API server
  await startApiServer();
}

/**
 * Handle agent:bootstrap event
 * Inject historical context into new agent sessions
 * AND capture incoming message to database
 */
async function handleAgentBootstrap(event) {
  console.log('[openclaw-mem] Agent bootstrap:', event.sessionKey);
  console.log('[openclaw-mem] Event keys:', Object.keys(event));
  console.log('[openclaw-mem] Context exists:', !!event.context);

  if (!await loadModules()) return;

  // IMPORTANT: Ensure event.context exists (modify event directly, not a local copy)
  if (!event.context) {
    console.log('[openclaw-mem] WARNING: event.context is missing, creating it');
    event.context = {};
  }

  console.log('[openclaw-mem] Context keys:', Object.keys(event.context));

  const workspaceDir = event.context.workspaceDir || path.join(os.homedir(), '.openclaw', 'workspace');
  const sessionKey = event.sessionKey || 'unknown';

  console.log('[openclaw-mem] workspaceDir:', workspaceDir);
  console.log('[openclaw-mem] bootstrapFiles exists:', !!event.context.bootstrapFiles);
  console.log('[openclaw-mem] bootstrapFiles is array:', Array.isArray(event.context.bootstrapFiles));
  console.log('[openclaw-mem] bootstrapFiles length before:', event.context.bootstrapFiles?.length);
  // Debug: show structure of first file
  if (event.context.bootstrapFiles?.[0]) {
    const sample = event.context.bootstrapFiles[0];
    console.log('[openclaw-mem] Sample file keys:', Object.keys(sample));
    console.log('[openclaw-mem] Sample file name:', sample.name);
    console.log('[openclaw-mem] Sample has content:', !!sample.content);
  }

  // Raw messages are no longer stored individually — session summaries capture the important bits.
  // This eliminates noise from greetings and low-value messages.
  console.log('[openclaw-mem] Skipping per-message capture (handled via session summary)');

  // Ensure API server is running
  await startApiServer();

  // Build context to inject (async for LLM extraction)
  const memContext = await contextBuilder.buildContext(workspaceDir, {
    observationLimit: 50,
    fullDetailCount: 5,
    useLLMExtraction: USE_LLM_EXTRACTION
  });

  // Build tool instructions for memory retrieval
  const toolInstructions = `
---

## 🧠 Memory Recall Tools

**When the user asks for details beyond the summary above, use these scripts:**

**1. Search memories (find relevant IDs)**
\`\`\`bash
~/.openclaw/hooks/openclaw-mem/mem-search.sh "keywords" 10
\`\`\`

**2. Get full details (use found IDs)**
\`\`\`bash
~/.openclaw/hooks/openclaw-mem/mem-get.sh ID1 ID2
\`\`\`

**Don't answer detail questions from summaries alone — fetch the full content first.**
`;

  if (memContext) {
    console.log(`[openclaw-mem] Built context: ${memContext.length} chars`);

    // Load hot-context files (recent cron outputs, R&D findings, intel summaries)
    let hotContextSection = '';
    try {
      const hotDir = path.join(workspaceDir, 'memory', 'hot-context');
      const hotFiles = await fs.readdir(hotDir).catch(() => []);
      const now = Date.now();
      const ttlMs = 24 * 60 * 60 * 1000; // 24h TTL
      const fresh = [];
      for (const f of hotFiles) {
        if (!f.endsWith('.md')) continue;
        try {
          const fpath = path.join(hotDir, f);
          const stat = await fs.stat(fpath);
          if (now - stat.mtimeMs < ttlMs) {
            const content = await fs.readFile(fpath, 'utf-8');
            fresh.push(`### ${f}\n${content.slice(0, 5000)}`);
          }
        } catch { /* skip */ }
      }
      if (fresh.length > 0) {
        hotContextSection = `\n\n---\n## 🔥 Recent Intelligence (last 24h)\n\n${fresh.join('\n\n---\n')}`;
        console.log(`[openclaw-mem] ✓ Loaded ${fresh.length} hot-context files`);
      }
    } catch (err) {
      console.log(`[openclaw-mem] Hot-context load failed (silent): ${err.message}`);
    }

    // Qdrant bootstrap search — pre-load relevant memories for this session
    let qdrantSection = '';
    try {
      const resp = await fetch(`${MEMORY_API_BASE}/search?q=active+tasks+recent+decisions+important+context&limit=25`, {
        signal: AbortSignal.timeout(5000)
      });
      if (resp.ok) {
        const data = await resp.json();
        const results = data.results || [];
        if (results.length > 0) {
          const lines = results.map(r => {
            const text = (r.text || r.payload?.text || '').slice(0, 400);
            const score = r.score ? ` [${r.score.toFixed(2)}]` : '';
            return text ? `• ${text}${score}` : null;
          }).filter(Boolean);
          if (lines.length > 0) {
            qdrantSection = `\n\n---\n## 🧠 Pre-loaded Memory (Qdrant)\n\n${lines.join('\n')}`;
            console.log(`[openclaw-mem] ✓ Qdrant bootstrap: ${results.length} memories pre-loaded`);
          }
        }
      }
    } catch (err) {
      console.log(`[openclaw-mem] Qdrant bootstrap failed (silent): ${err.message}`);
    }

    // Strategy: Write memory context to a dedicated file on disk
    // This ensures AI can read it with the Read tool
    const memContextFile = path.join(workspaceDir, 'SESSION-MEMORY.md');
    try {
      await fs.writeFile(memContextFile, memContext + hotContextSection + qdrantSection + toolInstructions, 'utf-8');
      console.log(`[openclaw-mem] ✓ Written SESSION-MEMORY.md to disk (${memContext.length} chars + hot-context + qdrant + tool instructions)`);
    } catch (err) {
      console.error('[openclaw-mem] Failed to write SESSION-MEMORY.md:', err.message);
    }

    // DISABLED: bootstrap injection bloats context. SESSION-MEMORY.md on disk is sufficient.
    // if (event.context.bootstrapFiles && Array.isArray(event.context.bootstrapFiles)) {
    //   const memoryFile = event.context.bootstrapFiles.find(f => f.name === 'MEMORY.md');
    //   if (memoryFile && memoryFile.content && !memoryFile.missing) {
    //     memoryFile.content = memoryFile.content + '\n\n---\n\n# Session Memory\n\nSee SESSION-MEMORY.md for recent activity and conversation history.\n\n' + memContext + toolInstructions;
    //     console.log('[openclaw-mem] ✓ Appended to MEMORY.md in bootstrapFiles');
    //   }
    // }
  // HONCHO CONTEXT INJECTION: read honcho-context.md if fresh and inject into bootstrap
  try {
    const honchoContextPath = path.join(workspaceDir, 'memory', 'honcho-context.md');
    const honchoStat = await fs.stat(honchoContextPath);
    if (Date.now() - honchoStat.mtimeMs < 120000) { // 2 min freshness window
      const honchoContent = await fs.readFile(honchoContextPath, 'utf-8');
      if (honchoContent.length > 50) {
        if (!event.context.bootstrapFiles) event.context.bootstrapFiles = [];
        event.context.bootstrapFiles.push({
          name: 'HONCHO-CONTEXT.md',
          content: honchoContent.slice(0, 4000),
          source: 'honcho'
        });
        console.log(`[openclaw-mem] ✓ Honcho context injected into bootstrap (${honchoContent.length} chars)`);
      }
    }
  } catch (err) {
    // File doesn't exist or is stale — skip silently
    console.log(`[openclaw-mem] Honcho context not injected: ${err.code || err.message}`);
  }

  } else {
    console.log('[openclaw-mem] No context to inject (empty memory)');
    // Still write tool instructions even if no context
    const memContextFile = path.join(workspaceDir, 'SESSION-MEMORY.md');
    try {
      await fs.writeFile(memContextFile, '# Session Memory\n\nNo recent observations found.\n' + toolInstructions, 'utf-8');
      console.log('[openclaw-mem] ✓ Written SESSION-MEMORY.md with tool instructions only');
    } catch (err) {
      console.error('[openclaw-mem] Failed to write SESSION-MEMORY.md:', err.message);
    }
  }
}

// Track active sessions by sessionKey
const activeSessions = new Map();

/**
 * Get or create a session ID for a given sessionKey
 */
function getOrCreateSessionForKey(sessionKey, workspaceDir) {
  if (activeSessions.has(sessionKey)) {
    return activeSessions.get(sessionKey);
  }

  const sessionId = generateId();
  database.createSession(sessionId, workspaceDir, sessionKey, 'bootstrap');
  activeSessions.set(sessionKey, sessionId);

  // Clean up old sessions after 1 hour
  setTimeout(() => {
    if (activeSessions.get(sessionKey) === sessionId) {
      activeSessions.delete(sessionKey);
      database.endSession(sessionId);
    }
  }, 60 * 60 * 1000);

  console.log(`[openclaw-mem] Created new session ${sessionId} for key ${sessionKey}`);
  return sessionId;
}

/**
 * Handle command:new event
 * Save session content before reset
 */
async function handleCommandNew(event) {
  console.log('[openclaw-mem] Command new - saving session');

  if (!await loadModules()) return;

  const context = event.context || {};
  const sessionKey = event.sessionKey || 'unknown';
  const sessionId = generateId();

  // Get workspace and session info
  const workspaceDir = context.workspaceDir ||
                       context.cfg?.agents?.defaults?.workspace ||
                       path.join(os.homedir(), '.openclaw', 'workspace');

  const sessionEntry = context.previousSessionEntry || context.sessionEntry || {};
  const sessionFile = sessionEntry.sessionFile;

  // Create session record
  database.createSession(sessionId, workspaceDir, sessionKey, context.commandSource || 'command');

  // Extract session content
  const messages = await extractSessionContent(sessionFile, 20);

  if (messages && messages.length > 0) {
    console.log(`[openclaw-mem] Extracted ${messages.length} messages from session`);
    // Raw messages are no longer stored individually — only the AI summary matters.
    console.log('[openclaw-mem] Generating AI summary...');

    // Generate AI summary using DeepSeek
    let aiSummary = null;
    try {
      aiSummary = await summarizeSession(messages, { sessionKey });
      console.log('[openclaw-mem] Kimi summary result:', aiSummary ? 'success' : 'null');
    } catch (err) {
      console.error('[openclaw-mem] Kimi summary error:', err.message);
    }

    if (aiSummary && (aiSummary.request || aiSummary.learned || aiSummary.completed || aiSummary.next_steps)) {
      const summaryContent = JSON.stringify(aiSummary);
      database.saveSummary(
        sessionId,
        summaryContent,
        aiSummary.request,
        aiSummary.investigated || null,
        aiSummary.learned,
        aiSummary.completed,
        aiSummary.next_steps
      );
      console.log('[openclaw-mem] ✓ AI summary saved');
    } else {
      // Fallback summary
      const userMessages = messages.filter(m => m.role === 'user');
      const assistantMessages = messages.filter(m => m.role === 'assistant');
      const fallbackRequest = userMessages[0]?.content?.slice(0, 200) || 'Session started';
      const fallbackCompleted = assistantMessages.slice(-1)[0]?.content?.slice(0, 200) || '';

      database.saveSummary(
        sessionId,
        `Session with ${messages.length} messages`,
        fallbackRequest,
        null,
        null,
        fallbackCompleted ? `Discussed: ${fallbackCompleted}` : null,
        null
      );
      console.log('[openclaw-mem] ✓ Fallback summary saved');
    }
  }

  // Extract and save tool call observations
  const toolCalls = await extractToolCalls(sessionFile, 50);
  if (toolCalls && toolCalls.length > 0) {
    console.log(`[openclaw-mem] Saving ${toolCalls.length} tool call observations from session`);
    let savedCount = 0;
    for (const tc of toolCalls) {
      try {
        const { toolName, toolInput, toolResponse } = tc;

        // Privacy check
        const inputStr = JSON.stringify(toolInput);
        if (shouldExclude(inputStr)) continue;

        // Skip noise tools
        const skipTools = ['AskUserQuestion', 'TaskList', 'TaskGet'];
        if (skipTools.includes(toolName)) continue;

        // Build summary
        let summary = '';
        if (toolInput?.file_path) {
          summary = `${toolName}: ${toolInput.file_path}`;
        } else if (toolInput?.command) {
          summary = `${toolName}: ${String(toolInput.command).slice(0, 80)}`;
        } else if (toolInput?.pattern) {
          summary = `${toolName}: ${toolInput.pattern}`;
        } else if (toolInput?.query) {
          summary = `${toolName}: ${String(toolInput.query).slice(0, 80)}`;
        } else if (toolInput?.url) {
          summary = `${toolName}: ${toolInput.url}`;
        } else {
          summary = `${toolName} operation`;
        }

        const filesRead = extractFilesRead(toolName, toolInput);
        const filesModified = extractFilesModified(toolName, toolInput);
        const toolType = classifyToolType(toolName, toolInput, toolResponse);

        let narrative = '';
        if (filesModified.length > 0) {
          narrative = `Modified ${filesModified.join(', ')}`;
        } else if (filesRead.length > 0) {
          narrative = `Read ${filesRead.join(', ')}`;
        } else if (toolInput?.command) {
          narrative = `Executed: ${String(toolInput.command).slice(0, 100)}`;
        } else if (toolInput?.query) {
          narrative = `Searched for: ${toolInput.query}`;
        }

        const saveResult = database.saveObservation(
          sessionId,
          toolName,
          toolInput,
          { output: toolResponse },
          {
            summary: summary.slice(0, 200),
            concepts: `${toolName} ${summary}`.slice(0, 500),
            tokensDiscovery: estimateTokens(toolResponse),
            tokensRead: estimateTokens(summary),
            type: toolType,
            narrative: narrative.slice(0, 1000),
            facts: null,
            filesRead,
            filesModified
          }
        );

        if (saveResult?.success) {
          savedCount++;
          // Fire-and-forget embedding
          const embeddingText = [summary, narrative].filter(Boolean).join(' ').trim();
          if (embeddingText.length > 10) {
            callGatewayEmbeddings(embeddingText).then(embedding => {
              if (embedding) {
                database.saveEmbedding(Number(saveResult.id), embedding);
              }
            }).catch(() => {});
          }
        }
      } catch (err) {
        console.error('[openclaw-mem] Error saving observation:', err.message);
      }
    }
    console.log(`[openclaw-mem] ✓ Saved ${savedCount}/${toolCalls.length} tool observations`);
  }

  // End session
  database.endSession(sessionId);
}

/**
 * Handle agent:response event
 * Skip storing raw assistant messages — session summary at stop/new captures the important bits.
 * This avoids noise from greetings, acknowledgments, and other low-value messages.
 */
async function handleAgentResponse(event) {
  console.log('[openclaw-mem] Agent response event (skipped — captured via session summary)');
}

/**
 * Handle message events
 * Skip storing raw messages — session summary at stop/new captures the important bits.
 * This avoids noise from greetings, acknowledgments, and other low-value messages.
 */
async function handleMessage(event) {
  console.log('[openclaw-mem] Message event');

  // AUTO-RECALL: async Qdrant search for user messages — fire and forget
  try {
    // Extract message content from event
    const messageContent = event.message?.content || event.content || event.text || '';
    if (!messageContent || typeof messageContent !== 'string' || messageContent.trim().length === 0) {
      console.log('[openclaw-mem] No message content found, skipping auto-recall');
      return;
    }

    // Skip slash commands
    if (messageContent.trim().startsWith('/')) {
      console.log('[openclaw-mem] Skipping slash command');
      return;
    }

    const workspaceDir = event.context?.workspaceDir || path.join(os.homedir(), '.openclaw', 'workspace');
    
    // Extract keywords from message
    const keywords = messageContent
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 3)
      .slice(0, 12)
      .join(' ');

    if (keywords.length <= 5) {
      console.log('[openclaw-mem] Message too short for auto-recall');
      return;
    }

    (async () => {
      try {
        const resp = await fetch(`${MEMORY_API_BASE}/search?q=${encodeURIComponent(keywords)}&limit=8`, {
          signal: AbortSignal.timeout(3000)
        });
        if (!resp.ok) {
          console.log(`[openclaw-mem] Auto-recall search failed: ${resp.status}`);
          return;
        }
        const data = await resp.json();
        const results = data.results || [];
        if (results.length === 0) {
          console.log('[openclaw-mem] Auto-recall: no results found');
          return;
        }

        // Filter results to only include those with actual text content
        const filteredResults = results.filter(r => {
          const text = r.text || r.payload?.text || '';
          return text && text.length > 0;
        });

        if (filteredResults.length === 0) {
          console.log('[openclaw-mem] Auto-recall: no results with text content');
          return;
        }

        const lines = [`# Auto-Recall — ${new Date().toISOString()}\n`, `Query: ${keywords}\n`];
        for (const r of filteredResults) {
          const score = r.score ? ` [${r.score.toFixed(2)}]` : '';
          const text = (r.text || r.payload?.text || '').slice(0, 300);
          if (text) lines.push(`• ${text}${score}`);
        }

        await fs.writeFile(
          path.join(workspaceDir, 'memory', 'last-recall.md'),
          lines.join('\n'),
          'utf-8'
        );
        console.log(`[openclaw-mem] ✓ Auto-recall: ${filteredResults.length} results written to last-recall.md`);
      } catch (err) {
        console.log(`[openclaw-mem] Auto-recall search failed (silent): ${err.message}`);
      }
    })();
  } catch (err) {
    console.log(`[openclaw-mem] Auto-recall setup failed (silent): ${err.message}`);
  }
}

/**
 * Check if content should be excluded from memory (privacy protection)
 */
function shouldExclude(content) {
  if (!content || typeof content !== 'string') return false;

  // <private> tag exclusion
  if (content.includes('<private>') || content.includes('</private>')) return true;

  // Sensitive file patterns
  const sensitivePatterns = [
    '.env',
    'credentials',
    'secret',
    'password',
    'api_key',
    'apikey',
    'api-key',
    'private_key',
    'privatekey',
    'access_token',
    'accesstoken',
    'auth_token',
    'authtoken'
  ];

  const lowerContent = content.toLowerCase();
  for (const pattern of sensitivePatterns) {
    if (lowerContent.includes(pattern)) return true;
  }

  return false;
}

/**
 * Extract files read from tool call
 */
function extractFilesRead(toolName, toolInput) {
  if (!toolInput) return [];

  switch (toolName) {
    case 'Read':
      return toolInput.file_path ? [toolInput.file_path] : [];
    case 'Grep':
      return toolInput.path ? [toolInput.path] : [];
    case 'Glob':
      // Glob returns matched files, but input doesn't contain them
      return [];
    default:
      return [];
  }
}

/**
 * Extract files modified from tool call
 */
function extractFilesModified(toolName, toolInput) {
  if (!toolInput) return [];

  switch (toolName) {
    case 'Edit':
      return toolInput.file_path ? [toolInput.file_path] : [];
    case 'Write':
      return toolInput.file_path ? [toolInput.file_path] : [];
    case 'NotebookEdit':
      return toolInput.notebook_path ? [toolInput.notebook_path] : [];
    default:
      return [];
  }
}

/**
 * Classify tool call type
 */
function classifyToolType(toolName, toolInput, toolResponse) {
  // File modification tools
  if (['Edit', 'Write', 'NotebookEdit'].includes(toolName)) {
    return 'modification';
  }

  // File reading tools
  if (['Read', 'Grep', 'Glob'].includes(toolName)) {
    return 'discovery';
  }

  // Command execution
  if (toolName === 'Bash') {
    const command = toolInput?.command || '';
    if (command.includes('git commit') || command.includes('git push')) {
      return 'commit';
    }
    if (command.includes('npm test') || command.includes('pytest') || command.includes('jest')) {
      return 'testing';
    }
    if (command.includes('npm install') || command.includes('pip install')) {
      return 'setup';
    }
    return 'command';
  }

  // Web tools
  if (['WebFetch', 'WebSearch'].includes(toolName)) {
    return 'research';
  }

  // Task/Agent tools
  if (toolName === 'Task') {
    return 'delegation';
  }

  return 'other';
}

/**
 * Handle tool:post event
 * Records every tool call for memory tracking
 */
async function handleToolPost(event) {
  console.log('[openclaw-mem] Tool post event');

  if (!await loadModules()) return;

  const toolName = event.tool_name || event.toolName || 'Unknown';
  const toolInput = event.tool_input || event.toolInput || event.input || {};
  const toolResponse = event.tool_response || event.toolResponse || event.response || event.output || {};
  const sessionKey = event.sessionKey || 'unknown';
  const workspaceDir = event.context?.workspaceDir || path.join(os.homedir(), '.openclaw', 'workspace');

  // Skip certain tools that generate noise
  const skipTools = ['AskUserQuestion', 'TaskList', 'TaskGet'];
  if (skipTools.includes(toolName)) {
    console.log(`[openclaw-mem] Skipping ${toolName} (noise filter)`);
    return;
  }

  // Privacy check - skip sensitive content
  const inputStr = JSON.stringify(toolInput);
  const responseStr = JSON.stringify(toolResponse);
  if (shouldExclude(inputStr) || shouldExclude(responseStr)) {
    console.log(`[openclaw-mem] Skipping ${toolName} (privacy filter)`);
    return;
  }

  // Extract metadata
  const filesRead = extractFilesRead(toolName, toolInput);
  const filesModified = extractFilesModified(toolName, toolInput);
  const toolType = classifyToolType(toolName, toolInput, toolResponse);

  // Build summary
  let summary = '';
  if (toolInput.file_path) {
    summary = `${toolName}: ${toolInput.file_path}`;
  } else if (toolInput.command) {
    summary = `${toolName}: ${toolInput.command.slice(0, 80)}`;
  } else if (toolInput.pattern) {
    summary = `${toolName}: ${toolInput.pattern}`;
  } else if (toolInput.query) {
    summary = `${toolName}: ${toolInput.query.slice(0, 80)}`;
  } else if (toolInput.url) {
    summary = `${toolName}: ${toolInput.url}`;
  } else {
    summary = `${toolName} operation`;
  }

  // Build basic narrative (fallback)
  let narrative = '';
  if (filesModified.length > 0) {
    narrative = `Modified ${filesModified.join(', ')}`;
  } else if (filesRead.length > 0) {
    narrative = `Read ${filesRead.join(', ')}`;
  } else if (toolInput.command) {
    narrative = `Executed command: ${toolInput.command.slice(0, 100)}`;
  } else if (toolInput.query) {
    narrative = `Searched for: ${toolInput.query}`;
  }

  // Get or create session
  let sessionId = getOrCreateSessionForKey(sessionKey, workspaceDir);

  // Try LLM extraction for richer metadata
  let extractedType = toolType;
  let extractedNarrative = narrative;
  let extractedFacts = null;
  let extractedConcepts = `${toolName} ${summary}`.slice(0, 500);

  if (USE_LLM_EXTRACTION && extractor && extractor.extractFromToolCall) {
    try {
      const extracted = await extractor.extractFromToolCall({
        tool_name: toolName,
        tool_input: toolInput,
        tool_response: toolResponse,
        filesRead,
        filesModified
      });

      if (extracted) {
        extractedType = extracted.type || toolType;
        extractedNarrative = extracted.narrative || narrative;
        extractedFacts = extracted.facts;
        extractedConcepts = extracted.concepts?.join(', ') || extractedConcepts;
        // Use LLM-generated title as summary if available
        if (extracted.title) {
          summary = extracted.title;
        }
      }
      console.log(`[openclaw-mem] LLM extracted: type=${extractedType}, title=${summary.slice(0, 60)}, concepts=${extractedConcepts}`);
    } catch (err) {
      console.log(`[openclaw-mem] LLM extraction failed, using fallback: ${err.message}`);
    }
  }

  // Save observation with extended metadata
  const saveResult = database.saveObservation(
    sessionId,
    toolName,
    toolInput,
    toolResponse,
    {
      summary: summary.slice(0, 200),
      concepts: extractedConcepts,
      tokensDiscovery: estimateTokens(responseStr),
      tokensRead: estimateTokens(summary),
      type: extractedType,
      narrative: extractedNarrative.slice(0, 1000),
      facts: extractedFacts,
      filesRead: filesRead,
      filesModified: filesModified
    }
  );

  console.log(`[openclaw-mem] ✓ Tool ${toolName} recorded (type: ${extractedType})`);

  // Fire-and-forget: generate embedding for the new observation
  if (saveResult.success && saveResult.id) {
    const embeddingText = [summary, extractedNarrative].filter(Boolean).join(' ').trim();
    if (embeddingText.length > 10) {
      callGatewayEmbeddings(embeddingText).then(embedding => {
        if (embedding) {
          database.saveEmbedding(Number(saveResult.id), embedding);
          console.log(`[openclaw-mem] ✓ Embedding saved for observation #${saveResult.id}`);
        }
      }).catch(err => {
        console.log(`[openclaw-mem] Embedding generation failed: ${err.message}`);
      });
    }
  }
}

/**
 * Handle user:prompt:submit event (UserPromptSubmit)
 * Records user prompts to user_prompts table
 */
async function handleUserPromptSubmit(event) {
  console.log('[openclaw-mem] User prompt submit event');

  if (!await loadModules()) return;

  const sessionKey = event.sessionKey || 'unknown';
  const workspaceDir = event.context?.workspaceDir || path.join(os.homedir(), '.openclaw', 'workspace');

  // Extract user prompt from various possible locations in event
  const prompt = event.prompt || event.content || event.message || event.text || event.input;

  if (!prompt || typeof prompt !== 'string' || prompt.trim().length === 0) {
    console.log('[openclaw-mem] No prompt content found in event');
    return;
  }

  // Skip slash commands
  if (prompt.trim().startsWith('/')) {
    console.log('[openclaw-mem] Skipping slash command');
    return;
  }

  // Privacy check
  if (shouldExclude(prompt)) {
    console.log('[openclaw-mem] Skipping prompt (privacy filter)');
    return;
  }

  // Get or create session
  const sessionId = getOrCreateSessionForKey(sessionKey, workspaceDir);

  // Save to user_prompts table
  database.saveUserPrompt(sessionId, prompt);
  console.log(`[openclaw-mem] ✓ User prompt saved (${prompt.slice(0, 50)}...)`);

  // User prompts are saved to user_prompts table only (no observation duplication).

  // HONCHO INGEST: push user message into Honcho for deriver processing — fire and forget
  try {
    const HONCHO_PEER = 'user';
    // Use a stable session name derived from the OpenClaw session key
    const honchoSessionName = `openclaw-${sessionKey.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 60)}`;

    if (prompt.length > 10) { // Skip trivially short messages
      (async () => {
        try {
          // Create or get session in Honcho
          const sessionResp = await fetch(`${HONCHO_BASE}/workspaces/${HONCHO_WORKSPACE}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name: honchoSessionName,
              peers: {
                user: { observe_me: true, observe_others: true },
                agent: { observe_me: true, observe_others: false }
              }
            }),
            signal: AbortSignal.timeout(3000)
          });
          // 200 = existing, 201 = created — both are fine
          if (!sessionResp.ok && sessionResp.status !== 409) {
            console.log(`[openclaw-mem] Honcho session create failed: ${sessionResp.status}`);
            return;
          }

          // Push the message
          const msgResp = await fetch(`${HONCHO_BASE}/workspaces/${HONCHO_WORKSPACE}/sessions/${honchoSessionName}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              messages: [{
                content: prompt.slice(0, 10000),
                peer_id: HONCHO_PEER,
                metadata: {
                  source: 'openclaw',
                  session_key: sessionKey,
                  timestamp: new Date().toISOString()
                }
              }]
            }),
            signal: AbortSignal.timeout(3000)
          });

          if (msgResp.ok) {
            console.log(`[openclaw-mem] ✓ Honcho ingest: user message pushed to session ${honchoSessionName}`);
          } else {
            const errText = await msgResp.text().catch(() => '');
            console.log(`[openclaw-mem] Honcho ingest failed: ${msgResp.status} ${errText.slice(0, 200)}`);
          }
        } catch (err) {
          console.log(`[openclaw-mem] Honcho ingest failed (silent): ${err.message}`);
        }
      })();
    }
  } catch (err) {
    console.log(`[openclaw-mem] Honcho ingest setup failed (silent): ${err.message}`);
  }

  // AUTO-RECALL: async Qdrant search — fire and forget, never blocks the turn
  try {
    const keywords = prompt
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 3)
      .slice(0, 12)
      .join(' ');

    if (keywords.length > 5) {
      (async () => {
        try {
          const resp = await fetch(`${MEMORY_API_BASE}/search?q=${encodeURIComponent(keywords)}&limit=8`, {
            signal: AbortSignal.timeout(3000)
          });
          if (!resp.ok) return;
          const data = await resp.json();
          const results = data.results || [];
          if (results.length === 0) return;

          const lines = [`# Auto-Recall — ${new Date().toISOString()}\n`, `Query: ${keywords}\n`];
          for (const r of results) {
            const score = r.score ? ` [${r.score.toFixed(2)}]` : '';
            const text = (r.text || r.payload?.text || '').slice(0, 300);
            if (text) lines.push(`• ${text}${score}`);
          }

          await fs.writeFile(
            path.join(workspaceDir, 'memory', 'last-recall.md'),
            lines.join('\n'),
            'utf-8'
          );
          console.log(`[openclaw-mem] ✓ Auto-recall: ${results.length} results written to last-recall.md`);
        } catch (err) {
          console.log(`[openclaw-mem] Auto-recall search failed (silent): ${err.message}`);
        }
      })();
    }
  } catch (err) {
    console.log(`[openclaw-mem] Auto-recall setup failed (silent): ${err.message}`);
  }

  // HONCHO CONTEXT: async query to Honcho dialectic API — fire and forget, never blocks the turn
  // Queries both /context (representation + peer card, fast ~0.6s) and /chat (dialectic, ~1.5s)
  // Writes combined result to memory/honcho-context.md
  try {
    const HONCHO_PEER = 'user';

    // Extract meaningful search terms from the prompt
    const searchTerms = prompt
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 3)
      .slice(0, 15)
      .join(' ');

    if (searchTerms.length > 5) {
      (async () => {
        try {
          const honchoContextFile = path.join(workspaceDir, 'memory', 'honcho-context.md');
          const lines = [`# Honcho User Context — ${new Date().toISOString()}\n`, `Prompt: ${prompt.slice(0, 200)}\n`];

          // 1. Get peer context (representation + peer card) — curated by search query
          //    This is fast (~0.6s) and gives us the structured observations
          try {
            const contextUrl = `${HONCHO_BASE}/workspaces/${HONCHO_WORKSPACE}/peers/${HONCHO_PEER}/context?` +
              `search_query=${encodeURIComponent(searchTerms)}` +
              `&include_most_frequent=true&max_conclusions=15`;
            const contextResp = await fetch(contextUrl, {
              signal: AbortSignal.timeout(5000)
            });
            if (contextResp.ok) {
              const contextData = await contextResp.json();
              const rep = contextData.representation || '';
              const peerCard = contextData.peer_card || '';
              if (rep) {
                lines.push(`\n## Representation (what Honcho knows about the user)\n\n${rep.slice(0, 3000)}`);
              }
              if (peerCard) {
                lines.push(`\n## Peer Card\n\n${peerCard}`);
              }
              console.log(`[openclaw-mem] ✓ Honcho context: ${rep.length} chars representation`);
            }
          } catch (ctxErr) {
            console.log(`[openclaw-mem] Honcho context fetch failed (silent): ${ctxErr.message}`);
          }

          // 2. Dialectic chat — ask Honcho what it knows relevant to the prompt
          //    This calls the agentic dialectic which reasons over memory (~1.5s with minimal)
          try {
            const chatResp = await fetch(`${HONCHO_BASE}/workspaces/${HONCHO_WORKSPACE}/peers/${HONCHO_PEER}/chat`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                query: `Based on what you know about the user, what context is most relevant to this message: "${prompt.slice(0, 500)}"`,
                reasoning_level: 'minimal'
              }),
              signal: AbortSignal.timeout(10000)
            });
            if (chatResp.ok) {
              const chatData = await chatResp.json();
              const content = chatData.content || '';
              if (content) {
                lines.push(`\n## Dialectic Context (Honcho's understanding)\n\n${content.slice(0, 2000)}`);
              }
              console.log(`[openclaw-mem] ✓ Honcho dialectic: ${content.length} chars`);
            }
          } catch (chatErr) {
            console.log(`[openclaw-mem] Honcho dialectic failed (silent): ${chatErr.message}`);
          }

          // Only write if we got something useful beyond the header
          if (lines.length > 2) {
            await fs.writeFile(honchoContextFile, lines.join('\n'), 'utf-8');
            console.log(`[openclaw-mem] ✓ Honcho context written to honcho-context.md`);
          }
        } catch (err) {
          console.log(`[openclaw-mem] Honcho context query failed (silent): ${err.message}`);
        }
      })();
    }
  } catch (err) {
    console.log(`[openclaw-mem] Honcho context setup failed (silent): ${err.message}`);
  }
}

/**
 * Handle agent:stop event (Stop)
 * Called when the model stops/completes a turn
 */
async function handleAgentStop(event) {
  console.log('[openclaw-mem] Agent stop event');

  if (!await loadModules()) return;

  const sessionKey = event.sessionKey || 'unknown';
  const workspaceDir = event.context?.workspaceDir || path.join(os.homedir(), '.openclaw', 'workspace');
  const stopReason = event.reason || event.stop_reason || 'unknown';

  // Get session
  const sessionId = getOrCreateSessionForKey(sessionKey, workspaceDir);

  // Record the stop event as an observation
  const summary = `Agent stopped: ${stopReason}`;
  database.saveObservation(
    sessionId,
    'AgentStop',
    { reason: stopReason, sessionKey },
    { stopped: true, timestamp: new Date().toISOString() },
    {
      summary,
      concepts: `stop, ${stopReason}`,
      tokensDiscovery: 10,
      tokensRead: 5,
      type: 'lifecycle',
      narrative: `Agent turn completed with reason: ${stopReason}`,
      facts: null,
      filesRead: null,
      filesModified: null
    }
  );

  console.log(`[openclaw-mem] ✓ Agent stop recorded (reason: ${stopReason})`);

  // HONCHO INGEST: push agent response into Honcho — fire and forget
  try {
    const honchoSessionName = `openclaw-${sessionKey.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 60)}`;

    // Extract agent response text from the event
    const agentResponse = event.response || event.content || event.text || '';
    const responseText = typeof agentResponse === 'string' ? agentResponse :
      (Array.isArray(agentResponse) ? agentResponse.find(c => c.type === 'text')?.text : '') || '';

    if (responseText && responseText.length > 10) {
      (async () => {
        try {
          // Ensure session exists
          await fetch(`${HONCHO_BASE}/workspaces/${HONCHO_WORKSPACE}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name: honchoSessionName,
              peers: {
                user: { observe_me: true, observe_others: true },
                agent: { observe_me: true, observe_others: false }
              }
            }),
            signal: AbortSignal.timeout(3000)
          });
          // Push agent message
          await fetch(`${HONCHO_BASE}/workspaces/${HONCHO_WORKSPACE}/sessions/${honchoSessionName}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              messages: [{
                content: responseText.slice(0, 10000),
                peer_id: 'agent',
                metadata: {
                  source: 'openclaw',
                  session_key: sessionKey,
                  timestamp: new Date().toISOString()
                }
              }]
            }),
            signal: AbortSignal.timeout(3000)
          });
          console.log(`[openclaw-mem] ✓ Honcho ingest: agent response pushed to session ${honchoSessionName}`);
        } catch (err) {
          console.log(`[openclaw-mem] Honcho agent ingest failed (silent): ${err.message}`);
        }
      })();
    }
  } catch (err) {
    console.log(`[openclaw-mem] Honcho agent ingest setup failed (silent): ${err.message}`);
  }

  // Generate summary on stop (Claude-Mem parity)
  try {
    const existing = database.getSummaryBySession(sessionId);
    if (existing) {
      console.log('[openclaw-mem] Summary already exists for session, skipping stop summary');
      return;
    }

    const agentId = event.context?.agentId || 'main';
    const sessionFile = path.join(os.homedir(), '.openclaw', 'agents', agentId, 'sessions', `${sessionKey}.jsonl`);
    let sessionFileExists = false;
    try {
      await fs.access(sessionFile);
      sessionFileExists = true;
    } catch {
      sessionFileExists = false;
    }

    if (!sessionFileExists) {
      console.log('[openclaw-mem] No session file found for stop summary');
      return;
    }

    const messages = await extractSessionContent(sessionFile, SUMMARY_MAX_MESSAGES);
    if (!messages || messages.length === 0) {
      console.log('[openclaw-mem] No messages found for stop summary');
      return;
    }

    let summary = null;
    try {
      summary = await summarizeSession(messages, { sessionKey });
    } catch {
      summary = null;
    }

    if (summary && (summary.request || summary.learned || summary.completed || summary.next_steps)) {
      const summaryContent = JSON.stringify(summary);
      database.saveSummary(
        sessionId,
        summaryContent,
        summary.request,
        summary.investigated || null,
        summary.learned,
        summary.completed,
        summary.next_steps
      );
      console.log('[openclaw-mem] ✓ Stop summary saved');
    } else {
      // Fallback summary if LLM failed
      const userMessages = messages.filter(m => m.role === 'user');
      const assistantMessages = messages.filter(m => m.role === 'assistant');
      const summaryContent = `Session with ${messages.length} messages (${userMessages.length} user, ${assistantMessages.length} assistant)`;
      const firstUserMsg = userMessages[0]?.content?.slice(0, 200) || '';
      const lastAssistant = assistantMessages.slice(-1)[0]?.content?.slice(0, 100) || 'various topics';

      database.saveSummary(
        sessionId,
        summaryContent,
        firstUserMsg,
        null,
        null,
        `Discussed: ${lastAssistant}`,
        null
      );
      console.log('[openclaw-mem] ✓ Stop summary saved (fallback)');
    }
  } catch (err) {
    console.error('[openclaw-mem] Stop summary error:', err.message);
  }

  // If this is an end_turn or max_tokens stop, we might want to
  // trigger a summary generation for the turn
  if (stopReason === 'end_turn' || stopReason === 'stop_sequence') {
    console.log('[openclaw-mem] Turn completed normally');
  } else if (stopReason === 'max_tokens') {
    console.log('[openclaw-mem] Turn stopped due to max tokens');
  }
}

/**
 * Main hook handler
 */
const openclawMemHandler = async (event) => {
  const eventType = event.type;
  const eventAction = event.action;
  const eventSessionKey = event.sessionKey;

  if (isInternalSessionKey(eventSessionKey)) {
    console.log('[openclaw-mem] Skipping internal session:', eventSessionKey);
    return;
  }

  console.log('[openclaw-mem] Event:', eventType, eventAction || '', '(v2026-02-03-1629)');

  try {
    if (eventType === 'gateway' && eventAction === 'startup') {
      await handleGatewayStartup(event);
      return;
    }

    if (eventType === 'agent' && eventAction === 'bootstrap') {
      await handleAgentBootstrap(event);
      return;
    }

    if (eventType === 'command' && eventAction === 'new') {
      await handleCommandNew(event);
      return;
    }

    // Handle agent response to capture assistant messages
    if (eventType === 'agent' && eventAction === 'response') {
      await handleAgentResponse(event);
      return;
    }

    // Handle tool:post events to capture tool calls
    if (eventType === 'tool' && eventAction === 'post') {
      await handleToolPost(event);
      return;
    }

    // Handle user:prompt:submit (UserPromptSubmit) - when user submits a prompt
    if ((eventType === 'user' && eventAction === 'prompt') ||
        (eventType === 'prompt' && eventAction === 'submit') ||
        (eventType === 'user' && eventAction === 'submit')) {
      await handleUserPromptSubmit(event);
      return;
    }

    // Handle agent:stop (Stop) - when model stops/completes a turn
    if (eventType === 'agent' && eventAction === 'stop') {
      await handleAgentStop(event);
      return;
    }

    // Handle message events (alternative event type)
    if (eventType === 'message') {
      await handleMessage(event);
      return;
    }

  } catch (err) {
    console.error('[openclaw-mem] Handler error:', err.message);
    console.error(err.stack);
  }
};

export default openclawMemHandler;
