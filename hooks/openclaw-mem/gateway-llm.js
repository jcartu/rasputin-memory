/**
 * DeepSeek LLM helper
 * Calls the DeepSeek OpenAI-compatible endpoint to summarize sessions.
 */

const SUMMARY_SESSION_PREFIX = 'mem-summary:';
const DEFAULT_DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1';
const DEFAULT_DEEPSEEK_MODEL = 'deepseek-chat';

function getDeepSeekBaseUrl() {
  return process.env.DEEPSEEK_BASE_URL
    || DEFAULT_DEEPSEEK_BASE_URL;
}

function getDeepSeekApiKey() {
  return process.env.DEEPSEEK_API_KEY || '';
}

function getDeepSeekModel() {
  return process.env.DEEPSEEK_MODEL
    || DEFAULT_DEEPSEEK_MODEL;
}

function truncateText(text, maxChars) {
  if (!text) return '';
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars) + '…';
}

function formatTranscript(messages, maxChars = 12000) {
  const lines = [];
  for (const m of messages) {
    const role = (m.role || 'unknown').toUpperCase();
    const content = String(m.content || '').replace(/\s+/g, ' ').trim();
    if (!content) continue;
    lines.push(`${role}: ${content}`);
  }
  return truncateText(lines.join('\n'), maxChars);
}

function parseSummaryJson(text) {
  if (!text) return null;
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    const obj = JSON.parse(match[0]);
    return obj && typeof obj === 'object' ? obj : null;
  } catch {
    return null;
  }
}

function normalizeSummaryFields(obj) {
  if (!obj) return null;
  const pick = (key) => {
    const val = obj[key];
    if (typeof val === 'string') return val.trim();
    if (val == null) return '';
    return String(val).trim();
  };
  return {
    request: pick('request'),
    investigated: pick('investigated'),
    learned: pick('learned'),
    completed: pick('completed'),
    next_steps: pick('next_steps')
  };
}

async function callGatewayChat(messages, options = {}) {
  const {
    sessionKey = 'unknown',
    temperature = 0.2,
    max_tokens = 300,
    model
  } = options;
  const apiKey = getDeepSeekApiKey();
  if (!apiKey) {
    console.log('[openclaw-mem] No DEEPSEEK_API_KEY found');
    return null;
  }
  const baseUrl = getDeepSeekBaseUrl();
  const resolvedModel = model || getDeepSeekModel();
  const url = `${baseUrl}/chat/completions`;
  const payload = {
    model: resolvedModel,
    stream: false,
    temperature,
    max_tokens,
    messages
  };

  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  };

  try {
    console.log('[openclaw-mem] Calling DeepSeek API...');
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const errText = await res.text();
      console.error('[openclaw-mem] DeepSeek API error:', res.status, errText);
      return null;
    }
    const json = await res.json();
    const content = json?.choices?.[0]?.message?.content || '';
    console.log('[openclaw-mem] DeepSeek response received');
    return content;
  } catch (err) {
    console.error('[openclaw-mem] DeepSeek fetch error:', err.message);
    return null;
  }
}

export async function summarizeSession(messages, options = {}) {
  const { sessionKey = 'unknown' } = options;
  const transcript = formatTranscript(messages, 12000);
  if (!transcript) return null;

  const buildPrompts = (strict = false) => {
    const systemPrompt = `You are a session summarizer for an AI agent memory system. Your summaries help the agent recall past work in future sessions.

INSTRUCTIONS:
- Focus on OUTCOMES and DELIVERABLES, not conversational flow
- Use action verbs: implemented, fixed, configured, discovered, decided, explored
- Be specific: include file names, tool names, error messages, key decisions
- Write in the language the user used (Chinese if they spoke Chinese, English if English)

OUTPUT FORMAT: Return ONLY a valid JSON object with these fields:
{
  "request": "What the user wanted to accomplish (1 sentence, specific)",
  "investigated": "What was explored or researched to fulfill the request",
  "learned": "Key technical insights, discoveries, or new understanding gained",
  "completed": "Concrete deliverables: what was built, fixed, configured, or decided",
  "next_steps": "Unfinished work or logical follow-up actions (null if fully completed)"
}

QUALITY GUIDELINES:
- "request" should capture the real goal, not just "user asked a question"
- "investigated" should list specific files read, APIs explored, architectures examined
- "learned" should contain reusable knowledge (not "learned how to do X" but the actual insight)
- "completed" should be a concrete outcome someone can verify
- "next_steps" should be actionable, not vague

${strict ? 'CRITICAL: Output ONLY the JSON object. No markdown, no explanation, no code fences.' : ''}`;

    const userPrompt = 'Session transcript:\n' + transcript + '\n\nJSON:';
    return [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ];
  };

  // First attempt
  let content = await callGatewayChat(buildPrompts(false), { sessionKey, temperature: 0.2, max_tokens: 600 });
  let parsed = parseSummaryJson(content || '');
  if (parsed) return normalizeSummaryFields(parsed);

  // Retry once with stricter instruction
  content = await callGatewayChat(buildPrompts(true), { sessionKey, temperature: 0.1, max_tokens: 600 });
  parsed = parseSummaryJson(content || '');
  if (parsed) return normalizeSummaryFields(parsed);

  return null;
}

// ============ Local Embedding Model (Qwen3-Embedding-0.6B) ============

const EMBEDDING_MODEL = 'Xenova/multilingual-e5-small';
const EMBEDDING_DIMS = 384;
const EMBEDDING_PREFIX = 'query: ';

// Singleton: lazily initialized embedding pipeline
let _extractorPromise = null;

function getExtractor() {
  if (!_extractorPromise) {
    _extractorPromise = (async () => {
      try {
        const { pipeline } = await import('@huggingface/transformers');
        console.log('[openclaw-mem] Loading embedding model (first run downloads ~110MB)...');
        const extractor = await pipeline('feature-extraction', EMBEDDING_MODEL);
        console.log('[openclaw-mem] Embedding model loaded');
        return extractor;
      } catch (err) {
        console.error('[openclaw-mem] Failed to load embedding model:', err.message);
        _extractorPromise = null; // Allow retry
        return null;
      }
    })();
  }
  return _extractorPromise;
}

/**
 * Generate embedding vector for text using local Qwen3-Embedding-0.6B model.
 * Returns Float32Array of 1024 dimensions, or null on failure.
 */
export async function callGatewayEmbeddings(text) {
  try {
    const extractor = await getExtractor();
    if (!extractor) return null;

    const input = EMBEDDING_PREFIX + text;
    const output = await extractor(input, {
      pooling: 'mean',
      normalize: true,
    });

    return new Float32Array(output.data);
  } catch (err) {
    console.error('[openclaw-mem] Embedding generation error:', err.message);
    return null;
  }
}

/**
 * Generate embeddings for multiple texts sequentially.
 * Returns array of Float32Array, or null entries on failure.
 */
export async function batchEmbeddings(texts) {
  const extractor = await getExtractor();
  if (!extractor) return texts.map(() => null);

  const results = [];
  for (const text of texts) {
    try {
      const input = EMBEDDING_PREFIX + text;
      const output = await extractor(input, {
        pooling: 'mean',
        normalize: true,
      });
      results.push(new Float32Array(output.data));
    } catch (err) {
      console.error('[openclaw-mem] Batch embedding error:', err.message);
      results.push(null);
    }
  }
  return results;
}

export { EMBEDDING_DIMS };

export const INTERNAL_SUMMARY_PREFIX = SUMMARY_SESSION_PREFIX;
export { callGatewayChat };
