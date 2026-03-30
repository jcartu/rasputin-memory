/**
 * OpenClaw-Mem LLM Extractor
 *
 * Structured observation extraction inspired by claude-mem's observer agent pattern.
 * Uses DeepSeek API to produce rich, searchable memory records.
 */

import { callGatewayChat } from './gateway-llm.js';

// ── Valid concept categories (fixed taxonomy for consistent search) ──
const VALID_CONCEPTS = [
  'how-it-works',      // understanding mechanisms
  'why-it-exists',     // purpose or rationale
  'what-changed',      // modifications made
  'problem-solution',  // issues and their fixes
  'gotcha',            // traps or edge cases
  'pattern',           // reusable approach
  'trade-off'          // pros/cons of a decision
];

// ── Cache ──
const conceptCache = new Map();
const CACHE_MAX_SIZE = 1000;
const CACHE_TTL = 60 * 60 * 1000; // 1 hour

function getCacheKey(text) {
  let hash = 0;
  const str = text.slice(0, 500);
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString(16);
}

function cleanCache() {
  if (conceptCache.size > CACHE_MAX_SIZE) {
    const now = Date.now();
    for (const [key, value] of conceptCache.entries()) {
      if (now - value.timestamp > CACHE_TTL) {
        conceptCache.delete(key);
      }
    }
    if (conceptCache.size > CACHE_MAX_SIZE) {
      const entries = [...conceptCache.entries()];
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      const toRemove = entries.slice(0, entries.length - CACHE_MAX_SIZE / 2);
      for (const [key] of toRemove) {
        conceptCache.delete(key);
      }
    }
  }
}

/**
 * Extract concepts from text using LLM
 */
export async function extractConcepts(text, options = {}) {
  if (!text || typeof text !== 'string' || text.trim().length < 10) {
    return [];
  }

  const cacheKey = getCacheKey(text);
  const cached = conceptCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.concepts;
  }

  try {
    const content = await callGatewayChat([
      {
        role: 'system',
        content: `You are a knowledge classifier. Categorize the given text into 2-4 concept categories from this fixed list:
- how-it-works: understanding mechanisms or implementation details
- why-it-exists: purpose, rationale, or motivation
- what-changed: modifications, updates, or configuration changes
- problem-solution: issues encountered and their fixes
- gotcha: traps, edge cases, or surprising behavior
- pattern: reusable approaches or best practices
- trade-off: pros/cons analysis or design decisions

Return ONLY a JSON array of matching categories. No explanation.`
      },
      {
        role: 'user',
        content: text.slice(0, 2000)
      }
    ], { sessionKey: 'extract-concepts', temperature: 0.1, max_tokens: 100 });

    if (!content) return [];

    let concepts = [];
    try {
      const match = content.match(/\[[\s\S]*?\]/);
      if (match) {
        concepts = JSON.parse(match[0]);
      }
    } catch (parseErr) {
      console.error('[openclaw-mem] Failed to parse concepts response:', parseErr.message);
      return [];
    }

    // Validate against fixed taxonomy
    concepts = concepts
      .filter(c => typeof c === 'string')
      .map(c => c.trim().toLowerCase())
      .filter(c => VALID_CONCEPTS.includes(c))
      .slice(0, 4);

    cleanCache();
    conceptCache.set(cacheKey, { concepts, timestamp: Date.now() });
    return concepts;
  } catch (err) {
    console.error('[openclaw-mem] Concept extraction error:', err.message);
    return [];
  }
}

/**
 * Extract structured observation from a tool call
 *
 * Produces rich, searchable records with:
 * - Accurate type classification
 * - Descriptive title (short, action-oriented)
 * - Detailed narrative (what happened, how it works, why it matters)
 * - Structured facts (self-contained, grep-friendly)
 * - Fixed concept categories
 */
export async function extractFromToolCall(data) {
  const { tool_name, tool_input, tool_response, filesRead, filesModified } = data;

  // Provide generous context (2000 chars each, not 300)
  const inputStr = typeof tool_input === 'string'
    ? tool_input.slice(0, 2000)
    : JSON.stringify(tool_input, null, 0).slice(0, 2000);

  const responseStr = typeof tool_response === 'string'
    ? tool_response.slice(0, 2000)
    : JSON.stringify(tool_response, null, 0).slice(0, 2000);

  try {
    const content = await callGatewayChat([
      {
        role: 'system',
        content: `You are OpenClaw-Mem, a specialized observer that creates searchable memory records for FUTURE SESSIONS.

Your job: analyze a tool call and produce a structured observation capturing what was LEARNED, BUILT, FIXED, or CONFIGURED.

RULES:
- Record deliverables and capabilities, not process steps
- Use action verbs: implemented, fixed, deployed, configured, migrated, optimized, discovered, decided
- The "narrative" field is the most important: explain WHAT happened, HOW it works, and WHY it matters
- Facts must be self-contained statements (each fact should make sense without the others)
- Title should be a short noun phrase (3-10 words) capturing the core topic

TYPE DEFINITIONS (pick exactly one):
- bugfix: something was broken and is now fixed
- feature: new capability or functionality added
- refactor: code restructured without behavior change
- change: generic modification (docs, config, dependencies)
- discovery: learning about existing system, reading code, exploring
- decision: architectural or design choice with rationale

CONCEPT CATEGORIES (pick 1-3):
- how-it-works: understanding mechanisms
- why-it-exists: purpose or rationale
- what-changed: modifications made
- problem-solution: issues and their fixes
- gotcha: traps or edge cases
- pattern: reusable approach
- trade-off: pros/cons of a decision

Return ONLY valid JSON, no markdown fences, no explanation.`
      },
      {
        role: 'user',
        content: `Tool: ${tool_name}
Input: ${inputStr}
Output: ${responseStr}
Files read: ${filesRead?.join(', ') || 'none'}
Files modified: ${filesModified?.join(', ') || 'none'}

Return JSON:
{
  "type": "one of: bugfix|feature|refactor|change|discovery|decision",
  "title": "Short descriptive title (3-10 words)",
  "narrative": "2-4 sentences: what was done, how it works, why it matters. Be specific and include key details.",
  "facts": ["Self-contained fact 1", "Self-contained fact 2", "...up to 5"],
  "concepts": ["category1", "category2"]
}`
      }
    ], { sessionKey: 'extract-toolcall', temperature: 0.2, max_tokens: 800 });

    if (!content) throw new Error('empty response');

    const match = content.match(/\{[\s\S]*\}/);
    if (match) {
      const result = JSON.parse(match[0]);

      // Validate type
      const validTypes = ['bugfix', 'feature', 'refactor', 'change', 'discovery', 'decision'];
      const type = validTypes.includes(result.type) ? result.type : 'discovery';

      // Validate concepts against fixed taxonomy
      const concepts = Array.isArray(result.concepts)
        ? result.concepts.filter(c => VALID_CONCEPTS.includes(c)).slice(0, 3)
        : [];

      return {
        type,
        title: (result.title || '').slice(0, 120),
        narrative: (result.narrative || '').slice(0, 1000),
        facts: Array.isArray(result.facts)
          ? result.facts.filter(f => typeof f === 'string').slice(0, 5)
          : [],
        concepts: concepts.length > 0 ? concepts : ['how-it-works']
      };
    }
  } catch (err) {
    console.error('[openclaw-mem] Tool extraction error:', err.message);
  }

  return {
    type: 'discovery',
    title: '',
    narrative: '',
    facts: [],
    concepts: ['how-it-works']
  };
}

/**
 * Batch extract concepts from multiple texts
 */
export async function batchExtractConcepts(texts) {
  const results = new Map();

  const uncached = [];
  for (const text of texts) {
    const cacheKey = getCacheKey(text);
    const cached = conceptCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      results.set(text, cached.concepts);
    } else {
      uncached.push(text);
    }
  }

  const BATCH_SIZE = 5;
  for (let i = 0; i < uncached.length; i += BATCH_SIZE) {
    const batch = uncached.slice(i, i + BATCH_SIZE);
    const promises = batch.map(text => extractConcepts(text));
    const batchResults = await Promise.all(promises);

    for (let j = 0; j < batch.length; j++) {
      results.set(batch[j], batchResults[j]);
    }
  }

  return results;
}

export default {
  extractConcepts,
  extractFromToolCall,
  batchExtractConcepts
};
