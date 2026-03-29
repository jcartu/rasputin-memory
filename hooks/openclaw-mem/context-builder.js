/**
 * OpenClaw-Mem Context Builder
 * Generates context to inject into new sessions using progressive disclosure
 */

import path from 'node:path';
import database from './database.js';
import { batchExtractConcepts } from './extractor.js';

// Token estimation (4 chars ≈ 1 token)
const CHARS_PER_TOKEN = 4;

function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(String(text).length / CHARS_PER_TOKEN);
}

// Type label mapping (Claude-Mem style)
const LEGEND_ORDER = [
  'session-request',
  'gotcha',
  'problem-solution',
  'decision',
  'bugfix',
  'feature',
  'refactor',
  'discovery'
];

const TYPE_DISPLAY_MAP = {
  decision: 'decision',
  bugfix: 'bugfix',
  feature: 'feature',
  refactor: 'refactor',
  discovery: 'discovery',
  testing: 'problem-solution',
  setup: 'session-request',
  modification: 'refactor',
  command: 'problem-solution',
  commit: 'decision',
  research: 'discovery',
  delegation: 'session-request',
  other: 'discovery',
  user_input: 'session-request',
  userprompt: 'session-request',
  userpromptsubmit: 'session-request',
  usermessage: 'session-request',
  assistantmessage: 'discovery'
};

const TOOL_NAME_TYPE_MAP = {
  UserPrompt: 'session-request',
  UserMessage: 'session-request',
  UserPromptSubmit: 'session-request',
  AssistantMessage: 'discovery',
  Read: 'discovery',
  Grep: 'discovery',
  Glob: 'discovery',
  WebFetch: 'discovery',
  WebSearch: 'discovery',
  Edit: 'refactor',
  Write: 'refactor',
  NotebookEdit: 'refactor',
  Bash: 'problem-solution'
};

function getTypeLabel(observation) {
  const rawType = observation?.type ? String(observation.type).toLowerCase() : '';
  if (rawType) {
    return TYPE_DISPLAY_MAP[rawType] || rawType;
  }
  const toolName = observation?.tool_name || observation?.toolName || '';
  if (toolName && TOOL_NAME_TYPE_MAP[toolName]) {
    return TOOL_NAME_TYPE_MAP[toolName];
  }
  return 'discovery';
}

// Format timestamp
function formatTime(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function formatDate(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return '';
  return date.toISOString().split('T')[0];
}

function formatDateHeading(dateOrKey) {
  if (!dateOrKey) return '';
  let date;
  if (/^\d{4}-\d{2}-\d{2}$/.test(dateOrKey)) {
    date = new Date(`${dateOrKey}T00:00:00`);
  } else {
    date = new Date(dateOrKey);
  }
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' });
}

function getProjectName(projectPath) {
  if (!projectPath || typeof projectPath !== 'string') return '';
  return path.basename(projectPath) || projectPath;
}

function normalizeText(text) {
  if (!text) return '';
  return String(text).replace(/\s+/g, ' ').trim();
}

function stripMarkup(text) {
  if (!text) return '';
  return String(text).replace(/<[^>]+>/g, '');
}

function truncateText(text, max = 80) {
  if (!text) return '';
  const clean = normalizeText(text);
  if (clean.length <= max) return clean;
  return clean.slice(0, Math.max(0, max - 3)) + '...';
}

function safeJsonParse(value, fallback) {
  if (value === null || value === undefined) return fallback;
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function normalizeArray(value) {
  if (!value) return [];
  const parsed = safeJsonParse(value, []);
  if (Array.isArray(parsed)) return parsed;
  return [];
}

function normalizeObservation(observation) {
  if (!observation) return observation;
  return {
    ...observation,
    tool_input: safeJsonParse(observation.tool_input, {}),
    tool_response: safeJsonParse(observation.tool_response, {}),
    facts: normalizeArray(observation.facts),
    files_read: normalizeArray(observation.files_read),
    files_modified: normalizeArray(observation.files_modified)
  };
}

// Filter out low-value observations (like recall test queries)
function filterHighValueObservations(observations) {
  const lowValuePatterns = [
    '请查看 SESSION-MEMORY',
    'SESSION-MEMORY.md 里没有',
    '请查看 SESSION-MEMORY.md，告诉我',
    '记忆检索当前不可用',
    '/memory search',
    '/memory get',
    '之前没有记录到',
    '没有任何关于'
  ];

  return observations.filter(obs => {
    const summary = obs.summary || '';
    // Always filter out observations matching low-value patterns
    const isLowValue = lowValuePatterns.some(pattern => summary.includes(pattern));
    if (isLowValue) return false;

    // Keep observations that have actual content (not just metadata)
    return true;
  });
}

// Group observations by date
function groupByDate(observations) {
  const groups = new Map();
  for (const obs of observations) {
    const dateKey = formatDate(obs.timestamp) || 'Unknown';
    if (!groups.has(dateKey)) {
      groups.set(dateKey, []);
    }
    groups.get(dateKey).push(obs);
  }
  return groups;
}

// Build index table (Layer 1 - compact)
function buildIndexTable(observations, projectName = '') {
  if (!observations || observations.length === 0) {
    return '*(No recent observations)*';
  }

  const grouped = groupByDate(observations);
  const lines = [];

  for (const [dateKey, obs] of grouped.entries()) {
    const heading = formatDateHeading(dateKey) || dateKey;
    lines.push(`### ${heading}`);
    if (projectName) {
      lines.push(`**Project: ${projectName}**`);
    }
    lines.push('');
    lines.push('| ID | Time | T | Title | Tokens |');
    lines.push('|----|------|---|-------|--------|');

    for (const o of obs) {
      const id = `#${o.id}`;
      const time = formatTime(o.timestamp);
      const typeLabel = getTypeLabel(o);
      const title = o.narrative || o.summary || `${o.tool_name} operation`;
      const truncTitle = truncateText(title, 72);
      const tokens = `~${o.tokens_read || estimateTokens(title)}`;

      lines.push(`| ${id} | ${time} | ${typeLabel} | ${truncTitle} | ${tokens} |`);
    }
    lines.push('');
  }

  return lines.join('\n');
}

// Build full details (Layer 3 - expensive)
function buildFullDetails(observations, limit = 5) {
  if (!observations || observations.length === 0) {
    return '';
  }

  const toShow = observations.slice(0, limit);
  const lines = [];

  for (const raw of toShow) {
    const o = normalizeObservation(raw);
    const title = normalizeText(o.narrative || o.summary || `${o.tool_name} operation`);
    const typeLabel = getTypeLabel(o);
    const dateLabel = formatDateHeading(o.timestamp);
    const timeLabel = formatTime(o.timestamp);

    lines.push(`#### #${o.id} - ${truncateText(title, 120)}`);
    lines.push('');

    if (typeLabel) {
      lines.push(`**Type**: ${typeLabel}`);
    }
    if (dateLabel || timeLabel) {
      const when = [dateLabel, timeLabel].filter(Boolean).join(' ');
      lines.push(`**Time**: ${when}`);
    }
    if (o.tool_name) {
      lines.push(`**Tool**: ${o.tool_name}`);
    }
    lines.push('');

    if (o.summary) {
      lines.push(`**Summary**: ${normalizeText(o.summary)}`);
      lines.push('');
    }

    if (o.narrative && o.narrative !== o.summary) {
      lines.push(`**Narrative**: ${normalizeText(o.narrative)}`);
      lines.push('');
    }

    const observationFacts = Array.isArray(o.facts) ? o.facts.filter(Boolean) : [];
    if (observationFacts.length > 0) {
      lines.push('**Facts**:');
      for (const fact of observationFacts.slice(0, 6)) {
        lines.push(`- ${normalizeText(fact)}`);
      }
      lines.push('');
    }

    if (Array.isArray(o.files_read) && o.files_read.length > 0) {
      const files = o.files_read.map(f => `\`${f}\``).join(', ');
      lines.push(`**Files Read**: ${files}`);
      lines.push('');
    }

    if (Array.isArray(o.files_modified) && o.files_modified.length > 0) {
      const files = o.files_modified.map(f => `\`${f}\``).join(', ');
      lines.push(`**Files Modified**: ${files}`);
      lines.push('');
    }

    // Show key facts from tool input
    const input = o.tool_input || {};
    const inputFacts = [];

    if (input.file_path) inputFacts.push(`- File: \`${input.file_path}\``);
    if (input.command) inputFacts.push(`- Command: \`${input.command.slice(0, 100)}\``);
    if (input.pattern) inputFacts.push(`- Pattern: \`${input.pattern}\``);
    if (input.query) inputFacts.push(`- Query: ${input.query.slice(0, 100)}`);
    if (input.url) inputFacts.push(`- URL: ${input.url}`);

    if (inputFacts.length > 0) {
      lines.push('**Details**:');
      lines.push(...inputFacts);
      lines.push('');
    }

    lines.push('---');
    lines.push('');
  }

  return lines.join('\n');
}

// Build token economics summary
function buildTokenEconomics(observations) {
  let totalDiscovery = 0;
  let totalRead = 0;

  for (const o of observations) {
    totalDiscovery += o.tokens_discovery || 0;
    totalRead += o.tokens_read || estimateTokens(o.summary || '');
  }

  const savings = totalDiscovery - totalRead;
  const savingsPercent = totalDiscovery > 0 ? Math.round((savings / totalDiscovery) * 100) : 0;

  if (totalDiscovery === 0) {
    return `**Observations**: ${observations.length} | **Read cost**: ~${totalRead} tokens`;
  }

  return `**Token ROI**: Discovery ~${totalDiscovery} | Read ~${totalRead} | Saved ~${savings} (${savingsPercent}%)`;
}

function buildLegendLine(observations) {
  const typesSeen = new Set();
  for (const obs of observations) {
    typesSeen.add(getTypeLabel(obs));
  }
  const extras = [...typesSeen].filter(t => !LEGEND_ORDER.includes(t)).sort();
  const legend = [...LEGEND_ORDER, ...extras];
  return `**Legend:** ${legend.join(' | ')}`;
}

// Build retrieval instructions
function buildRetrievalInstructions() {
  return `
---

**MCP 3-Layer Retrieval (progressive disclosure)**:
1. \`search({ query, limit })\` → index only
2. \`timeline({ anchor, depth_before, depth_after })\` → local context
3. \`get_observations({ ids })\` → full details (only after filtering)

**Chat aliases**:
- \`/memory search <query>\`
- \`/memory timeline <id>\`
- \`/memory get <id>\`
`;
}

/**
 * Build topic summaries from user's actual recorded concepts
 * Uses LLM to extract meaningful keywords from full message content
 */
async function buildTopicSummaries() {
  // Get all recent observations
  const recentObs = database.getRecentObservations(null, 50);

  if (recentObs.length === 0) return '';

  // Collect unique message contents for LLM extraction
  const textsToExtract = [];
  const textToObsMap = new Map();

  for (const obs of recentObs) {
    if (obs.concepts && obs.concepts.length > 20) {
      const text = obs.concepts.slice(0, 500);
      if (!textToObsMap.has(text)) {
        textsToExtract.push(text);
        textToObsMap.set(text, [obs]);
      } else {
        textToObsMap.get(text).push(obs);
      }
    }
  }

  // Limit to 10 unique texts for API efficiency
  const limitedTexts = textsToExtract.slice(0, 10);

  // Use LLM to extract concepts from messages
  let conceptsMap;
  try {
    conceptsMap = await batchExtractConcepts(limitedTexts);
  } catch (err) {
    console.error('[openclaw-mem] LLM extraction failed:', err.message);
    return '';
  }

  // Count keyword frequency
  const keywordCounts = {};
  const keywordToObs = {};

  for (const [text, concepts] of conceptsMap.entries()) {
    const observations = textToObsMap.get(text) || [];
    for (const concept of concepts) {
      keywordCounts[concept] = (keywordCounts[concept] || 0) + observations.length;
      if (!keywordToObs[concept]) {
        keywordToObs[concept] = [];
      }
      keywordToObs[concept].push(...observations);
    }
  }

  // Get top keywords (mentioned at least twice)
  const topKeywords = Object.entries(keywordCounts)
    .filter(([_, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([keyword, _]) => keyword);

  if (topKeywords.length === 0) return '';

  const sections = [];
  const seenIds = new Set();

  // Build sections for each top keyword
  for (const keyword of topKeywords.slice(0, 5)) {
    // Get observations associated with this keyword
    const relatedObs = keywordToObs[keyword] || [];
    const newObs = relatedObs.filter(o => !seenIds.has(o.id));

    if (newObs.length > 0) {
      sections.push(`### ${keyword}`);
      sections.push('');
      for (const obs of newObs.slice(0, 2)) {
        seenIds.add(obs.id);
        const summary = obs.summary || '';
        if (summary.length > 10) {
          sections.push(`- **#${obs.id}**: ${summary.slice(0, 150)}${summary.length > 150 ? '...' : ''}`);
        }
      }
      sections.push('');
    }
  }

  if (sections.length > 0) {
    return '## Historical Topics\n\nKey concepts extracted from your conversations:\n\n' + sections.join('\n');
  }
  return '';
}

/**
 * Build complete context for session injection
 * @param {string} projectPath - Project path for filtering
 * @param {object} options - Configuration options
 * @returns {Promise<string|null>} - Generated context or null if empty
 */
export async function buildContext(projectPath, options = {}) {
  const {
    observationLimit = 50,
    fullDetailCount = 5,
    showTokenEconomics = true,
    showRetrievalInstructions = true,
    useLLMExtraction = true
  } = options;

  // Fetch recent observations and filter out low-value ones
  const rawObservations = database.getRecentObservations(projectPath, observationLimit * 3); // Fetch more to compensate for filtering
  const observations = filterHighValueObservations(rawObservations).slice(0, observationLimit);

  if (observations.length === 0) {
    return null; // No context to inject
  }

  // Fetch recent summaries
  const summaries = database.getRecentSummaries(projectPath, 3);
  const projectName = getProjectName(projectPath || observations[0]?.project_path);

  // Build context parts
  const parts = [];

  // Header
  parts.push('<openclaw-mem-context>');
  parts.push('# [openclaw-mem] recent context');
  parts.push('');
  parts.push(buildLegendLine(observations));
  parts.push('');

  // Token economics
  if (showTokenEconomics) {
    parts.push(buildTokenEconomics(observations));
    parts.push('');
  }

  // Index table (all observations, compact)
  parts.push(buildIndexTable(observations, projectName));
  parts.push('');

  // Topic summaries (from LLM extraction)
  if (useLLMExtraction) {
    try {
      const topicSummaries = await buildTopicSummaries();
      if (topicSummaries) {
        parts.push(topicSummaries);
        parts.push('');
      }
    } catch (err) {
      console.error('[openclaw-mem] Topic extraction failed:', err.message);
    }
  }

  // Full details (top N)
  if (fullDetailCount > 0) {
    const details = buildFullDetails(observations, fullDetailCount);
    if (details) {
      parts.push('## Recent Details');
      parts.push('');
      parts.push(details);
    }
  }

  // Session summaries
  if (summaries.length > 0) {
    parts.push('## Latest Session Summary');
    parts.push('');
    const s = summaries[0];
    if (s.request) parts.push(`- **Goal**: ${normalizeText(s.request)}`);
    if (s.learned) parts.push(`- **Learned**: ${normalizeText(s.learned)}`);
    if (s.completed) parts.push(`- **Completed**: ${normalizeText(s.completed)}`);
    if (s.next_steps) parts.push(`- **Next**: ${normalizeText(s.next_steps)}`);
    parts.push('');
  }

  // Retrieval instructions
  if (showRetrievalInstructions) {
    parts.push(buildRetrievalInstructions());
  }

  parts.push('</openclaw-mem-context>');

  return parts.join('\n');
}

/**
 * Search observations and return formatted results
 */
export function searchContext(query, limit = 20) {
  const results = database.searchObservations(query, limit);

  if (results.length === 0) {
    return `No observations found for query: "${query}"`;
  }

  const lines = [
    `## Search Results for "${query}"`,
    '',
    '| ID | Time | T | Title | Tokens |',
    '|----|------|---|-------|--------|'
  ];

  for (const r of results) {
    const summary = r.summary_highlight || r.summary || `${r.tool_name} operation`;
    const cleanSummary = stripMarkup(summary);
    const title = truncateText(cleanSummary, 72);
    const time = formatTime(r.timestamp);
    const typeLabel = getTypeLabel(r);
    const tokens = `~${r.tokens_read || estimateTokens(cleanSummary)}`;
    lines.push(`| #${r.id} | ${time} | ${typeLabel} | ${title} | ${tokens} |`);
  }

  lines.push('');
  lines.push(`*${results.length} results. Use \`timeline\` or \`get_observations\` for full details.*`);

  return lines.join('\n');
}

/**
 * Get full observation details by IDs
 */
export function getObservationDetails(ids) {
  const observations = database.getObservations(ids);

  if (observations.length === 0) {
    return `No observations found for IDs: ${ids.join(', ')}`;
  }

  return buildFullDetails(observations, observations.length);
}

/**
 * Get timeline around an observation
 */
export function getTimeline(anchorId, depthBefore = 3, depthAfter = 2) {
  const anchor = database.getObservation(anchorId);
  if (!anchor) {
    return `Observation #${anchorId} not found`;
  }

  // Get surrounding observations from same session
  const allObs = database.getRecentObservations(null, 100);
  const anchorIdx = allObs.findIndex(o => o.id === anchorId);

  if (anchorIdx === -1) {
    return buildFullDetails([anchor], 1);
  }

  const startIdx = Math.max(0, anchorIdx - depthAfter); // Note: list is DESC, so after = before in time
  const endIdx = Math.min(allObs.length, anchorIdx + depthBefore + 1);
  const timeline = allObs.slice(startIdx, endIdx).reverse();

  const lines = [
    `## Timeline around #${anchorId}`,
    ''
  ];

  for (const o of timeline) {
    const marker = o.id === anchorId ? '→' : ' ';
    const time = formatTime(o.timestamp);
    const typeLabel = getTypeLabel(o);
    const title = truncateText(o.narrative || o.summary || `${o.tool_name} operation`, 90);
    lines.push(`${marker} ${time} ${typeLabel} #${o.id}: ${title}`);
  }

  return lines.join('\n');
}

function normalizeIds(input) {
  const ids = [];
  const pushId = (value) => {
    if (value === null || value === undefined) return;
    const cleaned = String(value).replace(/^#/, '').trim();
    if (!cleaned) return;
    const parsed = Number(cleaned);
    if (!Number.isNaN(parsed)) ids.push(parsed);
  };

  if (Array.isArray(input)) {
    input.forEach(pushId);
    return ids;
  }

  if (typeof input === 'string') {
    input.split(/[,\s]+/).forEach(pushId);
    return ids;
  }

  pushId(input);
  return ids;
}

/**
 * MCP-style unified interfaces
 */
export function search(args = {}) {
  if (typeof args === 'string') {
    return searchContext(args);
  }
  const query = args.query || args.q;
  const limit = args.limit ?? args.maxResults ?? 20;
  if (!query) return 'No query provided.';
  return searchContext(query, limit);
}

export function timeline(args = {}) {
  if (typeof args === 'number' || typeof args === 'string') {
    const anchorId = Number(String(args).replace(/^#/, ''));
    if (Number.isNaN(anchorId)) return 'No anchor ID provided.';
    return getTimeline(anchorId);
  }
  const anchor = args.anchor ?? args.id ?? args.observation_id ?? args.observationId;
  const depthBefore = Number(args.depth_before ?? args.before ?? 3);
  const depthAfter = Number(args.depth_after ?? args.after ?? 2);
  const anchorId = Number(String(anchor ?? '').replace(/^#/, ''));
  if (Number.isNaN(anchorId)) return 'No anchor ID provided.';
  return getTimeline(anchorId, depthBefore, depthAfter);
}

export function get_observations(args = {}) {
  const ids = Array.isArray(args)
    ? normalizeIds(args)
    : normalizeIds(args.ids ?? args.id ?? args.observation_ids ?? args.observationIds);
  if (!ids.length) return 'No observation IDs provided.';
  return getObservationDetails(ids);
}

export function __IMPORTANT() {
  return [
    'Use the 3-layer workflow for memory retrieval:',
    '1) search → index only',
    '2) timeline → local context',
    '3) get_observations → full details after filtering'
  ].join('\n');
}

export default {
  buildContext,
  searchContext,
  getObservationDetails,
  getTimeline,
  search,
  timeline,
  get_observations,
  __IMPORTANT
};
