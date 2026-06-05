// Vercel Node serverless function (legacy req/res signature). Proxies
// chat completions to OpenRouter with per-lesson context injected as
// the system prompt. The widget POSTs:
//   { lesson_key: "1.2-intro-python", messages: [{role, content}, ...] }
//
// Bundles in chatbot/context/<lesson_key>.json are picked up via the
// includeFiles entry in vercel.json.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONTEXT_DIR = path.resolve(__dirname, '..', 'context');

const manifest = JSON.parse(readFileSync(path.join(CONTEXT_DIR, 'manifest.json'), 'utf8'));
const ALLOWED = new Set(manifest.lessons.map(l => l.key));
const MODEL = process.env.OPENROUTER_MODEL || 'anthropic/claude-haiku-4.5';
const MAX_HISTORY_TURNS = 20;
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

export const SYSTEM_PROMPT = `
You are a tutor for the Tamkeen Data Science & AI course. The student is working on an assignment for a specific lesson. Your job is to GUIDE their thinking without revealing the answer.

Hard rules (do NOT break these):
1. Never state the correct multiple-choice option (e.g. "the answer is b").
2. Never paraphrase the correct option's text verbatim or word-swap it.
3. Never write a complete solution for a coding task. You may show a 1-3 line starter pattern that points in the right direction.
4. Never reveal content from another lesson — your context window is scoped to a single lesson, so just stay within it.

Behaviour:
- Start by asking what the student has already considered, UNLESS they have already shared their reasoning.
- Reward reasoning: if the student says "I think the answer is X because Y", engage with their reasoning before nudging.
- Use the AUTHORED HINTS as your primary source of guidance — they were written by the instructor.
- Use the LESSON PAGES to add detail when the student asks "why" or wants deeper context.
- If the student asks you to just give the answer, decline politely and offer one more nudge.
- Keep responses concise (under 200 words usually). Use markdown sparingly. Code blocks for code only.
- If a question is outside the scope of THIS lesson, say so and redirect to the lesson README.

Pointing to sources (IMPORTANT):
- Whenever you reference lesson material, tell the student EXACTLY where to find it as a clickable markdown link, and name the section, e.g. "see [Bias–Variance → The Dartboard Picture](URL#the-dartboard-picture)".
- Each LESSON PAGE below is labelled "### <path> — <URL>". Use that exact URL — never invent or guess one. To point at the task itself, use the ASSIGNMENT URL.
- To deep-link a section, append "#" + the heading as a slug: lowercase it, turn spaces into hyphens, drop punctuation (e.g. "## The Dartboard Picture" → "#the-dartboard-picture"). If unsure of the slug, link the page without an anchor — the page alone is still correct.
`.trim();

// Returns the system message content as a single large text block flagged for
// prompt caching. The lesson context is identical across every turn of a
// conversation AND across students on the same lesson, so caching this ~80k-token
// prefix (ephemeral, ~5min TTL) collapses repeat latency and cost. OpenRouter
// forwards cache_control to Anthropic for Claude models.
export function buildSystemMessage(bundle) {
  const pages = bundle.lesson_pages
    .map(p => `### ${p.name} — ${p.url}\n\n${p.body}`)
    .join('\n\n---\n\n');
  const text = [
    SYSTEM_PROMPT,
    `\n\n=== LESSON: ${bundle.title} (${bundle.lesson_key}) ===\n`,
    `=== ASSIGNMENT (${bundle.assignment_filename}) — ${bundle.assignment_url} ===\n\n${bundle.assignment}`,
    bundle.hints ? `\n\n=== AUTHORED HINTS ===\n\n${bundle.hints}` : '',
    `\n\n=== LESSON PAGES (cite these URLs) ===\n\n${pages}`,
  ].join('');
  return [{ type: 'text', text, cache_control: { type: 'ephemeral' } }];
}

const bundleCache = new Map();
function loadBundle(lessonKey) {
  if (bundleCache.has(lessonKey)) return bundleCache.get(lessonKey);
  const file = path.join(CONTEXT_DIR, `${lessonKey}.json`);
  const bundle = JSON.parse(readFileSync(file, 'utf8'));
  bundleCache.set(lessonKey, bundle);
  return bundle;
}

function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

function sendJson(res, status, payload) {
  setCorsHeaders(res);
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(payload));
}

export default async function handler(req, res) {
  setCorsHeaders(res);

  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.end();
    return;
  }
  if (req.method !== 'POST') return sendJson(res, 405, { error: 'POST only' });

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) return sendJson(res, 500, { error: 'OPENROUTER_API_KEY not configured' });

  // Vercel auto-parses JSON when content-type is application/json.
  const body = typeof req.body === 'object' && req.body !== null ? req.body : null;
  if (!body) return sendJson(res, 400, { error: 'JSON body required' });

  const { lesson_key, messages } = body;
  if (!lesson_key || typeof lesson_key !== 'string') return sendJson(res, 400, { error: 'lesson_key required' });
  if (!ALLOWED.has(lesson_key)) return sendJson(res, 400, { error: `unknown lesson_key: ${lesson_key}` });
  if (!Array.isArray(messages) || messages.length === 0) return sendJson(res, 400, { error: 'messages array required' });

  const trimmed = messages
    .filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .slice(-MAX_HISTORY_TURNS);
  if (trimmed.length === 0) return sendJson(res, 400, { error: 'no valid messages' });

  let bundle;
  try { bundle = loadBundle(lesson_key); }
  catch (e) { return sendJson(res, 500, { error: `context load failed: ${e.message}` }); }

  let upstream;
  try {
    upstream = await fetch(OPENROUTER_URL, {
      method: 'POST',
      headers: {
        'authorization': `Bearer ${apiKey}`,
        'content-type': 'application/json',
        'http-referer': process.env.OPENROUTER_REFERRER || 'https://codebyshennan.github.io/dsai',
        'x-title': 'Tamkeen DSAI Assignment Tutor',
      },
      body: JSON.stringify({
        model: MODEL,
        stream: true,
        temperature: 0.4,
        messages: [
          { role: 'system', content: buildSystemMessage(bundle) },
          ...trimmed,
        ],
      }),
    });
  } catch (e) {
    return sendJson(res, 502, { error: `fetch failed: ${e.message}` });
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => '');
    return sendJson(res, 502, { error: `upstream ${upstream.status}`, detail: text.slice(0, 500) });
  }

  // Stream the OpenRouter SSE straight through. The widget consumes
  // `data: {choices:[{delta:{content}}]}` chunks (and the terminal [DONE]),
  // so a byte-for-byte passthrough is all that's needed — no re-framing.
  res.statusCode = 200;
  res.setHeader('content-type', 'text/event-stream; charset=utf-8');
  res.setHeader('cache-control', 'no-cache, no-transform');
  res.setHeader('connection', 'keep-alive');

  const reader = upstream.body.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(value);
    }
  } catch (e) {
    // Mid-stream upstream failure: surface as an SSE comment, then close.
    res.write(`: stream error ${String(e.message).slice(0, 200)}\n\n`);
  }
  res.end();
}
