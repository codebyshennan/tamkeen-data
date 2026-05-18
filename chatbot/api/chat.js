// Edge function: proxies chat completions to OpenRouter with per-lesson
// context injected as the system prompt. The widget POSTs:
//   { lesson_key: "1.2-intro-python", messages: [{role, content}, ...] }
//
// We load the matching bundle from chatbot/context/<lesson_key>.json,
// prepend the Socratic system prompt + lesson context as a single system
// message, and stream the assistant response back as Server-Sent Events.

import manifest from '../context/manifest.json' with { type: 'json' };

const ALLOWED = new Set(manifest.lessons.map(l => l.key));
const MODEL = process.env.OPENROUTER_MODEL || 'anthropic/claude-haiku-4.5';
const MAX_HISTORY_TURNS = 20;
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

export const config = { runtime: 'edge' };

const SYSTEM_PROMPT = `
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
`.trim();

function buildSystemMessage(bundle) {
  const pages = bundle.lesson_pages
    .map(p => `### ${p.name}\n\n${p.body}`)
    .join('\n\n---\n\n');
  return [
    SYSTEM_PROMPT,
    `\n\n=== LESSON: ${bundle.title} (${bundle.lesson_key}) ===\n`,
    `=== ASSIGNMENT (${bundle.assignment_filename}) ===\n\n${bundle.assignment}`,
    bundle.hints ? `\n\n=== AUTHORED HINTS ===\n\n${bundle.hints}` : '',
    `\n\n=== LESSON PAGES ===\n\n${pages}`,
  ].join('');
}

const bundleCache = new Map();
async function loadBundle(lessonKey) {
  if (bundleCache.has(lessonKey)) return bundleCache.get(lessonKey);
  const mod = await import(`../context/${lessonKey}.json`, { with: { type: 'json' } });
  bundleCache.set(lessonKey, mod.default);
  return mod.default;
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', 'access-control-allow-origin': '*' },
  });
}

export default async function handler(req) {
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'access-control-allow-origin': '*',
        'access-control-allow-methods': 'POST, OPTIONS',
        'access-control-allow-headers': 'content-type',
      },
    });
  }
  if (req.method !== 'POST') return jsonResponse(405, { error: 'POST only' });

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) return jsonResponse(500, { error: 'OPENROUTER_API_KEY not configured' });

  let body;
  try { body = await req.json(); }
  catch { return jsonResponse(400, { error: 'invalid JSON body' }); }

  const { lesson_key, messages } = body;
  if (!lesson_key || typeof lesson_key !== 'string') return jsonResponse(400, { error: 'lesson_key required' });
  if (!ALLOWED.has(lesson_key)) return jsonResponse(400, { error: `unknown lesson_key: ${lesson_key}` });
  if (!Array.isArray(messages) || messages.length === 0) return jsonResponse(400, { error: 'messages array required' });

  const trimmed = messages
    .filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .slice(-MAX_HISTORY_TURNS);
  if (trimmed.length === 0) return jsonResponse(400, { error: 'no valid messages' });

  let bundle;
  try { bundle = await loadBundle(lesson_key); }
  catch (e) { return jsonResponse(500, { error: `context load failed: ${e.message}` }); }

  const upstream = await fetch(OPENROUTER_URL, {
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

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => '');
    return jsonResponse(502, { error: `upstream ${upstream.status}`, detail: text.slice(0, 500) });
  }

  // Re-stream OpenRouter SSE directly to the client.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache, no-transform',
      'connection': 'keep-alive',
      'access-control-allow-origin': '*',
    },
  });
}
