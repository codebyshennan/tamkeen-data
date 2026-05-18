# Assignment Chatbot — Socratic Tutor

## Overview

A chatbot that helps students work through module assignments **without giving away answers**. It knows the question they're on, has the relevant lesson context, and uses an authored hints layer to guide thinking. Built as a sibling to the docsite, embedded as a floating widget.

## Goals

- Reduce the "I'm stuck, what now?" drop-off on multi-choice assignments.
- Encourage students to **explain their reasoning**, not just request the answer.
- Keep the cost-per-student low enough to run for a whole cohort on Vercel's free tier + a single Anthropic/OpenAI key.

## Non-goals

- Not a general course Q&A bot (different retrieval surface, different prompt).
- Not a grader — does not score submissions.
- Not a chat history archive — sessions are ephemeral; no PII stored.

---

## Architecture

```
docs (Jekyll)        chatbot (Next.js on Vercel)         storage
─────────────        ───────────────────────────         ──────────
assignment page  →   /api/chat (Edge runtime)            pgvector (Supabase)
  ↑ widget                ├── retrieve(top-k lesson      │  ├── lessons
  │                       │   chunks + hint block)       │  │   (chunk, embed)
  │                       ├── compose prompt with        │  └── assignments
  │                       │   Socratic system + hints    │      (q_id, hint)
  │                       └── stream via AI SDK
  │                            ↓
  └─ widget displays stream (Markdown, code blocks)
```

### Retrieval (RAG)

**Indexed sources:**
- All `*.md` under `docs/<module>/<submodule>/` except `_assignments/` and `meta/`.
- `module-assignment-hints.md` files (the **authoritative** guidance source for questions).
- Lesson sections chunked by H2/H3 (~500 tokens, 50-token overlap).

**Never indexed:**
- `module-assignment-key.md` and any `*-key.md` — instructors only.
- The assignment questions themselves (the widget passes the current question as conversation context; no need to retrieve it).

**Embedding model:** `text-embedding-3-small` (OpenAI) — cheap, good enough for short technical text.

**Index build:** GitHub Action on every push to `main`. Writes to a Supabase `pgvector` table. ~10 min build, negligible cost.

### Prompt

System prompt (excerpt):

```
You are a tutor for the Tamkeen Data Science course. The student is
working on an assignment question. Your job is to GUIDE their thinking
without revealing the answer.

Rules:
1. Never state the correct option (e.g. "the answer is b").
2. Never paraphrase the correct answer text verbatim.
3. Always start by asking the student what they have considered so far,
   unless they have already shared their reasoning.
4. Use the AUTHORED HINTS below as your primary source of guidance.
   Use LESSON EXCERPTS to add detail only if the student asks "why".
5. If the student asks you to just give the answer, decline and offer
   one more nudge instead.

AUTHORED HINT FOR THIS QUESTION:
{hint_block}

LESSON EXCERPTS (top-3):
{retrieved_chunks}
```

### Guardrails

Pre-send filter on every model response:
1. Load `module-assignment-key.md` for the active question.
2. Extract the bolded correct option text.
3. If response contains a substring match (case-insensitive, normalized), **rewrite** the response: replace the leak with `[withheld — try to reason about it from the lesson]` and append a single Socratic question.

The guardrail is the last line of defense; the system prompt and the hints layer do most of the work.

### Cost model (rough)

- 200 students × 5 questions assisted × 8 turns × 1.5k tokens avg.
- ~12M tokens/month input + ~3M output.
- On Claude Haiku 4.5 (~$1/$5 per Mtok): **~$30/month**.
- Embedding rebuild: ~$1/month.

---

## UX

### Where it lives

- Floating widget (bottom-right) injected on every page under `*/_assignments/*`.
- Hidden elsewhere. The page passes the **current question number** via a `data-q="part1.q3"` attribute on the question container.
- Dedicated full-page view at `/assignments/chat/` for users who prefer it.

### First-message scaffolding

The widget opens prefilled with the question and three quick-pick prompts:
- "I don't understand what the question is asking."
- "I narrowed it down to two options — help me decide."
- "I think the answer is X because Y — am I on the right track?"

The third one is the gold standard. The bot reinforces it by responding *better* to reasoning than to requests.

### Conversation length cap

After 10 turns on the same question, the widget surfaces a **"Take a break and re-read the lesson?"** card with a link to the relevant section. Prevents endless looping.

---

## Implementation plan

| Phase | Scope | Effort |
|---|---|---|
| 0 | Author hints files for one full module (M1) — pilot already exists for Part 1 | 4–6 hr |
| 1 | Vercel Next.js scaffold + AI SDK streaming + system prompt | 1 day |
| 2 | Embedding pipeline (GH Action → Supabase pgvector) | 1 day |
| 3 | Retrieval API + Socratic prompt + hint injection | 1 day |
| 4 | Guardrail filter against answer key | 0.5 day |
| 5 | Widget embed in Jekyll + per-question wiring | 1 day |
| 6 | Pilot with one cohort, collect transcripts (with consent) | 1 week |
| 7 | Author hints for M2–M5 based on patterns in transcripts | 8–12 hr |

Total to live pilot: **~5 days of engineering** + hints authoring in parallel.

---

## Open questions

1. **Hosting the answer-key filter:** keep `*-key.md` in the public repo (current state) or move to a private repo the chatbot pulls from? Private is safer but adds CI complexity.
2. **Telemetry:** do we log (anonymized) which questions get the most chatbot use? Useful for improving lessons; needs consent banner.
3. **Multi-tenant:** if a future cohort has its own assignment variants, does the chatbot need a tenant switch, or do we just deploy per-cohort?
4. **Offline fallback:** if the API key budget is exhausted, does the widget gracefully degrade to "here is the static hints page" or hide entirely?

## Success criteria

- ≥60% of students who open the widget on a question they got wrong **eventually answer it correctly** without ever receiving the literal answer in chat (measured via post-attempt follow-up).
- ≤5% of responses contain leaked answer text (measured by sampling 100 transcripts).
- p95 first-token latency < 1.5 s on Edge runtime.

## Future work

- Voice mode (Whisper STT + cheap TTS) for accessibility.
- "Explain like I'm a beginner" / "stretch me" sliders that change the system prompt.
- Per-cohort instructor dashboard showing top-stuck questions.
