# Assignment Chatbot

Socratic tutor that helps students work through assignment questions without giving away answers. Per-page context is **scoped to the lesson's submodule** — the bot only sees that lesson's pages, assignment, and hints.

## Architecture

```
docs (Jekyll, GH Pages)         chatbot (Vercel Edge)         OpenRouter
─────────────────────           ─────────────────────         ──────────
assignment page                  /api/chat                     any model
  ↑ floating widget                ├── validate lesson_key
  │ reads lesson_key from URL      ├── load context JSON
  │ POSTs {lesson_key, messages}   ├── build system prompt
  │                                └── stream to OpenRouter ─▶ ─▶ ─▶
  └─ renders streamed SSE
```

## Per-lesson context bundling

`scripts/bundle.mjs` walks the docsite and produces one JSON file per lesson under `context/`. Each bundle contains:

- `assignment` — the student-facing assignment file (answer markers stripped, solutions stripped)
- `hints` — the companion hints file
- `lesson_pages[]` — every `.md` in the submodule directory (READMEs, concept pages)
- `title`, `submodule`, `lesson_key`

Run after any docs change:

```bash
cd chatbot && pnpm bundle
```

The endpoint imports the bundle matching the request's `lesson_key`. No vector store, no retrieval — context is small enough to fit in the model's window (~13–80k tokens per lesson).

## Deploy

### Prerequisites

1. **OpenRouter account** → grab an API key at https://openrouter.ai/keys.
2. **Vercel account** with the `vercel` CLI installed.

### Steps

```bash
cd chatbot
node scripts/bundle.mjs            # generate context/*.json
pnpm dlx vercel link               # link the dir to a new Vercel project
pnpm dlx vercel env add OPENROUTER_API_KEY   # paste the key when prompted (all envs)
pnpm dlx vercel --prod             # ship it
```

Note the deployed URL. Update `docs/_config.yml` (or your private `docs/_config.local.yml`):

```yaml
chatbot:
  endpoint: "https://your-chatbot.vercel.app/api/chat"
```

Rebuild the docsite (`bundle exec jekyll build`) and the widget will start hitting your endpoint.

### Optional environment variables

| Var | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | _(required)_ | https://openrouter.ai/keys |
| `OPENROUTER_MODEL` | `anthropic/claude-haiku-4.5` | any OpenRouter slug |
| `OPENROUTER_REFERRER` | `https://codebyshennan.github.io/dsai` | sent as `HTTP-Referer` for OpenRouter's leaderboards/attribution |

## Local development

```bash
cd chatbot
pnpm dlx vercel dev    # serves /api/chat on http://localhost:3000
```

Point the widget at it by editing `docs/_config.local.yml`:

```yaml
chatbot:
  endpoint: "http://localhost:3000/api/chat"
```

## How it stays Socratic

The system prompt in `api/chat.js` hard-codes the rules:

1. Never state the correct multiple-choice option.
2. Never write a complete solution; show at most a 1–3 line starter.
3. Use the AUTHORED HINTS first, lesson pages only when "why" is asked.
4. Start by asking what the student tried; reward reasoning over requests for answers.

Because the bundler **never** includes the upstream answer-marker source or any `*-key.md`, the model literally has no answer text to leak. The Socratic prompt prevents it from deriving answers from training knowledge and blurting them.

## Costs (rough)

200 students × 5 questions × ~6 turns × ~30k input + 300 output tokens per turn on Haiku 4.5 (~$1/$5 per MTok):
- Input:  ~180M tokens × $1   = $180/mo
- Output: ~1.8M tokens × $5   = $9/mo
- Total:  **~$190/mo** at full cohort load.

The biggest lever is the per-lesson bundle size — pruning lesson markdown sections that are unlikely to be referenced (slides, video resources, REVIEW notes) would cut input cost meaningfully. The bundler already excludes a few file patterns; tune `EXCLUDED_FILENAMES` for more.

If you swap `OPENROUTER_MODEL` to a cheaper model (e.g. `google/gemini-2.0-flash-001`) costs drop by an order of magnitude with some quality trade-off.

## What's not (yet) here

- Server-side rate limiting (Vercel free tier gets DDoS protection but no per-IP quota). Add Upstash Redis if you expose this widely.
- Answer-key leak detection — moot today (no keys exist in the new MCQ ports). Bring it back if you ever add keys.
- Telemetry. Add `console.log` in `api/chat.js` and pipe via Vercel logs if you want to track usage per lesson.
