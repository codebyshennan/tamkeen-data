# Catch-up panel — author template

A reusable block for module README pages so mid-joiners can triage Module N's prerequisites without reading every earlier module top-to-bottom. The goal is **what they need from previous modules to start *this* one**, not a generic curriculum summary.

## When to add it

- Drop into `docs/<N>-<module>/README.md` directly under the `## Overview` section.
- Add to every module from 2 onward. Module 1 doesn't need one (Module 0 is the only prerequisite).
- Re-review the panel when the cohort moves into the next module — what's "must know" vs "safe to defer" shifts as the curriculum runs.

## Drafting checklist

For the module you're writing the panel for:

1. **Open each submodule's first lesson** and list the concrete prerequisites it actually uses (functions it calls, syntax it expects, vocabulary it introduces without redefining). That's the *Must know* list — keep it to 3–5 concrete items, each linked to the specific page that teaches it, not to a top-level README.
2. **Identify earlier-module lessons whose concepts don't appear in this module.** Those go in *Safe to defer*, with a note about which later module makes them required.
3. **Build the fast track as a sequence** — read X, read Y, run Z notebook. Cap total time at ~2 hours; if it's longer, the prerequisites are too broad and the module probably has hidden coupling worth fixing instead.
4. **Update `Last reviewed` to the current month** when you touch any of the three lists. A stale date is more harmful than no date — it tells learners the panel is unverified.

## Template

Copy the block below, replace the bracketed placeholders, and paste under `## Overview` in the module README.

```markdown
## If you're catching up

Joining mid-cohort? You do **not** need every page of Module [N-1] before starting here. Use the table below to triage.

**Must know before Module [N] starts:**

- [Concrete skill] — [one-line scope] → [link to the specific lesson page, not the module README]
- [Concrete skill] — [one-line scope] → [link]
- [Concrete skill] — [one-line scope] → [link]

**Safe to defer (come back when needed):**

- [Lesson link] — [one-line reason it can wait, and which later module needs it].
- [Lesson link] — [one-line reason].

**~[time] fast track to start Module [N.1]:**

1. [Action — read / skim / run] [link]. (~[minutes])
2. [Action] [link]. (~[minutes])
3. [Action] [link]. (~[minutes])

[Optional escape hatch: which submodule(s) in this module can start without the full fast track. E.g. "You can start 2.1 SQL without pandas — the pandas requirement kicks in at 2.2."]

*Last reviewed: YYYY-MM*
```

## Worked example

See `docs/2-data-wrangling/README.md` for the panel produced by this template for Module 2 (the cohort's current module).

## Per-lesson companion: front matter

The catch-up panel works best when individual lesson pages also expose a learner's-eye estimate. The Jekyll layout already supports two front-matter fields (see project `CLAUDE.md`):

```yaml
---
reading_minutes: 15
objectives:
  - "Load a CSV into a DataFrame and check its shape, dtypes, and head."
  - "Select rows and columns by label and by position."
  - "Aggregate with groupby and a reducer."
---
```

Add these to the entry-point page of any lesson a catch-up panel links to from the **Must know** list — that way a mid-joiner can see "20 min, 3 outcomes" before they commit to reading, and the panel's time estimate stays honest.

Keep `objectives` to 3–5 items, written as verifiable actions (what they can *do* after, not what they will *learn about*).
