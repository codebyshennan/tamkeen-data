---
layout: default
title: Assignments
description: All module and lesson assignments in one place.
---

# Assignments

Every module ends with a graded assignment, and a few submodules have additional practice quizzes or projects. Use the table below to jump straight to the one you need. Each module assignment is multiple choice unless noted; estimated time is for a careful first pass.

> **Heads up — known issue:** assignment files live under `_assignments/` directories, which Jekyll currently excludes from the build. The links below resolve in the source repo but 404 on the live site. Tracked as a follow-up (see [_How to fix_](#how-to-fix-broken-assignment-links) below).

## Module assignments

| Module | Assignment | Format | Time |
|---|---|---|---|
| 1 — Data fundamentals | [Module 1 assignment](1-data-fundamentals/_assignments/module-assignment.md) | MCQ, 100 pts | 60–90 min |
| 2 — Data wrangling | [Module 2 assignment](2-data-wrangling/_assignments/module-assignment-student.md) | MCQ + short answer | 60–90 min |
| 3 — Data visualization | [Module 3 assignment](3-data-visualization/_assignments/module-assignment.md) | MCQ | 45–60 min |
| 4 — Statistical analysis | [Module 4 assignment](4-stat-analysis/_assignments/module-assignment.md) | MCQ | 60–90 min |
| 5 — Machine learning | [Module 5 assignment](5-ml-fundamentals/_assignments/module-assignment.md) | Two-part: MCQ + applied | 90–120 min |

## Submodule practice and projects

| Submodule | Assignment |
|---|---|
| 3.3 — BI with Tableau | [Build your first BI dashboard](3-data-visualization/3.3-bi-with-tableau/_assignments/bi-dashboard-project.md) — 2–3 hr project |
| 4.1 — Inferential statistics | [Practice quiz](4-stat-analysis/4.1-inferential-stats/_assignments/practice-quiz.md) |
| 4.2 — Hypothesis testing | [Practice quiz](4-stat-analysis/4.2-hypotheses-testing/_assignments/practice-quiz.md) |
| 4.3 — Relationships in data | [Practice quiz](4-stat-analysis/4.3-rship-in-data/_assignments/practice-quiz.md) |
| 4.4 — Statistical modelling | [Practice quiz](4-stat-analysis/4.4-stat-modelling/_assignments/practice-quiz.md) |

## How to use these

- **Skim the lesson READMEs first**, then attempt the assignment closed-book to gauge recall.
- It is normal to **look up syntax** you have not memorized; this is not a closed-book exam.
- Each module assignment ships with a sibling **answer key** (visible in the repo) for instructors — students should attempt the assignment first.
- Stuck on a question? Look for the **hints file** (where available) next to the assignment, which points to the lesson section that covers the concept and offers a way to think about it without giving the answer.

## For instructors

- Answer keys are at `module-assignment-key.md` in each `_assignments/` folder.
- Quiz items follow the style in `docs/meta/DOCUMENTATION_GUIDELINES.md`.
- Module 1 has a worked example of the **hints** companion file at [`1-data-fundamentals/_assignments/module-assignment-hints.md`](1-data-fundamentals/_assignments/module-assignment-hints.md); reuse the pattern for other modules.

## How to fix broken assignment links

Underscore-prefixed directories (`_assignments`) are excluded by Jekyll's default rules; that is why the assignment files do not appear under `_site/`. Two options:

1. **Rename** every `_assignments/` directory to `assignments/` (≈10 dirs, ~80 references in markdown/HTML). Cleanest, but a large mechanical change.
2. **Register as a collection** in `_config.yml`:

   ```yaml
   collections:
     assignments:
       output: true
       permalink: /:path/
   ```

   then move files into a top-level `_assignments/` collection. Less invasive but changes URLs.

Until then, the file paths in the tables above resolve when reading the source repo on GitHub.
