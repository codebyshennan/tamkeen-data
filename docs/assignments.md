---
layout: default
title: Assignments
description: All per-lesson assignments from the upstream tamkeen-data source, in one place.
---

# Assignments

These are the assignments shipped by the upstream [`SkillsUnion/tamkeen-data`](https://github.com/SkillsUnion/tamkeen-data) curriculum, mirrored here lesson-by-lesson. Each assignment has a companion **Hints** page that points you at the relevant lesson section and suggests how to think about each question — without giving the answer away.

> **Scope.** Only assignments from the upstream source are listed below. Earlier docsite-authored assignments are kept under each module's `assignments/archive/` for reference but are not part of the active curriculum.

## Module 1 — Data fundamentals

| Lesson | Assignment | Format | Hints |
|---|---|---|---|
| 1.1 — Intro to data analytics | [Quiz](1-data-fundamentals/1.1-intro-data-analytics/assignments/quiz.md) | 20 MCQs | [Hints](1-data-fundamentals/1.1-intro-data-analytics/assignments/quiz-hints.md) |
| 1.2 — Intro to Python | [Coding tasks](1-data-fundamentals/1.2-intro-python/assignments/coding.md) | Functions + class | [Hints](1-data-fundamentals/1.2-intro-python/assignments/coding-hints.md) |
| 1.3 — Intro to statistics | [Quiz](1-data-fundamentals/1.3-intro-statistics/assignments/quiz.md) | 20 MCQs | [Hints](1-data-fundamentals/1.3-intro-statistics/assignments/quiz-hints.md) |
| 1.4 — Linear algebra & NumPy | [Coding tasks](1-data-fundamentals/1.4-data-foundation-linear-algebra/assignments/coding.md) | 4 task groups + bonus | [Hints](1-data-fundamentals/1.4-data-foundation-linear-algebra/assignments/coding-hints.md) |
| 1.5 — Data analysis with pandas | [Coding tasks](1-data-fundamentals/1.5-data-analysis-pandas/assignments/coding.md) | 5 task groups | [Hints](1-data-fundamentals/1.5-data-analysis-pandas/assignments/coding-hints.md) |

## Module 2 — Data wrangling

| Lesson | Assignment | Format | Hints |
|---|---|---|---|
| 2.1 — SQL | [SQL exercises](2-data-wrangling/2.1-sql/assignments/coding.md) | 4 query groups | [Hints](2-data-wrangling/2.1-sql/assignments/coding-hints.md) |
| 2.2 — Data wrangling | [Coding tasks](2-data-wrangling/2.2-data-wrangling/assignments/coding.md) | 5 task groups | [Hints](2-data-wrangling/2.2-data-wrangling/assignments/coding-hints.md) |
| 2.3 — Exploratory data analysis | [Coding tasks](2-data-wrangling/2.3-eda/assignments/coding.md) | 6 task groups | [Hints](2-data-wrangling/2.3-eda/assignments/coding-hints.md) |

## Modules 3 – 5

The upstream `tamkeen-data` repo does not yet ship assignments for the visualization, statistical-analysis, or ML modules. When those land upstream they will be mirrored here.

## How to use these

- **Skim the lesson README first**, then attempt the assignment closed-book to gauge recall.
- It is normal to **look up syntax** — these are not closed-book exams.
- When you get stuck, open the **Hints** page in another tab. Each hint points at a lesson section and offers a way to think about the problem; none of them name the correct option or hand you a finished solution.
- Submit code as a Python script or Jupyter notebook per the assignment instructions.

## For instructors

Solutions for the coding assignments live alongside the source files in [`tamkeen-data`](https://github.com/SkillsUnion/tamkeen-data) under each lesson's `assignment.md` — the docsite copies strip the `## Solutions` block. MCQ answer markers (`_b. ..._`) are preserved in source; the docsite strips them.

Authored-but-not-from-source assignments (the previous monolithic module quizzes, the BI dashboard project, the 4.x practice quizzes) are kept under `<module>/assignments/archive/` rather than deleted.
