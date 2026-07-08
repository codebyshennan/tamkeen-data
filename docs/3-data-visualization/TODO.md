---
published: false
---

# Module 3: maintainer backlog

Internal checklist for **assets and enhancements** beyond the current lesson markdown. Authoring standards live in [`meta/DOCUMENTATION_GUIDELINES.md`](../meta/DOCUMENTATION_GUIDELINES.md).

## Already in the repo (do not duplicate as “missing”)

* Core lesson pages, troubleshooting, FAQ, glossary, best-practices hub, and module `README.md` under `3-data-visualization/`.
* Figure placeholders use blockquote prompts until PNGs are added under each submodule’s `assets/` folder.

## Visual assets (PNG or diagrams)

1. **3.1** — Chart decision tree, hierarchy diagram, color accessibility sketch, Matplotlib figure/axes diagram.
2. **3.2** — Seaborn vs Matplotlib one-pager, Plotly interaction diagram, optional performance flow.
3. **3.3** — Tableau shelf/pane diagram, dashboard layout patterns, LOD overview, connection types.
4. **3.4** — Story arc, narrative frameworks, annotation/layout examples.

## Optional content expansions

* More **worked Python examples** in lessons (animations, multi-panel figures, themes) where a submodule README marks depth as advanced.
* **BI** — Extra calculated-field / SQL examples only if the course schedule allows.
* **Storytelling** — Extra case study templates or rubrics for instructors.

## Tooling / automation

* Replace figure blockquotes with real `![alt](path)` once files exist and alt text matches the teaching goal.
* Regenerate slide decks after editing `slides/data.json` (`node slides/build.js …` from `docs/`).

## Assessment

* Extend [module assignment](https://github.com/codebyshennan/tamkeen-data/blob/main/docs/3-data-visualization/assignments/module-assignment.md) or add a separate project brief if the program requires a portfolio piece.
