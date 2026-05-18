#!/usr/bin/env python3
"""
Insert ## Helpful video sections with YouTube iframes into lesson .md files that
lack youtube.com/embed, using longest-prefix rules. Skips maintainer/generated files.

Run from repo:  python docs/scripts/add_helpful_videos.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent

# (prefix relative to docs/, video_id, iframe title, one-line blurb)
# Sorted longest-first; first match wins.
RULES: list[tuple[str, str, str, str]] = sorted(
    [
        # Module 2 — data wrangling
        (
            "2-data-wrangling/2.1-sql/joins.md",
            "9yeOJ0ZMUYw",
            "SQL Joins Explained",
            "Quick tour of join types in SQL (inner, left, right, full).",
        ),
        (
            "2-data-wrangling/2.1-sql/",
            "27axs9dO7AE",
            "What is SQL?",
            "High-level introduction to SQL and relational databases.",
        ),
        (
            "2-data-wrangling/2.2-data-wrangling/",
            "m1_33jhhiLE",
            "Learn PANDAS in 5 minutes",
            "Pandas DataFrames in a quick walkthrough—useful for cleaning and wrangling.",
        ),
        (
            "2-data-wrangling/2.3-eda/",
            "IFKQLDmRK0Y",
            "Quantiles and Percentiles, Clearly Explained",
            "Summarizing distributions with percentiles—common in exploratory analysis.",
        ),
        (
            "2-data-wrangling/2.4-data-engineering/",
            "eeSLDdz-aLg",
            "Apache Airflow Tutorial for Beginners",
            "DAGs, tasks, and scheduling—conceptual background for ETL-style pipelines.",
        ),
        (
            "2-data-wrangling/",
            "27axs9dO7AE",
            "What is SQL?",
            "Orientation: SQL and structured data for wrangling modules.",
        ),
        # Module 3 — visualization
        (
            "3-data-visualization/3.3-bi-with-tableau/",
            "lTNWfhmurUg",
            "Tableau Public Tutorial Download and Setup",
            "Short Tableau Public install; pair with the written guides in this folder.",
        ),
        (
            "3-data-visualization/3.2-adv-data-viz/",
            "RBSUwFGa6Fk",
            "What is Data Science?",
            "Context for plotting libraries and communication goals in advanced viz.",
        ),
        (
            "3-data-visualization/3.1-intro-data-viz/",
            "RBSUwFGa6Fk",
            "What is Data Science?",
            "Context for how visualization fits into analytics and communication.",
        ),
        (
            "3-data-visualization/3.4-data-storytelling/",
            "RBSUwFGa6Fk",
            "What is Data Science?",
            "Framing insights for others—related context for storytelling.",
        ),
        (
            "3-data-visualization/",
            "RBSUwFGa6Fk",
            "What is Data Science?",
            "Orientation for the course visualization materials.",
        ),
        # Module 4 — statistical analysis
        (
            "4-stat-analysis/4.4-stat-modelling/",
            "NF5_btOaCig",
            "Using Linear Models for t-tests and ANOVA, Clearly Explained",
            "StatQuest: connecting regression-style thinking to common tests.",
        ),
        (
            "4-stat-analysis/4.3-rship-in-data/",
            "zITIFTsivN8",
            "Multiple Regression, Clearly Explained",
            "Overview of regression with several predictors.",
        ),
        (
            "4-stat-analysis/4.2-hypotheses-testing/",
            "0oc49DyA3hU",
            "Hypothesis Testing and The Null Hypothesis, Clearly Explained",
            "Core ideas behind hypothesis tests and the null hypothesis.",
        ),
        (
            "4-stat-analysis/4.1-inferential-stats/",
            "TqOeMYtOc1w",
            "Confidence Intervals, Clearly Explained",
            "StatQuest introduction to confidence intervals.",
        ),
        (
            "4-stat-analysis/",
            "TqOeMYtOc1w",
            "Confidence Intervals, Clearly Explained",
            "Inferential statistics: estimating uncertainty with confidence intervals.",
        ),
        # Module 5 — ML
        (
            "5-ml-fundamentals/5.5-model-eval/",
            "fSytzGwwBVw",
            "Machine Learning Fundamentals: Cross Validation",
            "StatQuest: why cross-validation matters for model evaluation.",
        ),
        (
            "5-ml-fundamentals/5.4-unsupervised-learning/",
            "4b5d3muPQmA",
            "K-means Clustering, Clearly Explained",
            "StatQuest overview of K-means clustering.",
        ),
        (
            "5-ml-fundamentals/5.3-supervised-learning-2/",
            "4qVRBYAdLAo",
            "Supervised Learning: Crash Course AI",
            "Crash Course AI: supervised learning framing (~15 min).",
        ),
        (
            "5-ml-fundamentals/5.2-supervised-learning-1/",
            "4qVRBYAdLAo",
            "Supervised Learning: Crash Course AI",
            "Crash Course AI: supervised learning for classical algorithms.",
        ),
        (
            "5-ml-fundamentals/5.1-intro-to-ml/",
            "4qVRBYAdLAo",
            "Supervised Learning: Crash Course AI",
            "Crash Course AI: how supervised learning fits into ML workflows.",
        ),
        (
            "5-ml-fundamentals/",
            "vP06aMoz4v8",
            "Machine Learning Fundamentals: Sensitivity and Specificity",
            "StatQuest: evaluation metrics in a medical-testing analogy.",
        ),
        # Module 6
        (
            "6-capstone/",
            "RBSUwFGa6Fk",
            "What is Data Science?",
            "End-to-end context for planning and presenting a capstone project.",
        ),
        (
            "additional-resources/",
            "RBSUwFGa6Fk",
            "What is Data Science?",
            "Short orientation to the data science ecosystem.",
        ),
        (
            "1-data-fundamentals/",
            "RBSUwFGa6Fk",
            "What is Data Science?",
            "Short IBM overview of data science roles and lifecycle.",
        ),
        (
            "",
            "RBSUwFGa6Fk",
            "What is Data Science?",
            "Short orientation video for this lesson (default).",
        ),
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

EXCLUDE_DIR_PARTS = frozenset({".venv", "_site", "node_modules"})

SKIP_NAME_SUBSTR = (
    "/.venv/",
    "TODO.md",
    "IMPROVEMENTS.md",
    "CODE_BLOCK",
    "CODE-BLOCK",
    "REVIEW-",
    "GENERATED_",
    "_output.md",
    "module-assignment.md",
    "module-assignment-key.md",
    "module-assignment-student.md",
    "assignments.md",
    "slides/",
    "assets/README.md",
)

SKIP_PREFIXES = ("meta/",)


def should_skip(rel: str) -> bool:
    if rel.startswith(SKIP_PREFIXES):
        return True
    if rel.startswith("0-prep/"):
        return True
    if rel in ("README.md", "index.md", "CLAUDE.md"):
        return True
    for s in SKIP_NAME_SUBSTR:
        if s in rel:
            return True
    return False


IFRAME_TMPL = """## Helpful video

{blurb}

<iframe width="560" height="315" src="https://www.youtube.com/embed/{vid}" title="{title}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
"""


def match_rule(rel: str) -> tuple[str, str, str] | None:
    rel = rel.replace("\\", "/")
    for prefix, vid, title, blurb in RULES:
        if rel == prefix or rel.startswith(prefix):
            return vid, title, blurb
    return None


def insert_block(text: str, blurb: str, vid: str, title: str) -> str:
    if "youtube.com/embed" in text:
        return text
    if "## Helpful video" in text:
        return text
    block = IFRAME_TMPL.format(blurb=blurb, vid=vid, title=title.replace('"', "'"))
    m = re.search(r"^##\s+", text, re.M)
    if m:
        pos = m.start()
        return text[:pos] + block + "\n" + text[pos:]
    return text.rstrip() + "\n\n" + block + "\n"


def iter_markdown_files() -> list[Path]:
    out: list[Path] = []
    for p in DOCS.rglob("*.md"):
        rel = p.relative_to(DOCS)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIR_PARTS:
            continue
        rs = str(rel).replace("\\", "/")
        if should_skip(rs):
            continue
        out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    changed = 0
    skipped_has_embed = 0
    skipped_no_rule = 0
    for path in iter_markdown_files():
        rel = str(path.relative_to(DOCS))
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "youtube.com/embed" in text:
            skipped_has_embed += 1
            continue
        rule = match_rule(rel)
        if not rule:
            skipped_no_rule += 1
            continue
        vid, title, blurb = rule
        new_text = insert_block(text, blurb, vid, title)
        if new_text == text:
            continue
        if args.dry_run:
            print(f"would update: {rel}")
        else:
            path.write_text(new_text, encoding="utf-8")
            print(f"updated: {rel}")
        changed += 1
    print(
        f"done: changed={changed}, skipped_already_embedded={skipped_has_embed}, skipped_no_rule_match={skipped_no_rule}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
