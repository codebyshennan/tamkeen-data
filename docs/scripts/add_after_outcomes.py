#!/usr/bin/env python3
"""
Add **After this lesson:** / **After this submodule:** / **After this module:**
below the first H1, migrating from **Primary outcome:** where present.

Run from repo: uv run python docs/scripts/add_after_outcomes.py
Or: python docs/scripts/add_after_outcomes.py (from docs/)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent

SKIP_DIR_PARTS = frozenset(
    {
        "assignments",
        "slides",
        "meta",
        "_site",
        ".jekyll-cache",
        ".venv",
        "node_modules",
        "site-packages",
    }
)
SKIP_FILES = frozenset(
    {
        "CODE-BLOCK-PATTERN.md",
        "CODE_BLOCK_ANNOTATION.md",
        "REVIEW-ENHANCEMENTS.md",
        "DOCUMENTATION_GUIDELINES.md",
        "TODO.md",
        "IMPROVEMENTS.md",
        "GENERATED_RESOURCES_SUMMARY.md",
        "assignments.md",
    }
)
SKIP_PREFIXES = ("module-assignment",)


def should_skip(path: Path) -> bool:
    if any(p in SKIP_DIR_PARTS for p in path.parts):
        return True
    if path.name == "README.md" and path.parent.name == "assets":
        return True
    if path.name.endswith("_output.md"):
        return True
    if path.name in SKIP_FILES:
        return True
    if path.suffix != ".md":
        return True
    if path.name.startswith(SKIP_PREFIXES):
        return True
    # 0-prep already done
    if "0-prep" in path.parts:
        return True
    # Only course modules 1–6
    rel = path.relative_to(DOCS)
    if not rel.parts or not rel.parts[0][0].isdigit():
        return True
    return False


def has_after_line(text: str) -> bool:
    """True if an After-this line already exists near the top (after optional front matter)."""
    t = text
    if t.startswith("---"):
        end = t.find("\n---", 3)
        if end != -1:
            t = t[end + 4 :]
    for line in t.splitlines()[:25]:
        if re.match(r"^\*\*After this (lesson|guide|module|submodule|folder|reading)", line):
            return True
        if re.match(r"^After this guide,?\s", line, re.I):
            return True
    return False


def label_for_path(path: Path) -> str:
    """Pick Markdown label key for **After this …:**"""
    if path.name != "README.md":
        return "lesson"
    parent = path.parent.name
    # Module root: 1-data-fundamentals, 2-data-wrangling, …
    if re.match(r"^\d+-", parent) and parent.count("-") >= 1 and not re.match(r"^\d+\.\d+", parent):
        return "module"
    # Submodule: 1.1-intro-…
    if re.match(r"^\d+\.\d+", parent):
        return "submodule"
    return "submodule"


def clean_outcome_text(text: str) -> str:
    t = text.strip()
    for prefix in (
        "After this submodule you can ",
        "After this submodule ",
        "After this module you can ",
        "After this module ",
    ):
        if t.startswith(prefix):
            t = t[len(prefix) :].strip()
            if t and t[0].islower():
                t = t[0].upper() + t[1:]
            break
    return t


def insert_after_h1(content: str, line: str) -> str:
    """Insert `line` (with newline) immediately after the first # heading line."""
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    inserted = False
    for i, ln in enumerate(lines):
        out.append(ln)
        if not inserted and ln.startswith("# ") and not ln.startswith("# #"):
            # Skip if next non-empty is already our line
            out.append("\n")
            out.append(line.rstrip() + "\n")
            inserted = True
    if not inserted:
        return content
    return "".join(out)


def migrate_primary_outcome(content: str, path: Path) -> tuple[str, bool]:
    m = re.search(r"^\*\*Primary outcome:\*\*\s*(.+)$", content, re.MULTILINE)
    if not m:
        return content, False
    outcome = clean_outcome_text(m.group(1))
    kind = label_for_path(path)
    label = f"**After this {kind}:**"
    new_line = f"{label} {outcome}"

    new_content = re.sub(
        r"^\*\*Primary outcome:\*\*\s*.+$",
        "",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    # Collapse 3+ newlines under Overview
    new_content = re.sub(r"\n{3,}", "\n\n", new_content)

    if has_after_line(new_content):
        return content, False

    new_content = insert_after_h1(new_content, new_line)
    return new_content, True


def add_generic_after(content: str, path: Path) -> tuple[str, bool]:
    """For pages without Primary outcome: add a short outcome line from the H1 title."""
    if has_after_line(content):
        return content, False

    t = content
    if t.startswith("---"):
        end = t.find("\n---", 3)
        if end != -1:
            t = t[end + 4 :]

    m = re.search(r"^#\s+(.+)$", t, re.MULTILINE)
    if not m:
        return content, False
    title = m.group(1).strip()
    kind = label_for_path(path)
    if kind == "module":
        body = (
            f"you have a map of this module’s units, prerequisites, and how they connect to the rest of the course—"
            f"then work through each submodule in order unless your instructor says otherwise."
        )
    elif kind == "submodule":
        body = (
            f"you can use the lessons linked below and complete the exercises that match **{title}** in your course schedule."
        )
    else:
        body = (
            f"you can explain the core ideas in “{title}” and reproduce the examples here in your own notebook or environment."
        )

    line = f"**After this {kind}:** {body}"
    return insert_after_h1(content, line), True


def main() -> int:
    changed = 0
    paths = sorted(DOCS.rglob("*.md"))
    for path in paths:
        if should_skip(path):
            continue
        text = path.read_text(encoding="utf-8")
        orig = text

        text, did = migrate_primary_outcome(text, path)
        if did and text != orig:
            path.write_text(text, encoding="utf-8")
            changed += 1
            print(f"migrated: {path.relative_to(DOCS)}")
            continue

        if has_after_line(text):
            continue

        text, did2 = add_generic_after(text, path)
        if did2 and text != orig:
            path.write_text(text, encoding="utf-8")
            changed += 1
            print(f"generic: {path.relative_to(DOCS)}")

    print(f"Done. Updated {changed} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
