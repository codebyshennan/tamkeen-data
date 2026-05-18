#!/usr/bin/env python3
"""
Wrap ```python / ```sql / ```bash fenced blocks in code-explainer markup with
{% highlight %} and line callouts (for Jekyll + code-explainer.js).

Preserves leading indentation on opening/closing fences (list items).

Usage (from docs/):
  uv run python scripts/wrap_code_explainers.py 2-data-wrangling
  uv run python scripts/wrap_code_explainers.py --dry-run 2-data-wrangling/2.1-sql/basic-operations.md
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

LANG_MAP = {
    "python": "python",
    "py": "python",
    "sql": "sql",
    "bash": "bash",
    "sh": "bash",
}

WRAP_LANGS = frozenset(LANG_MAP.keys())

OPEN_FENCE = re.compile(r"^([ \t]*)```(\w+)\s*\s*$")


@dataclass
class FenceBlock:
    start: int  # index of first char of opening fence line
    end: int  # index after newline following closing fence (or EOF)
    prefix: str
    lang_fence: str
    body: str  # exact inner text (no trailing newline normalization)


def _iter_fenced_blocks(text: str) -> list[FenceBlock]:
    lines = text.splitlines(keepends=True)
    out: list[FenceBlock] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = OPEN_FENCE.match(line)
        if not m:
            i += 1
            continue
        prefix, lang = m.group(1), m.group(2)
        if lang not in WRAP_LANGS:
            i += 1
            continue
        start_char = sum(len(lines[j]) for j in range(i))
        start_line = i
        i += 1
        body_lines: list[str] = []
        close_re = re.compile("^" + re.escape(prefix) + r"```\s*\r?\n?$")
        while i < len(lines):
            if close_re.match(lines[i]):
                body = "".join(body_lines)
                end_line = i
                i += 1
                end_char = sum(len(lines[j]) for j in range(end_line + 1))
                out.append(
                    FenceBlock(
                        start=start_char,
                        end=end_char,
                        prefix=prefix,
                        lang_fence=lang,
                        body=body,
                    )
                )
                break
            body_lines.append(lines[i])
            i += 1
        else:
            # Unclosed fence — leave as-is
            i = start_line + 1
    return out


def _rouge_lang(lang_fence: str) -> str:
    return LANG_MAP.get(lang_fence, lang_fence)


def _lines_list(body: str) -> list[str]:
    if not body:
        return [""]
    lines = body.splitlines()
    return lines if lines else [""]


def _segment_line_ranges(line_count: int) -> list[tuple[int, int]]:
    """1-based inclusive line ranges for callouts."""
    if line_count <= 0:
        return [(1, 1)]
    if line_count <= 14:
        return [(1, line_count)]
    # Aim for ~12–18 lines per band, max 6 bands
    n = min(6, max(2, (line_count + 11) // 14))
    ranges: list[tuple[int, int]] = []
    for k in range(n):
        a = (k * line_count) // n + 1
        b = ((k + 1) * line_count) // n
        if b >= a:
            ranges.append((a, b))
    return ranges or [(1, line_count)]


def _title_for_range(lines: list[str], start: int, end: int) -> str:
    """start/end are 1-based inclusive."""
    segment = lines[start - 1 : end]
    for line in segment:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        if s.startswith("--"):
            s = s[2:].strip()
        if not s:
            continue
        if len(s) > 48:
            s = s[:45].rstrip() + "…"
        return s[0].upper() + s[1:] if s else "Section"
    return "Section"


def _needs_liquid_raw(body: str) -> bool:
    return "{%" in body or "{{" in body


def _build_callouts(lines: list[str]) -> str:
    n = len(lines)
    parts: list[str] = ['<aside class="code-explainer__callouts" aria-label="Code walkthrough">']
    ranges = _segment_line_ranges(n)
    for idx, (lo, hi) in enumerate(ranges):
        tint = (idx % 4) + 1
        title = html.escape(_title_for_range(lines, lo, hi))
        parts.append(f'  <div class="code-callout" data-lines="{lo}-{hi}" data-tint="{tint}">')
        parts.append('    <div class="code-callout__meta">')
        parts.append('      <span class="code-callout__lines"></span>')
        parts.append(f'      <span class="code-callout__title">{title}</span>')
        parts.append("    </div>")
        parts.append('    <div class="code-callout__body">')
        parts.append(f"      <p>Lines {lo}–{hi}: follow this band in the snippet.</p>")
        parts.append("    </div>")
        parts.append("  </div>")
    parts.append("</aside>")
    return "\n".join(parts)


def _indent(text: str, prefix: str) -> str:
    if not prefix:
        return text
    ends_nl = text.endswith("\n")
    lines = text.splitlines()
    body = "\n".join(prefix + line for line in lines)
    return body + ("\n" if ends_nl else "")


def _wrap_block(block: FenceBlock) -> str:
    rouge = _rouge_lang(block.lang_fence)
    lines = _lines_list(block.body)
    inner = block.body.rstrip("\n")
    if _needs_liquid_raw(inner):
        highlight_body = "{% raw %}\n" + inner + "\n{% endraw %}"
    else:
        highlight_body = inner

    code_block = (
        f'{{% highlight {rouge} %}}\n'
        f"{highlight_body}\n"
        "{% endhighlight %}\n"
    )

    callouts = _build_callouts(lines)

    core = (
        '<div class="code-explainer" data-code-explainer>\n'
        '<div class="code-explainer__code">\n\n'
        f"{code_block}"
        "</div>\n"
        f"{callouts}\n"
        "</div>\n"
    )
    return _indent(core, block.prefix)


def transform_file(text: str, _path: Path) -> str:
    blocks = _iter_fenced_blocks(text)
    if not blocks:
        return text
    pieces: list[str] = []
    pos = 0
    for b in blocks:
        pieces.append(text[pos : b.start])
        pieces.append(_wrap_block(b))
        pos = b.end
    pieces.append(text[pos:])
    return "".join(pieces)


def _collect_paths(docs_root: Path, targets: list[str]) -> list[Path]:
    out: list[Path] = []
    for t in targets:
        p = Path(t)
        if not p.is_absolute():
            p = docs_root / p
        p = p.resolve()
        if p.is_file() and p.suffix == ".md":
            out.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*.md")):
                if "assignments" in f.parts:
                    continue
                if f.name == "TODO.md":
                    continue
                out.append(f)
        else:
            raise SystemExit(f"not found: {p}")
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="Wrap fenced code in code-explainer markup.")
    ap.add_argument("targets", nargs="+", help="Markdown file(s) or directories under docs/")
    ap.add_argument("--dry-run", action="store_true", help="Print counts only, do not write")
    args = ap.parse_args()

    docs_root = Path(__file__).resolve().parent.parent
    paths = _collect_paths(docs_root, args.targets)
    changed = 0
    total_blocks = 0
    for path in paths:
        try:
            rel = path.relative_to(docs_root)
        except ValueError:
            rel = path
        text = path.read_text(encoding="utf-8")
        n = len(_iter_fenced_blocks(text))
        new_text = transform_file(text, path)
        if new_text != text:
            total_blocks += n
            changed += 1
            if args.dry_run:
                print(f"would update {rel} ({n} block(s))")
            else:
                path.write_text(new_text, encoding="utf-8")
                print(f"updated {rel} ({n} block(s))")

    if args.dry_run:
        print(f"Done dry-run: {changed} file(s), {total_blocks} fenced block(s).", file=sys.stderr)
    else:
        print(f"Done: {changed} file(s) modified.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
