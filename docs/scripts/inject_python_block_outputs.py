#!/usr/bin/env python3
"""
Walk course Markdown under docs/ and append stdout (and matplotlib figures) after
```python blocks that do not already show output on the page.

Skips a block when the first line is `# no-output` or `# skip-output`.
Requires the scientific Python stack used in the course (numpy, pandas, etc.).

Usage (from docs/):
  python3 scripts/inject_python_block_outputs.py
  python3 scripts/inject_python_block_outputs.py --dry-run
  python3 scripts/inject_python_block_outputs.py path/to/lesson.md
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import os
import re
import signal
import sys
import textwrap
import traceback
import warnings
from pathlib import Path

# Seconds per block (Unix only; avoids hangs on accidental infinite loops in lesson snippets).
BLOCK_EXEC_TIMEOUT_SEC = 60


class BlockExecTimeout(Exception):
    """Raised when alarm fires during snippet execution."""


def _sigalrm_handler(_signum: int, _frame: object) -> None:
    raise BlockExecTimeout()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODULE_ROOTS = frozenset(
    {
        "0-prep",
        "1-data-fundamentals",
        "2-data-wrangling",
        "3-data-visualization",
        "4-stat-analysis",
        "5-ml-fundamentals",
        "6-capstone",
    }
)

EXCLUDE_DIR_NAMES = frozenset(
    {"meta", "node_modules", ".git", "vendor", ".jekyll-cache", "_site", "slides"}
)

EXCLUDE_FILE_GLOBS = (
    "*_output.md",
    "GENERATED_*",
    "REVIEW-*",
)

EXCLUDE_FILE_NAMES = frozenset(
    {
        "GENERATED_RESOURCES_SUMMARY.md",
    }
)

OUTPUT_FENCE_LANGS = frozenset({"", "text", "console", "plaintext", "output", "txt"})

# Match ```python ... ``` with optional indent or blockquote (`> `) on each line (GFM)
PYTHON_BLOCK = re.compile(
    r"(?ms)^(?P<prefix>[> \t]*)(?:```python|``` py)[ \t]*\n(?P<body>.*?)^(?P=prefix)```[ \t]*(?:\n|$)",
    re.DOTALL,
)

# Jekyll {% highlight python %} … {% endhighlight %} (e.g. inside code-explainer wrappers)
HIGHLIGHT_PYTHON_BLOCK = re.compile(
    r"(?ms)\{%\s*highlight\s+python\s*%\}\s*\n(?P<body>.*?)\n\{%\s*endhighlight\s*%\}",
    re.DOTALL,
)

# Lines that mean "show this" for interactive-style snippets (last expression)
warnings.filterwarnings("ignore", category=FutureWarning)


def _is_under_course_module(path: Path, docs_root: Path) -> bool:
    try:
        rel = path.relative_to(docs_root)
    except ValueError:
        return False
    if not rel.parts:
        return False
    return rel.parts[0] in MODULE_ROOTS


def _should_skip_file(path: Path) -> bool:
    name = path.name
    for pat in EXCLUDE_FILE_GLOBS:
        if pat.startswith("*") and name.endswith(pat[1:]):
            return True
        if pat.endswith("*") and name.startswith(pat[:-1]):
            return True
    if name == "CLAUDE.md":
        return True
    if name in EXCLUDE_FILE_NAMES:
        return True
    return False


def _normalize_markdown_python_indent(code: str) -> str:
    """Strip common leading indent from fenced blocks nested in lists (invalid Python otherwise)."""
    code = code.strip("\n")
    if not code.strip():
        return code
    lines = code.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return code
    indent = min(len(ln) - len(ln.lstrip()) for ln in nonempty)
    if indent <= 0:
        return textwrap.dedent(code)
    stripped = "\n".join(ln[indent:] if len(ln) >= indent else ln for ln in lines)
    return textwrap.dedent(stripped)


def _strip_gfm_blockquote_lines(code: str) -> str:
    """Remove leading `>` from each line (blockquote-wrapped code blocks).

    After `>`, GFM often inserts a single delimiter space; strip that once only so
    `if` / loop body indentation is preserved (do not lstrip the whole line).
    """
    out: list[str] = []
    for line in code.splitlines():
        if line.startswith(">"):
            line = line[1:]
            if line.startswith(" "):
                line = line[1:]
        out.append(line)
    return "\n".join(out)


def _iter_markdown_files(docs_root: Path, paths: list[Path] | None) -> list[Path]:
    if paths:
        out = []
        for p in paths:
            p = p.resolve()
            if p.is_file() and p.suffix == ".md":
                out.append(p)
            elif p.is_dir():
                for f in p.rglob("*.md"):
                    if _is_under_course_module(f, docs_root) and not _should_skip_file(
                        f
                    ):
                        out.append(f)
        return sorted(set(out))

    out = []
    for root in MODULE_ROOTS:
        d = docs_root / root
        if not d.is_dir():
            continue
        for f in d.rglob("*.md"):
            if any(ex in f.parts for ex in EXCLUDE_DIR_NAMES):
                continue
            if _should_skip_file(f):
                continue
            out.append(f)
    return sorted(out)


def _first_fence_after(text: str) -> tuple[str | None, str | None]:
    """Return (language_or_empty, full_first_line) of first ``` fence in text, or (None, None)."""
    t = text.lstrip("\n\r")
    if not t.startswith("```"):
        return None, None
    line = t.split("\n", 1)[0]
    rest = line[3:].strip()
    return rest, line


def _has_following_output_or_figure(content: str, end_pos: int) -> bool:
    """After closing ``` of a python block, check for output fence or figure."""
    tail = content[end_pos:]
    # skip blank lines
    i = 0
    while i < len(tail) and tail[i] in "\n\r \t":
        i += 1
    if i >= len(tail):
        return False
    sub = tail[i:]
    if sub.startswith("!["):
        return True
    if sub.startswith("<figure"):
        return True
    lang, _ = _first_fence_after(sub)
    if lang is None:
        return False
    if lang in OUTPUT_FENCE_LANGS:
        return True
    return False


def _init_namespace(_cwd: Path) -> dict:
    """Shared imports for lesson code; chdir applied by caller."""
    ns: dict = {"__name__": "__main__", "__builtins__": __builtins__}

    def _try_exec(code: str) -> None:
        try:
            exec(code, ns, ns)
        except Exception:
            pass

    import math
    import random
    import re as _re
    from io import StringIO

    ns.update(
        {
            "math": math,
            "random": random,
            "_re": _re,
            "StringIO": StringIO,
        }
    )

    _try_exec("import numpy as np")
    _try_exec("import pandas as pd")
    _try_exec(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"
    )
    _try_exec("import seaborn as sns")
    _try_exec("from scipy import stats")
    # statsmodels: defer to lesson imports (eager import can pull in test harness noise on some setups).
    # sklearn: lesson blocks almost always import explicitly.

    return ns


def _run_python_block(code: str, ns: dict) -> tuple[str, list[Path], list[str]]:
    """
    Execute code; if last statement is an expression (Jupyter-style), print its value.
    Returns (stdout text, list of saved figure paths, list of per-figure title strings).
    Title strings are extracted from matplotlib suptitle or first-axes title (may be empty).
    """
    code = _normalize_markdown_python_indent(code)
    code = _strip_gfm_blockquote_lines(code)
    code = code.strip("\n")
    if not code.strip():
        return "", [], []

    # Close any stale figures left by a previously-failed block (prevents cross-block leakage).
    try:
        import matplotlib.pyplot as _plt_cleanup

        if _plt_cleanup.get_fignums():
            _plt_cleanup.close("all")
    except Exception:
        pass

    buf = io.StringIO()
    fig_paths: list[Path] = []
    fig_titles: list[str] = []

    tree = ast.parse(code)
    body = tree.body
    if not body:
        return "", [], []

    last = body[-1]
    last_is_expr = isinstance(last, ast.Expr)

    with contextlib.redirect_stdout(buf):
        try:
            if last_is_expr and len(body) == 1:
                val = eval(compile(ast.Expression(last.value), "<md>", "eval"), ns)
                if val is not None:
                    print(val)
            elif last_is_expr and len(body) > 1:
                exec(
                    compile(ast.Module(body[:-1], type_ignores=[]), "<md>", "exec"), ns
                )
                val = eval(compile(ast.Expression(last.value), "<md>", "eval"), ns)
                if val is not None:
                    print(val)
            else:
                exec(compile(tree, "<md>", "exec"), ns)
        except Exception:
            raise

    # Save any figures
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is not None and plt.get_fignums():
        base = ns.get("_fig_base", "fig")
        assets = Path("assets")
        assets.mkdir(exist_ok=True)
        for fig_num in list(plt.get_fignums()):
            fig = plt.figure(fig_num)
            try:
                fig.tight_layout()
            except Exception:
                pass
            idx = ns.get("_fig_counter", 0) + 1
            ns["_fig_counter"] = idx
            fname = f"{base}_fig_{idx}.png"
            out = assets / fname
            fig.savefig(out, dpi=150, bbox_inches="tight")
            fig_paths.append(out)
            # Extract title: suptitle first, then first-axes title
            title = ""
            try:
                title = fig.get_suptitle() or ""
                if not title:
                    axes = fig.get_axes()
                    if axes:
                        title = axes[0].get_title() or ""
            except Exception:
                pass
            fig_titles.append(title)
        plt.close("all")

    return buf.getvalue(), fig_paths, fig_titles


def _parse_fig_caption_pragma(code: str) -> str | None:
    """Return caption text from `# fig-caption: <text>` first-line pragma, or None."""
    first = code.lstrip().split("\n", 1)[0].strip()
    prefix = "# fig-caption:"
    if first.startswith(prefix):
        return first[len(prefix) :].strip()
    return None


def _skip_block_pragma(code: str) -> bool:
    first = code.lstrip().split("\n", 1)[0].strip()
    if first in ("# no-output", "# skip-output"):
        return True
    return False


def _strip_liquid_raw_if_wrapped(code: str) -> str:
    """Unwrap {% raw %}…{% endraw %} from Jekyll highlight bodies (code-explainer)."""
    c = code.strip()
    if c.startswith("{% raw %}"):
        end = c.rfind("{% endraw %}")
        if end != -1:
            return c[len("{% raw %}") : end].strip("\n")
    return code


def process_markdown_file(
    path: Path, dry_run: bool, log: argparse.FileType | None
) -> bool:
    """Return True if file was modified."""
    docs_root = path.parent
    while docs_root.name != "docs" and docs_root != docs_root.parent:
        docs_root = docs_root.parent
    if docs_root.name != "docs":
        docs_root = path.parent

    text = path.read_text(encoding="utf-8")
    cwd = path.parent

    def logmsg(msg: str) -> None:
        if log:
            log.write(msg + "\n")
        else:
            print(msg, file=sys.stderr)

    ns = _init_namespace(cwd)
    ns["_fig_base"] = path.stem
    ns["_fig_counter"] = 0

    replacements: list[tuple[int, str]] = []

    modified = False

    def _process_python_block(code: str, start: int, end: int) -> None:
        code = _strip_liquid_raw_if_wrapped(code)
        if _skip_block_pragma(code):
            return
        # Always execute so the shared namespace stays populated for downstream
        # blocks; only suppress the output insertion when the page already has it.
        has_output = _has_following_output_or_figure(text, end)

        # Capture pragma caption before code is mutated by execution helpers
        pragma_caption = _parse_fig_caption_pragma(code)

        old_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if hasattr(signal, "SIGALRM"):
                        old_alarm = signal.signal(signal.SIGALRM, _sigalrm_handler)
                        signal.alarm(BLOCK_EXEC_TIMEOUT_SEC)
                        try:
                            stdout, figs, fig_titles = _run_python_block(code, ns)
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_alarm)
                    else:
                        stdout, figs, fig_titles = _run_python_block(code, ns)
            except BlockExecTimeout:
                logmsg(
                    f"{path}:{text[:start].count(chr(10)) + 1} block skip (timeout {BLOCK_EXEC_TIMEOUT_SEC}s)"
                )
                return
            except SystemExit as e:
                logmsg(
                    f"{path}:{text[:start].count(chr(10)) + 1} block skip (SystemExit: {e})"
                )
                return
            except Exception as e:
                logmsg(
                    f"{path}:{text[:start].count(chr(10)) + 1} block skip ({e.__class__.__name__}: {e})"
                )
                if log:
                    traceback.print_exc(file=log)
                return
        finally:
            os.chdir(old_cwd)

        if has_output:
            return

        parts: list[str] = []
        for fig, mpl_title in zip(figs, fig_titles):
            rel = fig.as_posix()
            if cwd != path.parent:
                rel = os.path.relpath(fig, cwd)
            # Resolve figure number from filename suffix (e.g. _fig_3.png → 3)
            fig_num_match = re.search(r"_fig_(\d+)\.png$", fig.name)
            fig_num = (
                int(fig_num_match.group(1))
                if fig_num_match
                else ns.get("_fig_counter", 1)
            )
            caption_text = pragma_caption or mpl_title or "Generated visualization"
            caption = f"Figure {fig_num}: {caption_text}"
            parts.append(
                f'\n\n<figure>\n<img src="{rel}" alt="{path.stem}" />\n'
                f"<figcaption>{caption}</figcaption>\n</figure>\n"
            )
        if stdout.strip():
            parts.append(f"\n```\n{stdout.rstrip()}\n```\n")
        if not parts:
            return

        insertion = "".join(parts)
        replacements.append((end, insertion))

    for m in PYTHON_BLOCK.finditer(text):
        _process_python_block(m.group("body"), m.start(), m.end())

    for m in HIGHLIGHT_PYTHON_BLOCK.finditer(text):
        _process_python_block(m.group("body"), m.start(), m.end())

    if not replacements:
        return False

    new_text = text
    for end_pos, ins in sorted(replacements, key=lambda x: x[0], reverse=True):
        new_text = new_text[:end_pos] + ins + new_text[end_pos:]
        modified = True

    if modified and not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return modified


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inject Python stdout/figures after ```python blocks."
    )
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Markdown files or dirs (default: all course modules)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    ap.add_argument(
        "--log", type=argparse.FileType("w"), help="Write skip/trace details to file"
    )
    args = ap.parse_args()

    docs_root = Path(__file__).resolve().parent.parent
    files = _iter_markdown_files(docs_root, list(args.paths) if args.paths else None)
    changed = 0
    for f in files:
        try:
            if process_markdown_file(f, args.dry_run, args.log):
                print(f"updated: {f.relative_to(docs_root)}")
                changed += 1
        except OSError as e:
            print(f"error: {f}: {e}", file=sys.stderr)
    print(
        f"Done. {changed} file(s) {'would be ' if args.dry_run else ''}modified.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
