# Documentation Optimization Guidelines

This document provides comprehensive guidelines for creating, maintaining, and optimizing technical documentation for the Tamkeen course: **general writing style**, **setup/tutorial patterns**, and **beginner-level lesson generation** (including a per-submodule anchor map for authors and tooling).

## Core Principles

### 1. Beginner-First Approach
- **Assume no prior knowledge** - Don't assume readers know technical terms
- **Explain the "why"** - Always explain why something is needed, not just how
- **Use simple language** - Avoid jargon; when necessary, define it immediately
- **Provide context** - Help readers understand where this fits in the bigger picture

### 2. Clarity and Structure
- **Clear hierarchy** - Use consistent heading levels (H1 → H2 → H3)
- **Logical flow** - Information should build progressively
- **Scannable content** - Use bullet points, numbered lists, and short paragraphs
- **Visual breaks** - Use horizontal rules, code blocks, and callouts to break up text

### 3. Actionable Content
- **Step-by-step instructions** - Number steps clearly
- **Time estimates** - Tell readers how long tasks will take
- **Prerequisites** - Clearly state what's needed before starting
- **Expected outcomes** - Describe what success looks like

### Learner outcome line (first screen)

For every **lesson page** and **module/submodule README** under `docs/` (modules 1–6), put a single outcome line **immediately under the H1**:

- Lessons: `**After this lesson:** …`
- Submodule `README.md`: `**After this submodule:** …`
- Module root `README.md` (e.g. `1-data-fundamentals/README.md`): `**After this module:** …`

Prefer moving an existing `**Primary outcome:**` from the Overview section up to this position (and delete the duplicate under Overview) so success criteria appear before scrolling.

**Maintainer script:** `scripts/add_after_outcomes.py` migrates `**Primary outcome:**` and fills missing pages with a short generic line. It skips `0-prep/`, `assignments/`, `slides/`, `meta/`, `.venv/`, `TODO.md`, and similar. Re-run after bulk adds:

`uv run python scripts/add_after_outcomes.py` (from `docs/`).

### Jekyll / GFM (lists and inline code)

The site uses **`markdown: GFM`** (GitHub Flavored Markdown). **Inline backticks** (`` `like this` ``) inside **list items** can still be parsed oddly in some edge cases (Rouge, broken sentences). Inside bullets or numbered lists, use **bold** for short tokens (commands, paths) or raw HTML `<code>...</code>` when you need monospace. Outside lists, ordinary backticks are usually fine.

**Figures:** Prefer a blockquote callout instead of an empty `![alt]()` image:

`> **Figure (add screenshot or diagram):** Short description of what to capture.`

**Callout styling (repo-wide):** Ordinary Markdown blockquotes (`>` …) render with the same accent bar, surface, and muted body text as the **inline** lesson callout (see `assets/css/style.css`). No HTML required for that look.

**Lesson callout cards (HTML):** Use `<aside class="lesson-callout …">` when you need a **different layout**—external title above the bar, or Q&A with an italic prompt—rather than a single quoted block. Jekyll passes HTML through.

1. **Inline label** — one bordered block; put the label in bold at the start of the paragraph.

```html
<aside class="lesson-callout lesson-callout--inline" role="note">
  <p><strong>Real-world example:</strong> One or more sentences of body copy.</p>
</aside>
```

2. **External title** — title sits above; only the body has the vertical accent bar.

```html
<aside class="lesson-callout lesson-callout--titled" role="note">
  <p class="lesson-callout__label">Key takeaway</p>
  <div class="lesson-callout__body">
    <p>Supporting text for the takeaway.</p>
  </div>
</aside>
```

3. **Question and answer** — combine `lesson-callout--titled` and `lesson-callout--qa`; use `lesson-callout__question` for the italic prompt.

```html
<aside class="lesson-callout lesson-callout--titled lesson-callout--qa" role="note">
  <p class="lesson-callout__label">Common question</p>
  <div class="lesson-callout__body">
    <p class="lesson-callout__question">Why would I need this?</p>
    <p>Short answer for learners.</p>
  </div>
</aside>
```

Optional **`data-accent`** on the `<aside>` for tint + icon (same layout as typed callouts): `data-accent="success"`, `data-accent="warning"`, or `data-accent="purple"`. Icons and colors are defined in `assets/css/callouts.css`.

**Typed callouts (tint + icon):** `assets/js/callouts.js` scans lesson blockquotes and sets `data-callout-type` when the **first bold label** in the first paragraph matches a known pattern (case-insensitive), e.g. `**Warning:**`, `**Note:**`, `**Note for beginners:**`, `**Time needed:**`, `**Figure (...):**`, `**On screen:**` (same figure-style icon), `**Troubleshooting:**` (warning-style), `**Tip:**`, `**Tip for beginners:**`, `**Teacher's Note**`, `**Real-world example:**`, `**Best Practice:**`, `**Reflect:**`, `**Common Pitfall:**`, etc. Manual override: add `data-callout-type="note|warning|…"` on a `<blockquote>` or `<aside class="lesson-callout">` in HTML.

**Embedded videos (YouTube):** Prefer **standalone clips of about 15 minutes or less** for on-page `<iframe>` embeds. Longer or multi-hour courses belong in **Additional resources** as normal links, not as the primary embedded video.

**Bulk insert (maintainers):** `scripts/add_helpful_videos.py` adds a `## Helpful video` block (iframe + short blurb) to lesson `.md` files that do not already contain `youtube.com/embed`, using longest-prefix path rules under `docs/`. It skips `0-prep/` (hand-curated), `meta/`, root `index.md`, assignments/TODO-style names, `slides/`, and directories such as `.venv`, `_site`, and `node_modules`. From `docs/`: `pnpm run helpful-videos` (use `pnpm run helpful-videos -- --dry-run` to list targets without writing).

## Content Structure

### Essential Sections for Setup/Getting Started Docs

1. **Introduction/What is This?**
   ```
   - Simple, non-technical explanation
   - Real-world analogy if helpful
   - Key benefits/why use it
   - Who is this for?
   ```

2. **System Requirements**
   ```
   - Clear minimum requirements
   - Recommended specifications
   - Platform-specific notes
   - Prerequisites (other software needed)
   ```

3. **Installation/Setup**
   ```
   - Step-by-step instructions
   - Platform-specific tabs if needed
   - Visual placeholders for UI elements
   - Verification steps
   ```

4. **Initial Configuration**
   ```
   - First-time setup
   - Recommended settings
   - Optional vs required steps
   ```

5. **Common Issues & Troubleshooting**
   ```
   - Most frequent problems
   - Clear solutions
   - When to seek help
   ```

6. **Additional Resources**
   ```
   - Official documentation links
   - Learning materials
   - Community support
   ```

## Writing Style Guidelines

### Language and Tone

**Do:**
- ✅ Use active voice ("Click the button" not "The button should be clicked")
- ✅ Use second person ("You will see..." not "The user will see...")
- ✅ Be conversational but professional
- ✅ Use contractions for friendliness ("don't" instead of "do not")
- ✅ Break complex sentences into shorter ones

**Don't:**
- ❌ Use passive voice unnecessarily
- ❌ Assume technical knowledge
- ❌ Use acronyms without defining them first
- ❌ Write overly long paragraphs (aim for 3-5 sentences max)
- ❌ Use vague terms ("stuff", "things", "etc.")

### Technical Terms

**First Use:**
- Define the term immediately: "A DAG (Directed Acyclic Graph) is..."
- Or use inline explanation: "A virtual environment (a separate workspace for your Python packages)..."

**Subsequent Uses:**
- Use the term directly after it's been defined
- Consider a glossary for complex documents

### Code and Commands

**Format:**
```bash
# Always include comments explaining what each command does
# Step 1: Create a new directory
mkdir my-project
cd my-project

# Step 2: Initialize the environment
uv venv
```

**Best Practices:**
- Add comments explaining each step
- Show expected output when helpful
- Include error handling examples
- Provide platform-specific alternatives (Windows vs macOS)

## Visual Elements

### Image Placeholders

**Format:**
```
![Description Placeholder - Shows what the user should see]
```

**When to Use:**
- UI screenshots (login pages, dashboards, settings)
- Step-by-step visual guides
- Error messages or dialogs
- Before/after comparisons
- Architecture diagrams

**Placement:**
- Immediately after the relevant instruction
- Before complex multi-step processes
- To illustrate concepts that are hard to explain in text

### Code Blocks

**Always include:**
- Language identifier for syntax highlighting
- Comments explaining complex logic
- Expected output when relevant
- Error handling examples

**Example:**
```python
# Import required libraries
import pandas as pd

# Read the CSV file
# Replace 'data.csv' with your actual file path
df = pd.read_csv('data.csv')

# Display first few rows
print(df.head())  # Should show first 5 rows
```

### Callouts and Alerts

**Use consistently:**
- `> **Tip:**` - Helpful hints and shortcuts
- `> **Note:**` - Important information to remember
- `> **Warning:**` - Potential issues or gotchas
- `> **Important:**` - Critical information
- `> **Time needed:**` - Time estimates for tasks

## Organization Patterns

### For Setup Guides

1. **What is [Tool]?** - Introduction and context
2. **Why Use It?** - Benefits and use cases
3. **System Requirements** - What you need
4. **Installation** - Step-by-step setup
5. **Initial Configuration** - First-time setup
6. **Basic Usage** - Quick start example
7. **Common Issues** - Troubleshooting
8. **Next Steps** - What to learn next

### For Tutorial Guides

1. **Overview** - What you'll learn
2. **Prerequisites** - What you need to know
3. **Concepts** - Theory and background
4. **Step-by-Step Tutorial** - Hands-on practice
5. **Practice Exercises** - Reinforcement
6. **Summary** - Key takeaways
7. **Further Reading** - Additional resources

## Tamkeen beginner lessons

Use this section when **writing or revising lesson markdown** under `docs/`. **Audience:** learners who meet the module `README` prerequisites but benefit from plain language, motivation, and worked examples before advanced depth.

### Submodule anchor sample map

Each row is one **reference file** sampled for structure, tone, and recurring elements. Use it when onboarding authors or when an LLM needs a concrete anchor per unit (paths are relative to `docs/`).

| Module | Submodule | Sample file |
|--------|-----------|-------------|
| 0 | Prep | `0-prep/pedagogy.md` |
| 1.1 | Intro data analytics | `1-data-fundamentals/1.1-intro-data-analytics/data-collection.md` |
| 1.2 | Intro Python | `1-data-fundamentals/1.2-intro-python/basic-syntax-data-types.md` |
| 1.3 | Intro statistics | `1-data-fundamentals/1.3-intro-statistics/probability-fundamentals.md` |
| 1.4 | Linear algebra / NumPy | `1-data-fundamentals/1.4-data-foundation-linear-algebra/intro-numpy.md` |
| 1.5 | Pandas | `1-data-fundamentals/1.5-data-analysis-pandas/series.md` |
| 2.1 | SQL | `2-data-wrangling/2.1-sql/intro-databases.md` |
| 2.2 | Data wrangling | `2-data-wrangling/2.2-data-wrangling/missing-values.md` |
| 2.3 | EDA | `2-data-wrangling/2.3-eda/distributions.md` |
| 2.4 | Data engineering | `2-data-wrangling/2.4-data-engineering/etl-fundamentals.md` |
| 3.1 | Intro data viz | `3-data-visualization/3.1-intro-data-viz/visualization-principles.md` |
| 3.2 | Advanced data viz | `3-data-visualization/3.2-adv-data-viz/seaborn-guide.md` |
| 3.3 | BI / Tableau | `3-data-visualization/3.3-bi-with-tableau/tableau-basics.md` |
| 3.4 | Data storytelling | `3-data-visualization/3.4-data-storytelling/visual-storytelling.md` |
| 4.1 | Inferential stats | `4-stat-analysis/4.1-inferential-stats/confidence-intervals.md` |
| 4.2 | Hypothesis testing | `4-stat-analysis/4.2-hypotheses-testing/hypothesis-formulation.md` |
| 4.3 | Relationships in data | `4-stat-analysis/4.3-rship-in-data/simple-linear-regression.md` |
| 4.4 | Statistical modelling | `4-stat-analysis/4.4-stat-modelling/logistic-regression.md` |
| 5.1 | Intro ML | `5-ml-fundamentals/5.1-intro-to-ml/what-is-ml.md` |
| 5.2 | Supervised learning 1 | `5-ml-fundamentals/5.2-supervised-learning-1/knn/1-introduction.md` |
| 5.3 | Supervised learning 2 | `5-ml-fundamentals/5.3-supervised-learning-2/neural-networks/1-introduction.md` |
| 5.4 | Unsupervised learning | `5-ml-fundamentals/5.4-unsupervised-learning/clustering.md` |
| 5.5 | Model evaluation | `5-ml-fundamentals/5.5-model-eval/metrics.md` |
| 6 | Capstone | `6-capstone/README.md` |

**Nested ML topics (5.2, 5.3):** algorithm folders often use numbered files (`1-introduction.md`, `2-math-foundation.md`, etc.). Beginner lessons should align with `1-introduction.md` before pointing readers to deeper files.

### Patterns observed in course lessons

- **Module 0:** Explain how the course runs (e.g. flipped classroom, office hours) in short lists; reassure readers new to the format.
- **Openings:** Topic-focused H1; early hooks such as “Why this matters,” “Overview,” or “What is X?” before dense detail; one analogy or real-world scenario where it helps.
- **Structure:** H2/H3 path: concept → definition → example → code → pitfalls → next steps. Horizontal rules (`---`) often separate major sections in Python/Pandas-style pages. Long lessons end with “Next steps” or “Further reading”; capstone material uses phased timelines and deliverables.
- **Pedagogy:** Progressive disclosure (idea in words, then formula, then code); optional links to notebooks, Python Tutor, or AI prompts are enhancements, not prerequisites.
- **Code defaults:** Python unless the topic is SQL, BI, or another stack. Prefer `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `sklearn` with imports at the top and comments.
- **Math (GFM + KaTeX in layout):** `$$...$$` or `\\[...\\]` for display; inline `\\(...\\)` (double the backslashes in Markdown—GFM otherwise strips the backslash from TeX-style open/close delimiters). The layout does not use kramdown’s math preprocessor—preview the built site.
- **Diagrams:** `mermaid` for workflows (EDA, ETL, missing-data mechanisms); keep diagrams readable on GitHub Pages.
- **SQL:** Fenced blocks with `sql` and short comments.
- **BI / tools:** Numbered or YAML-style steps plus screenshot placeholders under `./assets/` as appropriate.
- **Media:** YouTube embeds: wrap the `<iframe>` in `<div class="video-embed">...</div>` (responsive, full width of the lesson column via `assets/css/style.css`), or use `{% include youtube.html id="VIDEO_ID" %}`. Add a one-line caption after the block; images use descriptive alt text (what the figure teaches, not only the filename).

### Reusable section patterns

| Pattern | When to use |
|---------|-------------|
| Prerequisites | Setup-heavy or tool-specific lessons |
| “What is X?” / definition | Every new term |
| Analogy block | Abstract concepts |
| Minimal code first | APIs, sklearn |
| “Good vs bad” or checklist | Hypotheses, metrics |
| “When to use / not use” | Algorithms |
| “Common mistakes” | Algorithms and evaluation |
| Comparison tables | Metrics, chart types |
| Project briefs | Capstone |

### Default outline for a new lesson

Remove sections that do not apply.

```markdown
# [Lesson title]

## [Why this matters / Overview]

- 2–4 sentences and/or bullets: what the learner can do after the lesson.

## Prerequisites

- What they should already know (link to module README or prior lessons).

## Core concepts

### [Concept 1]

- Plain definition; short analogy or example.

## Worked example

- Minimal code or SQL with comments; expected output or what to look for.

## Interpretation (optional)

- How to read numbers, charts, or model output.

## Common pitfalls

- 2–5 bullets: typical mistakes and fixes.

## Practice (optional)

- Small exercises or “try changing this” prompts.

## Next steps

- Links to the next lesson(s) in the submodule.
```

### Module-specific notes

- **Module 1 (all submodules):** For substantive fenced code, use a short title plus **Purpose** and optional **Walkthrough** before the block; avoid back-to-back runnable snippets with no intervening prose. Template and checklist: [`docs/1-data-fundamentals/CODE-BLOCK-PATTERN.md`](../1-data-fundamentals/CODE-BLOCK-PATTERN.md).
- **1.1 (data analytics):** Balance definitions with one runnable or skimmable example; avoid many production-style class stubs in one file unless the README marks it advanced.
- **1.2 (Python):** Encourage running code and reading errors; one main concept per section before stacking libraries.
- **1.3–1.4 and 4.x (stats):** Intuition and simulation/plots before heavy algebra; one main idea per lesson file; cross-link instead of merging topics (e.g. CIs vs tests).
- **1.5 (Pandas):** Show Series/DataFrame output in fenced blocks; spreadsheet mental model (column, row, label).
- **2.x (wrangling / EDA / engineering):** EDA: mermaid workflow plus one Python pipeline; engineering: ETL stages and diagrams before or beside Airflow/DAG snippets.
- **3.x (visualization):** Principles: perception (e.g. pre-attentive vs attentive) before tool syntax; state early if the lesson is code-first or UI-first.
- **5.x (ML):** Lead with problem type and data needs; intros: analogies, “when not to use,” small sklearn examples; metrics: tables plus when each metric misleads (e.g. imbalanced classes).
- **6 (capstone):** Deliverables, timeline, ethics, repo structure; link prior modules instead of re-teaching basics.

## Beginner-Friendly Techniques

### 1. Analogies and Comparisons
```
"Think of a virtual environment like a separate workspace - 
it keeps your project's packages separate from other projects."
```

### 2. Progressive Disclosure
- Start with the simplest explanation
- Add details in subsequent sections
- Use "Advanced" sections for complex topics

### 3. Context Setting
```
"Before we install X, let's understand why we need it..."
"This step is important because..."
```

### 4. Reassurance
```
"Don't worry if this seems complex - we'll break it down step by step."
"This is normal - most beginners find this confusing at first."
```

### 5. Visual Cues
```
"Look for the button that looks like a plug icon..."
"You should see a green checkmark appear..."
```

### 6. Anti-patterns (outline-only pages)

Avoid publishing a lesson that is mostly **topic headings + bullet lists** with no connective prose. Readers who are new to the topic need:

- At least **one sentence per major idea** explaining *why* it matters or *when* it applies.
- **Before** long bullet lists: a short paragraph that frames what the list is for.
- **After** a code block: one or two sentences on what to notice in the output or what to try next.

When revising legacy outline-only pages, prefer **expanding** sections over adding duplicate lists elsewhere.

## Quality Checklist

Before publishing, ensure:

- [ ] **Clarity**: Can a complete beginner follow this?
- [ ] **Completeness**: All steps are included, nothing is assumed
- [ ] **Accuracy**: All commands, links, and instructions are correct
- [ ] **Consistency**: Formatting, terminology, and style are consistent
- [ ] **Visuals**: Image placeholders are included where helpful
- [ ] **Testing**: Instructions have been tested on a clean system
- [ ] **Links**: All external links work and are current
- [ ] **Platform Coverage**: Windows, macOS, and Linux are covered where relevant
- [ ] **Troubleshooting**: Common issues are addressed
- [ ] **Updates**: Version numbers and dates are current

**Lesson pages (Tamkeen markdown under `docs/`):**

- [ ] **One primary outcome** stated in the first two sections
- [ ] **Jargon defined** on first use or linked to an earlier lesson
- [ ] **At least one worked example** (code, SQL, or UI steps) with a clear success criterion
- [ ] **Math** appears after intuition where used; notation matches the submodule
- [ ] **Assets/links** exist or are clearly marked as placeholders
- [ ] **Next steps** point to the next file in the learning path where applicable

## Maintenance Guidelines

### Regular Updates

**Check Quarterly:**
- Software version numbers
- System requirements
- Download links
- External resource links

**Update When:**
- New software versions are released
- UI changes occur in tools
- Common errors change
- Better methods are discovered
- User feedback indicates confusion

### Version Control

**Document:**
- Last updated date
- Software versions tested
- Platform versions tested
- Author/maintainer information

## Examples

### Good Example

```markdown
## What is Python?

Python is a programming language that's perfect for beginners. 
Think of it like learning a new language - but instead of talking 
to people, you're talking to computers!

**Why Python?**
- Easy to read and write (looks almost like English!)
- Used by major companies (Google, Netflix, Instagram)
- Great for data science, web development, and automation
- Huge community of helpful developers

> **Note:** Don't worry if you've never programmed before - 
> Python is designed to be beginner-friendly!
```

### Bad Example

```markdown
## Python

Python is a high-level, interpreted, general-purpose programming 
language. It supports multiple programming paradigms including 
procedural, object-oriented, and functional programming.

Installation:
pip install python
```

## Accessibility Considerations

- **Alt text**: All images should have descriptive alt text
- **Color**: Don't rely on color alone to convey information
- **Font size**: Use standard markdown formatting (don't use tiny text)
- **Screen readers**: Structure content so screen readers can navigate easily
- **Keyboard navigation**: Mention keyboard shortcuts where relevant

## Feedback and Iteration

### Collecting Feedback

- Monitor common questions from users
- Track which sections cause confusion
- Note where users get stuck
- Gather suggestions for improvements

### Continuous Improvement

- Update based on user feedback
- Add examples for common use cases
- Expand troubleshooting sections
- Simplify complex explanations

## Tools and Resources

### Documentation Tools
- Markdown editors (VS Code, Typora, Obsidian)
- Screenshot tools (for creating images)
- Link checkers (to verify external links)
- Grammar checkers (Grammarly, LanguageTool)

### Testing
- Test on clean systems (virtual machines)
- Test on different platforms
- Verify all commands work
- Check all links are accessible

## Conclusion

Great documentation is:
- **Clear** - Easy to understand
- **Complete** - Nothing is missing
- **Current** - Up-to-date information
- **Concise** - No unnecessary fluff
- **Caring** - Written with the reader in mind

Remember: Good documentation doesn't just inform - it empowers readers to succeed.
