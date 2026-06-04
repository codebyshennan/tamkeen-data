#!/usr/bin/env node
// Bundle per-lesson context for the assignment chatbot.
//
// For every assignment page that the docsite ships (quiz.md or coding.md
// under <module>/<submodule>/assignments/), produce a JSON file containing:
//   - lesson_key       e.g. "1.2-intro-python"
//   - title            human-readable lesson title (from README front matter or H1)
//   - assignment       the stripped student-facing assignment file
//   - hints            the companion hints file
//   - lesson_pages[]   { name, body } for every lesson .md in the submodule
//                      (excludes the assignments/ folder, archives, slides, notebooks)
//
// Output: chatbot/context/<lesson_key>.json
//
// The Edge endpoint loads the file matching the lesson_key sent by the widget
// and prepends it to the system prompt. This means the bot's context is
// strictly scoped to the lesson the student is on — no cross-module leakage.

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');
const DOCS = path.join(ROOT, 'docs');
const OUT = path.join(__dirname, '..', 'context');
// Jekyll data file so the docsite widget can derive its allow-list from the
// same source of truth instead of a hand-maintained copy (prevents drift).
const DATA_OUT = path.join(DOCS, '_data', 'chatbot_lessons.json');

const LESSONS = [
  // Module 1
  { key: '1.1-intro-data-analytics',          submodule: '1-data-fundamentals/1.1-intro-data-analytics',          assignment: 'quiz.md' },
  { key: '1.2-intro-python',                  submodule: '1-data-fundamentals/1.2-intro-python',                  assignment: 'coding.md' },
  { key: '1.3-intro-statistics',              submodule: '1-data-fundamentals/1.3-intro-statistics',              assignment: 'quiz.md' },
  { key: '1.4-data-foundation-linear-algebra', submodule: '1-data-fundamentals/1.4-data-foundation-linear-algebra', assignment: 'coding.md' },
  { key: '1.5-data-analysis-pandas',          submodule: '1-data-fundamentals/1.5-data-analysis-pandas',          assignment: 'coding.md' },
  // Module 2
  { key: '2.1-sql',                           submodule: '2-data-wrangling/2.1-sql',                              assignment: 'coding.md' },
  { key: '2.2-data-wrangling',                submodule: '2-data-wrangling/2.2-data-wrangling',                   assignment: 'coding.md' },
  { key: '2.3-eda',                           submodule: '2-data-wrangling/2.3-eda',                              assignment: 'coding.md' },
  // Module 3
  { key: '3.1-intro-data-viz',                submodule: '3-data-visualization/3.1-intro-data-viz',               assignment: 'coding.md' },
  { key: '3.2-adv-data-viz',                  submodule: '3-data-visualization/3.2-adv-data-viz',                 assignment: 'coding.md' },
  { key: '3.3-bi-with-tableau',               submodule: '3-data-visualization/3.3-bi-with-tableau',              assignment: 'quiz.md' },
  { key: '3.4-data-storytelling',             submodule: '3-data-visualization/3.4-data-storytelling',            assignment: 'quiz.md' },
  // Module 4
  { key: '4.1-inferential-stats',             submodule: '4-stat-analysis/4.1-inferential-stats',                 assignment: 'quiz.md' },
  { key: '4.2-hypotheses-testing',            submodule: '4-stat-analysis/4.2-hypotheses-testing',                assignment: 'coding.md' },
  { key: '4.3-rship-in-data',                 submodule: '4-stat-analysis/4.3-rship-in-data',                     assignment: 'coding.md' },
  { key: '4.4-stat-modelling',                submodule: '4-stat-analysis/4.4-stat-modelling',                    assignment: 'coding.md' },
  // Module 5
  { key: '5.1-intro-to-ml',                   submodule: '5-ml-fundamentals/5.1-intro-to-ml',                     assignment: 'quiz.md' },
  { key: '5.2-supervised-learning-1',         submodule: '5-ml-fundamentals/5.2-supervised-learning-1',           assignment: 'coding.md' },
  { key: '5.3-supervised-learning-2',         submodule: '5-ml-fundamentals/5.3-supervised-learning-2',           assignment: 'coding.md' },
  { key: '5.4-unsupervised-learning',         submodule: '5-ml-fundamentals/5.4-unsupervised-learning',           assignment: 'coding.md' },
  { key: '5.5-model-eval',                    submodule: '5-ml-fundamentals/5.5-model-eval',                      assignment: 'coding.md' },
];

const EXCLUDED_FILENAMES = new Set([
  'REVIEW-ENHANCEMENTS.md',
  'CODE-BLOCK-PATTERN.md',
  'TODO.md',
]);

// Subdirectories never treated as lesson content. assignments/ holds the
// student-facing task (bundled separately) plus answer keys; the rest are
// non-prose. Everything else is descended into so submodules that nest their
// lessons in topic folders (e.g. 5.2/decision-trees, 5.3/neural-networks)
// still contribute their content to the bot's context.
const EXCLUDED_DIRS = new Set(['assignments', 'slides', 'archive', 'assets']);

async function readIfExists(p) {
  try { return await fs.readFile(p, 'utf8'); }
  catch (e) { if (e.code === 'ENOENT') return null; throw e; }
}

function deriveTitle(readme, fallback) {
  if (!readme) return fallback;
  const fmMatch = readme.match(/^---\n([\s\S]*?)\n---/);
  if (fmMatch) {
    const titleLine = fmMatch[1].split('\n').find(l => /^title:/.test(l));
    if (titleLine) return titleLine.replace(/^title:\s*/, '').replace(/^["']|["']$/g, '').trim();
  }
  const h1 = readme.split('\n').find(l => l.startsWith('# '));
  if (h1) return h1.replace(/^#\s+/, '').trim();
  return fallback;
}

async function collectLessonPages(submoduleDir, dir = submoduleDir, pages = []) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (EXCLUDED_DIRS.has(e.name)) continue;
      await collectLessonPages(submoduleDir, full, pages);
      continue;
    }
    if (!e.isFile() || !e.name.endsWith('.md')) continue;
    if (EXCLUDED_FILENAMES.has(e.name)) continue;
    // name is the path relative to the submodule so nested files in different
    // topic folders (each with its own README/1-introduction) stay distinct.
    const name = path.relative(submoduleDir, full);
    pages.push({ name, body: await fs.readFile(full, 'utf8') });
  }
  // Root README first (the submodule overview), then alphabetical by path.
  pages.sort((a, b) => a.name === 'README.md' ? -1 : b.name === 'README.md' ? 1 : a.name.localeCompare(b.name));
  return pages;
}

async function bundleOne(lesson) {
  const submoduleDir = path.join(DOCS, lesson.submodule);
  const asgDir = path.join(submoduleDir, 'assignments');
  const asgFile = path.join(asgDir, lesson.assignment);
  const hintsFile = path.join(asgDir, lesson.assignment.replace(/\.md$/, '-hints.md'));

  const [assignment, hints, readme, pages] = await Promise.all([
    readIfExists(asgFile),
    readIfExists(hintsFile),
    readIfExists(path.join(submoduleDir, 'README.md')),
    collectLessonPages(submoduleDir),
  ]);

  if (!assignment) throw new Error(`Missing assignment file: ${asgFile}`);

  return {
    lesson_key: lesson.key,
    submodule: lesson.submodule,
    title: deriveTitle(readme, lesson.key),
    assignment_filename: lesson.assignment,
    assignment,
    hints,
    lesson_pages: pages,
    generated_at: new Date().toISOString(),
  };
}

async function main() {
  await fs.mkdir(OUT, { recursive: true });
  const summary = [];
  for (const lesson of LESSONS) {
    try {
      const bundle = await bundleOne(lesson);
      const outPath = path.join(OUT, `${lesson.key}.json`);
      await fs.writeFile(outPath, JSON.stringify(bundle, null, 2));
      const tokenEstimate = Math.round(JSON.stringify(bundle).length / 4);
      summary.push({ key: lesson.key, pages: bundle.lesson_pages.length, hints: !!bundle.hints, est_tokens: tokenEstimate });
      console.log(`  ✓ ${lesson.key.padEnd(40)} ${bundle.lesson_pages.length} pages, ~${tokenEstimate.toLocaleString()} tokens`);
    } catch (e) {
      console.error(`  ✗ ${lesson.key}: ${e.message}`);
      process.exitCode = 1;
    }
  }
  // Also write a manifest so the endpoint can validate lesson_keys.
  const generatedAt = new Date().toISOString();
  await fs.writeFile(
    path.join(OUT, 'manifest.json'),
    JSON.stringify({ lessons: summary, generated_at: generatedAt }, null, 2),
  );
  // Mirror the lesson list into the docsite's _data so the widget include
  // gates on the same set the endpoint serves. Generated — do not hand-edit.
  await fs.mkdir(path.dirname(DATA_OUT), { recursive: true });
  await fs.writeFile(
    DATA_OUT,
    JSON.stringify({ lessons: summary, generated_at: generatedAt }, null, 2),
  );
}

main().catch(e => { console.error(e); process.exit(1); });
