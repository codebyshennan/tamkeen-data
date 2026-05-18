/**
 * Generates _data/lesson_nav_order.yml — ordered lesson filenames per directory
 * for prev/next navigation. Run from docs/: node scripts/generate-lesson-nav-order.mjs
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS_ROOT = path.join(__dirname, "..");

const MODULE_ROOTS = [
  "0-prep",
  "1-data-fundamentals",
  "2-data-wrangling",
  "3-data-visualization",
  "4-stat-analysis",
  "5-ml-fundamentals",
  "6-capstone",
  "additional-resources",
];

const SKIP_DIR_NAMES = new Set([
  "meta",
  "node_modules",
  ".jekyll-cache",
  "vendor",
  ".git",
]);

/** Files that should never appear in learner nav (maintainer/internal files). */
const SKIP_FILES = new Set([
  "TODO.md",
  "CODE-BLOCK-PATTERN.md",
  "CODE_BLOCK_ANNOTATION.md",
  "IMPROVEMENTS.md",
  "REVIEW-ENHANCEMENTS.md",
  "GENERATED_RESOURCES_SUMMARY.md",
  "improvement-plan.md",
]);


/** Curriculum order overrides: dir key -> ordered filenames (relative to that dir). */
const OVERRIDES = {
  "0-prep": [
    "README.md",
    "anaconda.md",
    "python-ds-stack.md",
    "vscode.md",
    "jupyter-notebook.md",
    "dbeaver.md",
    "google-colab.md",
    "tableau.md",
    "snowflake.md",
    "databricks.md",
    "airflow.md",
    "pedagogy.md",
  ],
  "1-data-fundamentals": [
    "README.md",
    "1.1-intro-data-analytics/README.md",
    "1.2-intro-python/README.md",
    "1.3-intro-statistics/README.md",
    "1.4-data-foundation-linear-algebra/README.md",
    "1.5-data-analysis-pandas/README.md",
  ],
  "2-data-wrangling": [
    "README.md",
    "2.1-sql/README.md",
    "2.2-data-wrangling/README.md",
    "2.3-eda/README.md",
    "2.4-data-engineering/README.md",
  ],
  "3-data-visualization": [
    "README.md",
    "3.1-intro-data-viz/README.md",
    "3.2-adv-data-viz/README.md",
    "3.3-bi-with-tableau/README.md",
    "3.4-data-storytelling/README.md",
  ],
  "4-stat-analysis": [
    "README.md",
    "4.1-inferential-stats/README.md",
    "4.2-hypotheses-testing/README.md",
    "4.3-rship-in-data/README.md",
    "4.4-stat-modelling/README.md",
  ],
  "5-ml-fundamentals": [
    "README.md",
    "5.1-intro-to-ml/README.md",
    "5.2-supervised-learning-1/README.md",
    "5.3-supervised-learning-2/README.md",
    "5.4-unsupervised-learning/README.md",
    "5.5-model-eval/README.md",
  ],
  "1-data-fundamentals/1.1-intro-data-analytics": [
    "README.md",
    "data-collection.md",
    "data-privacy.md",
    "data-security.md",
    "workflow-concepts.md",
  ],
  "1-data-fundamentals/1.2-intro-python": [
    "README.md",
    "basic-syntax-data-types.md",
    "data-structures.md",
    "conditions-iterations.md",
    "functions.md",
    "classes-objects.md",
    "modules.md",
    "video-resources.md",
  ],
  "1-data-fundamentals/1.2-intro-python/notebooks": ["README.md"],
  "1-data-fundamentals/1.3-intro-statistics": [
    "README.md",
    "probability-fundamentals.md",
    "probability-distributions.md",
    "probability-distribution-families.md",
    "one-variable-statistics.md",
    "two-variable-statistics.md",
  ],
  "1-data-fundamentals/1.4-data-foundation-linear-algebra": [
    "README.md",
    "intro-numpy.md",
    "ndarray.md",
    "ndarray-basic.md",
    "ndarray-methods.md",
    "boolean-indexing.md",
    "linear-algebra.md",
  ],
  "1-data-fundamentals/1.5-data-analysis-pandas": [
    "README.md",
    "series.md",
    "dataframe.md",
    "data-types-index.md",
    "reindexing-dropping.md",
    "sorting-ranking.md",
    "arithmetic-alignment.md",
    "function-mapping.md",
  ],
  "2-data-wrangling/2.1-sql": [
    "README.md",
    "intro-databases.md",
    "basic-operations.md",
    "joins.md",
    "aggregations.md",
    "advanced-concepts.md",
    "project.md",
  ],
  "2-data-wrangling/2.2-data-wrangling": [
    "README.md",
    "data-quality.md",
    "missing-values.md",
    "outliers.md",
    "transformations.md",
    "project.md",
  ],
  "2-data-wrangling/2.3-eda": [
    "README.md",
    "distributions.md",
    "relationships.md",
    "time-series.md",
    "project.md",
  ],
  "2-data-wrangling/2.4-data-engineering": [
    "README.md",
    "data-storage.md",
    "data-integration.md",
    "etl-fundamentals.md",
    "project.md",
  ],
  "3-data-visualization/3.1-intro-data-viz": [
    "README.md",
    "visualization-principles.md",
    "data-prep-for-visualization.md",
    "matplotlib-basics.md",
    "annotations-and-highlighting.md",
    "troubleshooting-guide.md",
  ],
  "3-data-visualization/3.2-adv-data-viz": [
    "README.md",
    "seaborn-guide.md",
    "plotly-guide.md",
    "time-series-visualization.md",
    "real-world-case-study.md",
  ],
  "3-data-visualization/3.3-bi-with-tableau": [
    "README.md",
    "tableau-basics.md",
    "tableau-case-study.md",
    "looker-studio-case-study.md",
    "powerbi-case-study.md",
    "advanced-analytics.md",
  ],
  "3-data-visualization/3.4-data-storytelling": [
    "README.md",
    "visual-storytelling.md",
    "narrative-techniques.md",
    "case-studies.md",
  ],
  "4-stat-analysis/4.1-inferential-stats": [
    "README.md",
    "population-sample.md",
    "sampling-distributions.md",
    "confidence-intervals.md",
    "p-values.md",
    "parameters-statistics.md",
  ],
  "4-stat-analysis/4.2-hypotheses-testing": [
    "README.md",
    "experimental-design.md",
    "hypothesis-formulation.md",
    "statistical-tests.md",
    "ab-testing.md",
    "results-analysis.md",
  ],
  "4-stat-analysis/4.3-rship-in-data": [
    "README.md",
    "understanding-relationships.md",
    "correlation-analysis.md",
    "simple-linear-regression.md",
    "multiple-linear-regression.md",
    "model-diagnostics.md",
  ],
  "4-stat-analysis/4.4-stat-modelling": [
    "README.md",
    "logistic-regression.md",
    "polynomial-regression.md",
    "model-selection.md",
    "regularization.md",
    "model-interpretation.md",
  ],
  "1-data-fundamentals": ["README.md"],
  "1-data-fundamentals/assignments": ["module-assignment.md", "module-assignment-hints.md", "module-assignment-key.md"],
  "2-data-wrangling/assignments": ["module-assignment-student.md", "module-assignment-key.md"],
  "3-data-visualization/assignments": ["module-assignment.md", "module-assignment-key.md"],
  "4-stat-analysis/assignments": ["module-assignment.md", "module-assignment-key.md"],
  "5-ml-fundamentals/assignments": ["module-assignment.md", "module-assignment-key.md"],
  "5-ml-fundamentals/5.1-intro-to-ml": [
    "README.md",
    "what-is-ml.md",
    "ml-workflow.md",
    "feature-engineering.md",
    "bias-variance.md",
  ],
  "5-ml-fundamentals/5.4-unsupervised-learning": [
    "README.md",
    "clustering.md",
    "k-means-clustering.md",
    "dbscan.md",
    "hierarchical-clustering.md",
    "advanced-clustering.md",
    "pca.md",
    "t-sne.md",
    "tsne-umap.md",
  ],
  "5-ml-fundamentals/5.5-model-eval": [
    "README.md",
    "overfitting-underfitting.md",
    "bias-variance.md",
    "metrics.md",
    "accuracy.md",
    "confusion-matrix.md",
    "precision-recall.md",
    "roc-and-auc.md",
    "cross-validation.md",
    "validation-curves.md",
    "learning-curves.md",
    "early-stopping.md",
    "regularization.md",
    "model-selection.md",
    "hyperparameter-tuning.md",
    "feature-importance.md",
    "sklearn-pipelines.md",
  ],
};

function collectMdFiles(dirAbs) {
  const names = fs.readdirSync(dirAbs, { withFileTypes: true });
  const md = [];
  for (const ent of names) {
    if (!ent.isFile() || !ent.name.endsWith(".md")) continue;
    if (SKIP_FILES.has(ent.name)) continue;
    md.push(ent.name);
  }
  return md;
}

function defaultOrder(files) {
  const rest = files.filter((f) => f !== "README.md").sort((a, b) => a.localeCompare(b));
  const out = [];
  if (files.includes("README.md")) out.push("README.md");
  out.push(...rest);
  return out;
}

function walkModuleDirs(relDir, outDirs) {
  const abs = path.join(DOCS_ROOT, relDir);
  if (!fs.existsSync(abs)) return;
  const hasMd = collectMdFiles(abs).length > 0;
  if (hasMd) outDirs.push(relDir);

  let entries;
  try {
    entries = fs.readdirSync(abs, { withFileTypes: true });
  } catch {
    return;
  }
  for (const ent of entries) {
    if (!ent.isDirectory()) continue;
    if (SKIP_DIR_NAMES.has(ent.name)) continue;
    if (ent.name.startsWith(".")) continue;
    walkModuleDirs(path.join(relDir, ent.name).replace(/\\/g, "/"), outDirs);
  }
}

function buildOrders() {
  const allDirs = [];
  for (const root of MODULE_ROOTS) {
    walkModuleDirs(root, allDirs);
  }
  allDirs.sort((a, b) => a.localeCompare(b));

  const orders = {};
  for (const dirKey of allDirs) {
    const abs = path.join(DOCS_ROOT, dirKey);
    const files = collectMdFiles(abs);
    if (files.length === 0) continue;

    let order;
    if (OVERRIDES[dirKey]) {
      const want = new Set(OVERRIDES[dirKey]);
      const extra = files.filter((f) => !want.has(f)).sort((a, b) => a.localeCompare(b));
      order = OVERRIDES[dirKey].filter((f) => files.includes(f));
      order.push(...extra.filter((f) => !order.includes(f)));
    } else {
      order = defaultOrder(files);
    }
    orders[dirKey] = order;
  }
  return orders;
}

function toYamlString(orders) {
  const lines = [
    "# Auto-generated by scripts/generate-lesson-nav-order.mjs — edit OVERRIDES in that script, then re-run.",
    "# Keys are directory paths relative to docs/ (no leading slash).",
    "",
  ];
  const keys = Object.keys(orders).sort((a, b) => a.localeCompare(b));
  for (const k of keys) {
    lines.push(`${JSON.stringify(k)}:`);
    for (const f of orders[k]) {
      lines.push(`  - ${JSON.stringify(f)}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

const orders = buildOrders();
const outPath = path.join(DOCS_ROOT, "_data", "lesson_nav_order.yml");
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, toYamlString(orders), "utf8");
console.log(`Wrote ${outPath} (${Object.keys(orders).length} directories)`);
