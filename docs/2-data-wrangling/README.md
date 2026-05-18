# Module 2: Data Wrangling

**After this module, you can query relational databases with SQL, clean and transform messy datasets, explore data distributions and relationships, and build basic data pipelines.**

## Overview

Covers the full data preparation cycle — from structured query language through wrangling, EDA, and data engineering fundamentals. Prerequisites: Module 1.

## If you're catching up

Joining mid-cohort? You do **not** need every page of Module 1 before starting here. Use the table below to triage.

**Must know before Module 2 starts:**

- Python basics — variables, lists, dicts, functions, reading errors → [1.2 Introduction to Python](../1-data-fundamentals/1.2-intro-python/README.md)
- Working with tables in pandas — `DataFrame`, indexing, `groupby`, joins → [1.5 Data analysis with pandas](../1-data-fundamentals/1.5-data-analysis-pandas/README.md)
- One- and two-variable summaries — mean, median, spread, correlation → [1.3 Introduction to statistics](../1-data-fundamentals/1.3-intro-statistics/README.md) (sections [one-variable](../1-data-fundamentals/1.3-intro-statistics/one-variable-statistics.md) and [two-variable](../1-data-fundamentals/1.3-intro-statistics/two-variable-statistics.md))

**Safe to defer (come back when needed):**

- [1.1 Introduction to data analytics](../1-data-fundamentals/1.1-intro-data-analytics/README.md) — context-only; no code that Module 2 builds on.
- [1.4 Linear algebra & NumPy](../1-data-fundamentals/1.4-data-foundation-linear-algebra/README.md) — only required for Module 5 (ML) onward. Read it before [5.3](../5-ml-fundamentals/5.3-classification/README.md), not before 2.x.
- Deeper [probability distributions](../1-data-fundamentals/1.3-intro-statistics/probability-distributions.md) — needed for Module 4 (statistical analysis), not Module 2.

**~2-hour fast track to start Module 2.1:**

1. Skim [1.2 Intro to Python](../1-data-fundamentals/1.2-intro-python/README.md) — if you can read the [functions](../1-data-fundamentals/1.2-intro-python/functions.md) and [data structures](../1-data-fundamentals/1.2-intro-python/data-structures.md) pages without surprises, move on. (~30 min)
2. Read [1.5 pandas — Series](../1-data-fundamentals/1.5-data-analysis-pandas/series.md) and [DataFrame](../1-data-fundamentals/1.5-data-analysis-pandas/dataframe.md), then run [`pandas.ipynb`](../1-data-fundamentals/1.5-data-analysis-pandas/pandas.ipynb) end-to-end. (~60 min)
3. Skim [two-variable statistics](../1-data-fundamentals/1.3-intro-statistics/two-variable-statistics.md) so correlation/covariance language in 2.3 EDA isn't new. (~20 min)

You can start [2.1 SQL](2.1-sql/README.md) without pandas — SQL only needs comfort with tables, rows, and columns. The pandas requirement kicks in at [2.2 Data Wrangling](2.2-data-wrangling/README.md).

*Last reviewed: 2026-05*

## Lesson Path

| Order | Submodule | Focus |
|-------|-----------|-------|
| 2.1 | [SQL](2.1-sql/README.md) | Databases, SELECT, JOINs, aggregations, advanced SQL |
| 2.2 | [Data Wrangling](2.2-data-wrangling/README.md) | Quality, missing values, outliers, transformations |
| 2.3 | [EDA](2.3-eda/README.md) | Distributions, relationships, time-series patterns |
| 2.4 | [Data Engineering](2.4-data-engineering/README.md) | Storage, integration, ETL fundamentals |

After Module 2, continue to [Module 3: Data Visualization](../3-data-visualization/README.md).
