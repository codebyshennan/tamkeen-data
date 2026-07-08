# E-commerce Data Wrangling Project: From Raw Data to Actionable Insights

**After this lesson:** You deliver a cleaned, documented dataset (or notebook) that shows how you assessed quality, handled **missing values** and **outliers**, and validated results, aligned with the [wrangling lessons](./).

## Helpful video

Pandas DataFrames in a quick walkthrough, useful for cleaning and wrangling.

## Overview

**Prerequisites:** Complete [data quality](data-quality.md), [missing values](missing-values.md), and [transformations](transformations.md) (or equivalent experience). Same Python stack as the [module README](./).

> **Time needed:** Often 6-12 hours including documentation.

## Why this matters

This project is the wrangling capstone: you show **judgment** (what to impute, cap, or flag) and **traceability** (what changed between raw exports and analysis-ready tables), not only that you can call pandas functions.

## Business Context

As a data scientist at "GlobalMart", you face a critical challenge: the company's raw data needs significant cleaning and preparation before it can be used for advanced analytics. This project will guide you through the complete data wrangling process, from initial assessment to final validation.

### Business Objectives

1. **Customer Segmentation**: Identify distinct customer groups for targeted marketing
2. **Pricing Optimization**: Analyze price elasticity and optimize pricing strategies
3. **Recommendation System**: Build a reliable product recommendation engine
4. **Churn Prediction**: Develop early warning system for customer churn

## Project Workflow

## Dataset Description

### Data Schema

Customers table

Defines the `customers` entity with demographic fields (`age`, `gender`, `country`), contact info (`email`), and two date columns that will later be used to validate temporal consistency.

Transactions table

Records each purchase event. Foreign keys to both `customers` and `products` enforce referential integrity; `amount`, `payment_method`, and `device_type` are the primary cleaning and outlier-detection targets.

Products table

Holds catalogue attributes: `price` and `stock_level` need range validation, `category` and `brand` need standardised casing, and `description` is a free-text field prone to nulls.

## Implementation Guide

### 1. Data Quality Assessment (20%)

Function signature and results scaffold

Declares the function and initialises a `quality_report` dict with four empty sub-dicts, one for each quality dimension (completeness, validity, consistency, uniqueness), so downstream code can populate them independently.

Completeness and validity checks

Completeness: counts and percentages of null values per column. Validity: iterates over numeric columns to record `min`/`max` ranges, useful for spotting impossible values like negative ages or prices above business thresholds.

Consistency and uniqueness checks, then return

Consistency: guards with `if … in df.columns` before comparing `last_login_date >= registration_date`-the `.mean()` gives the pass rate. Uniqueness: cardinality ratio per column (values between 0 and 1; close to 1 signals near-unique, close to 0 signals high repetition). The complete report dict is returned.

### 2. Data Cleaning Implementation (30%)

Class definition and configuration

Defines the `DataCleaner` class and its `__init__`. The optional `config` dict sets domain-specific thresholds (`age_range`, `price_range`, `outlier_threshold`) so the same class works for different business contexts without code changes.

clean\_customer\_data method

Works on a copy to avoid mutating the original. Fills missing ages with the median and missing countries with the mode, casts the date column, nullifies ages outside the configured range (rather than dropping rows), and uppercases country codes for consistency.

clean\_transaction\_data method

Deduplicates on the natural key (`customer_id` + `timestamp` + `amount`), then uses a z-score threshold to null-out extreme `amount` values rather than deleting rows, preserving row count for downstream joins. Payment method strings are lowercased for grouping consistency.

### 3. Feature Engineering (30%)

create\_customer\_features method

Computes four RFM-style signals per customer from the transactions table: lifetime value (total spend), purchase frequency (transaction count), average order value, and recency (days since last purchase). All four series are assembled into a single `DataFrame` indexed by `customer_id`.

create\_product\_features method

Derives two product signals: sales velocity (transaction count per product) and price elasticity. The nested `calculate_elasticity` helper computes the percentage change in demand divided by the percentage change in price for each product group, a standard economics metric for sensitivity analysis.

### 4. Data Validation (20%)

Function signature and results scaffold

Takes the cleaned `df` and a `validation_rules` dict (allowing callers to inject custom rule functions). Initialises three empty sub-dicts so each check populates its own section of the report independently.

Completeness and consistency checks

Completeness: stores null counts and percentages. Consistency: iterates over caller-supplied rule functions, each receives the full DataFrame and returns a scalar or Series, making the validator extensible without touching this code.

Statistical validation and return

For every numeric column, records mean, standard deviation, skewness, and excess kurtosis, a quick distributional health-check. High skew or kurtosis after cleaning can indicate remaining outliers or transformation needs. The full report dict is returned.

## Project Deliverables

### 1. Code Repository Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── transactions.csv
│   │   └── products.csv
│   └── processed/
│       └── final_dataset.csv
├── notebooks/
│   ├── 1_exploration.ipynb
│   ├── 2_cleaning.ipynb
│   └── 3_transformation.ipynb
├── src/
│   ├── data_quality.py
│   ├── cleaning.py
│   └── transformation.py
└── docs/
    ├── data_dictionary.md
    └── quality_report.md
```

### 2. Quality Report Template

```markdown
# Data Quality Report

## Executive Summary
- Key findings
- Critical issues
- Recommendations

## Detailed Analysis
1. Missing Data
   - Patterns identified
   - Treatment strategies
   - Impact assessment

2. Outliers
   - Detection methods
   - Treatment decisions
   - Business implications

3. Transformations
   - Techniques applied
   - Validation results
   - Performance metrics
```

## Gotchas

* **Filling missing ages with the median before validating the range**: if you call `fillna(median)` first and then null out out-of-range values, you may silently re-introduce the median as a valid age for rows whose original value was actually invalid; validate first, then impute.
* **Z-score outlier detection breaks on skewed `amount` distributions**: z-scores assume approximate normality; transaction amounts are often right-skewed, so a z-score threshold of 3 may miss extreme high-value outliers while flagging legitimate high spenders. Consider IQR or domain-based caps instead.
* **Deduplicating on `(customer_id, timestamp, amount)` misses near-duplicate rows**: two rows with the same customer and amount but slightly different timestamps (e.g., from retry logic) are not caught; add `product_id` to the subset or round timestamps before deduplication.
* **`pd.Timestamp.now()` is timezone-naive by default**: computing `days_since_last_purchase` against a timezone-aware `timestamp` column raises a TypeError; use `pd.Timestamp.now(tz='UTC')` or strip timezone info from the column consistently.
* **`calculate_elasticity` divides by zero silently**: if `price_pct_change` is 0 (no price change in the period), the division produces `inf` or `NaN` without raising an error; add a guard or filter out groups with no price variation before computing elasticity.
* **Quality reports serialise to nested dicts, not DataFrames**: the `assess_data_quality` return value contains Series and scalar objects that may not render cleanly in a Jupyter notebook or export to JSON without explicit conversion; call `.to_dict()` on Series before packaging the report.

## Best Practices

1. **Version Control**
   * Use Git for code versioning
   * Document all data transformations
   * Track data quality metrics
2. **Performance Optimization**
   * Use efficient data structures
   * Implement parallel processing
   * Optimize memory usage
3. **Documentation**
   * Maintain clear documentation
   * Create data dictionaries
   * Document assumptions and decisions

## Evaluation Criteria

1. **Code Quality (30%)**
   * Clean, well-organized code
   * Proper error handling
   * Efficient implementations
2. **Documentation (20%)**
   * Clear explanations
   * Comprehensive data dictionary
   * Well-documented decisions
3. **Results (50%)**
   * Data quality improvements
   * Feature engineering effectiveness
   * Validation metrics

Remember: "The quality of your data wrangling directly impacts the reliability of your analytics!"
