# Exploratory Data Analysis Assignment

**After this lesson:** You produce a short EDA report (notebook or slides) with distributions, key relationships, and clear limitations, grounded in the [EDA README](./) workflow.

## Helpful video

Summarizing distributions with percentiles, common in exploratory analysis.

## Overview

**Prerequisites:** [Distributions](distributions.md), [relationships](relationships.md), and [time series](time-series.md) readings (or parallel skimming). [Wrangling (Module 2.2)](../2.2-data-wrangling/) should be done or in progress.

> **Time needed:** Often 6-10 hours including polish.

## Why this matters

EDA deliverables should read like **evidence**, not a gallery of plots: each figure should answer a stated question, and the write-up should name **limitations** (time window, selection bias, missing fields) alongside conclusions.

In this assignment, you perform exploratory data analysis on a realistic e-commerce-style dataset. You apply the workflow from the readings to uncover patterns, relationships, and trends, and to document what you cannot claim from the data alone.

## Dataset Description

You'll be working with an e-commerce dataset containing:

* Customer transactions
* Product information
* Temporal data
* Customer demographics
* Sales metrics

## Setup

Required libraries

Six imports covering data manipulation (pandas, numpy), visualisation (matplotlib, seaborn), statistical tests (scipy), and time-series decomposition (statsmodels). The last line loads the assignment dataset.

## Tasks

### 1. Data Distribution Analysis (25 points)

a) Numeric Variables (15 points)

* Analyze the distribution of sales amounts
* Examine customer spending patterns
* Study product pricing distributions
* Identify and handle outliers
* Transform skewed distributions if necessary

b) Categorical Variables (10 points)

* Analyze product category distributions
* Examine customer demographics
* Study geographical distributions
* Create meaningful visualizations for each

### 2. Relationship Analysis (25 points)

a) Numeric Relationships (10 points)

Scaffold for numeric relationship analysis

A stub function to fill in: compute Pearson/Spearman correlations and create scatter plots or a heatmap, then return the results dict.

b) Categorical Relationships (10 points)

* Cross-tabulations of categories
* Chi-square tests of independence
* Visualization of category relationships

c) Mixed Variable Analysis (5 points)

* Compare numeric variables across categories
* Analyze variance between groups
* Create box plots and violin plots

### 3. Time Series Analysis (25 points)

a) Temporal Patterns (10 points)

* Daily sales patterns
* Weekly trends
* Monthly seasonality
* Year-over-year growth

b) Decomposition (10 points)

* Trend analysis
* Seasonal patterns
* Residual analysis
* Moving averages

c) Anomaly Detection (5 points)

* Identify unusual patterns
* Detect seasonal anomalies
* Flag suspicious transactions

### 4. Advanced Analysis (15 points)

a) Customer Segmentation

RFM metric stubs

The three comment placeholders mark where you compute Recency (days since last purchase), Frequency (order count), and Monetary value (total spend) per customer.

Clustering stub

A placeholder for the clustering step, e.g. KMeans on the scaled RFM matrix, that assigns each customer a segment label returned as `segments`.

b) Product Analysis

* Analyze product affinities
* Study category performance
* Identify top performers

c) Geographic Analysis

* Regional sales patterns
* Location-based trends
* Market penetration analysis

### 5. Documentation and Presentation (10 points)

a) Analysis Report

* Executive summary
* Key findings
* Methodology description
* Recommendations

b) Visualizations

* Clear and informative plots
* Proper labeling
* Consistent styling
* Interactive elements (optional)

## Deliverables

1. Jupyter Notebook containing:
   * All analysis code
   * Visualizations
   * Markdown explanations
   * Results interpretation
2. Summary Report (PDF) including:
   * Methodology overview
   * Key findings
   * Business recommendations
   * Future analysis suggestions
3. Presentation Slides:
   * Key visualizations
   * Main insights
   * Actionable recommendations

## Evaluation Criteria

* Code quality and organization (20%)
* Analysis depth and accuracy (30%)
* Visualization effectiveness (20%)
* Insights and interpretation (20%)
* Documentation clarity (10%)

## Solution Template

Setup and data loading

Imports and `load_and_prepare_data`: loads the CSV, parses the date column, and provides a stub for missing-value handling.

Distribution analysis (numeric and categorical)

Loops over numeric columns to plot histograms+KDE and print describe stats, then loops over categorical columns to plot bar charts of value counts.

Relationship analysis

Computes a numeric correlation matrix and visualises it as an annotated heatmap. A stub comment marks where categorical relationship code should go.

Time series analysis

Sets the date as index, resamples to daily totals, and plots the trend. A stub comment marks where seasonal decomposition should be added.

Report generation

Bundles summary statistics, correlation analysis, temporal patterns, and key findings into a single report dict, the deliverable for stakeholders.

Main execution block

Runs the full pipeline end to end: load → analyse distributions → analyse relationships → analyse time series → generate report, with stub result placeholders to fill in.

## Gotchas

* **Pearson correlation hides non-linear relationships**: the default `numeric_data.corr()` only captures linear association; two variables with a clear U-shaped relationship will show near-zero Pearson r, so always pair the heatmap with scatter plots for suspicious pairs.
* **`resample('D')['sales'].sum()` drops days with zero sales**: if a day has no transactions it simply doesn't appear in the daily Series, which distorts trend lines and decomposition; use `.asfreq('D', fill_value=0)` after resampling to make gaps explicit.
* **Seasonal decomposition requires a complete, regular time series**: `seasonal_decompose` raises or produces garbage if there are missing dates or the period is shorter than two full cycles; ensure your date index is monotonic and dense before calling it.
* **Plotting every numeric column in a loop pollutes the notebook**: calling `plt.show()` inside the distribution loop creates a new figure per column, which works fine locally but produces dozens of static images in a PDF report; aggregate small-multiple grids with `plt.subplots` for deliverables.
* **Chi-square tests assume expected frequencies ≥ 5**: for sparse category combinations (rare product categories × rare countries), expected counts may fall below this threshold and the p-value becomes unreliable; collapse infrequent levels before running the test.
* **Confusing correlation with causation in the executive summary**: EDA can only surface associations; writing "X causes Y" based on a correlation is a common grading deduction; always frame findings as "X is associated with Y" and flag what confounders might exist.

## Tips for Success

1. **Start with Questions**
   * Define analysis objectives
   * Form hypotheses
   * Plan visualization strategy
   * Consider business context
2. **Be Systematic**
   * Follow a structured approach
   * Document your process
   * Validate findings
   * Cross-check results
3. **Focus on Insights**
   * Look beyond basic statistics
   * Consider business implications
   * Identify actionable findings
   * Provide clear recommendations
4. **Create Clear Visualizations**
   * Choose appropriate plots
   * Use consistent styling
   * Add proper labels
   * Include explanations

## Bonus Challenges

1. **Advanced Visualization**
   * Create interactive plots
   * Build a dashboard
   * Implement custom visualizations
   * Add animation
2. **Statistical Analysis**
   * Hypothesis testing
   * Confidence intervals
   * Effect size calculations
   * Power analysis
3. **Machine Learning Integration**
   * Clustering analysis
   * Anomaly detection
   * Pattern recognition
   * Predictive modeling

Good luck! Remember to focus on generating actionable insights from your analysis!
