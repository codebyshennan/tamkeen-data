# Choosing the Right Visualization: A Beginner's Guide

**After this lesson:** you can match **data types** and **analysis goals** to common chart families and avoid obvious mismatches (for example, pie charts for trends).

## Helpful video

Orientation for the course visualization materials.

## Overview

> **Note:** This is a **reference-style** lesson: use it beside [Visualization principles](3.1-intro-data-viz/visualization-principles.md) and your Python or BI tool of choice.

## Prerequisites

* Basic idea of **numeric** vs **categorical** data and, when relevant, **time series** (from EDA or statistics modules).
* Optional: skim [Common mistakes](common-mistakes.md) for paired "do/don't" examples.

## Understanding Your Data Type

Before choosing a visualization, identify what type of data you're working with:

### 1. Numerical Data (Quantitative)

* **Continuous**: Can take any value (e.g., height, weight, temperature)
* **Discrete**: Only specific values (e.g., number of students, count of items)

### 2. Categorical Data (Qualitative)

* **Nominal**: Categories with no order (e.g., colors, names)
* **Ordinal**: Categories with order (e.g., satisfaction levels: low, medium, high)

### 3. Time Series Data

* Data points collected over time (e.g., daily temperature, monthly sales)

_When in doubt, start with a bar chart or scatter plot, they work for most business questions and are the easiest to read correctly._

## Matching Data Types to Visualizations

### For Numerical Data

#### 1. Distribution of Values

**Best Charts:**

* Histogram
* Box Plot
* Density Plot

**When to Use:**

* Understanding the spread of data
* Finding outliers
* Checking for normal distribution

**Example:**

**Purpose:** Sketch a one-variable distribution with frequency on the y-axis and binned values on the x-axis.

**Walkthrough:** `hist` chooses bin counts via `bins=`; add title and axis labels so "what is counted" is explicit. Assumes `plt` and `data` exist.

```python
# Histogram for showing distribution
plt.hist(data, bins=30)
plt.title('Distribution of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
```

#### 2. Comparing Numbers

**Best Charts:**

* Bar Chart
* Column Chart
* Dot Plot

**When to Use:**

* Comparing quantities across categories
* Showing rankings
* Displaying survey results

### For Categorical Data

#### 1. Proportions

**Best Charts:**

* Pie Chart (for 2-6 categories)
* Donut Chart
* Treemap (for many categories)

**When to Use:**

* Showing parts of a whole
* Displaying percentages
* Comparing proportions

#### 2. Comparing Categories

**Best Charts:**

* Bar Chart
* Column Chart
* Lollipop Chart

**When to Use:**

* Comparing values across categories
* Showing rankings
* Displaying survey results

### For Time Series Data

#### 1. Trends Over Time

**Best Charts:**

* Line Chart
* Area Chart
* Sparkline

**When to Use:**

* Showing changes over time
* Identifying trends
* Displaying patterns

#### 2. Seasonal Patterns

**Best Charts:**

* Multiple Line Chart
* Stacked Area Chart
* Heat Map

**When to Use:**

* Comparing multiple time series
* Finding seasonal patterns
* Showing periodic changes

## Common Analysis Goals and Recommended Charts

### 1. Showing Composition

**Goal**: Show how parts make up a whole **Best Charts:**

* Pie Chart
* Stacked Bar Chart
* Treemap

### 2. Showing Distribution

**Goal**: Show how values are spread out **Best Charts:**

* Histogram
* Box Plot
* Violin Plot

### 3. Showing Relationship

**Goal**: Show how variables relate to each other **Best Charts:**

* Scatter Plot
* Bubble Chart
* Heat Map

### 4. Showing Comparison

**Goal**: Compare values across categories **Best Charts:**

* Bar Chart
* Radar Chart
* Parallel Coordinates

### 5. Showing Change Over Time

**Goal**: Show how values change over time **Best Charts:**

* Line Chart
* Area Chart
* Candlestick Chart (for financial data)

## Decision Tree for Choosing Visualizations

1. **What's your main goal?**
   * Comparison → Bar/Column Chart
   * Distribution → Histogram/Box Plot
   * Composition → Pie/Stacked Bar Chart
   * Relationship → Scatter Plot
   * Trend → Line Chart
2. **How many variables?**
   * One variable → Histogram/Bar Chart
   * Two variables → Scatter Plot/Line Chart
   * Three variables → Bubble Chart/3D Scatter
   * Many variables → Parallel Coordinates/Heat Map
3. **What's your audience?**
   * General Public → Simple Charts (Bar, Line, Pie)
   * Technical Audience → Complex Charts (Box Plot, Heat Map)
   * Executive → Summary Charts (Sparkline, Dashboard)

## Tips for Effective Visualization

1. **Keep it Simple**
   * Start with basic charts
   * Add complexity only if needed
   * Remove unnecessary elements
2. **Consider Your Audience**
   * Match complexity to audience expertise
   * Use familiar chart types
   * Provide clear explanations
3. **Tell a Story**
   * Focus on key message
   * Guide viewer's attention
   * Provide context
4. **Make it Accessible**
   * Use colorblind-friendly colors
   * Include clear labels
   * Add descriptive titles

## Examples in Different Domains

### Business

* Sales Trends → Line Chart
* Market Share → Pie Chart
* Revenue by Product → Bar Chart
* Customer Satisfaction → Heat Map

### Science

* Experimental Results → Box Plot
* Correlations → Scatter Plot
* Time Series Data → Line Chart
* Distributions → Histogram

### Social Sciences

* Survey Results → Bar Chart
* Demographics → Pie Chart
* Trends Over Time → Line Chart
* Relationships → Scatter Plot

## Common Pitfalls to Avoid

1. **Don't use pie charts for more than 6 categories**
   * Use a bar chart instead
   * Consider grouping small categories
2. **Don't use 3D charts unless necessary**
   * They often distort the data
   * 2D is usually clearer
3. **Don't use dual axes without clear reason**
   * They can be misleading
   * Consider separate charts
4. **Don't forget to label**
   * Add clear titles
   * Label axes
   * Include units

## Gotchas

* **Pie charts with more than 6 slices become unreadable even when the rule is nominally followed**: six segments is a ceiling, not a target; if any slice is under \~5% of the total it will be impossible to label clearly and should be folded into an "Other" category or replaced with a sorted bar chart.
* **Bar charts and column charts are not interchangeable**: vertical bars (column charts) are conventional for time-ordered data; horizontal bar charts read better when category labels are long or when there are more than \~8 categories; mixing orientations on the same dashboard creates visual inconsistency.
* **Histograms are not bar charts**: a histogram shows the distribution of one continuous variable by binning it; using `plt.bar` with pre-counted data looks similar but requires you to already know the bin edges and counts, while `plt.hist` computes them automatically.
* **Scatter plots require both variables to be on a ratio or interval scale**: plotting an ordinal variable (e.g. satisfaction: low/medium/high) on a scatter plot axis implies equal spacing between categories, which is often false; use a strip plot or box plot instead.
* **Heat maps hide the shape of distributions**: showing a correlation matrix as a heat map tells you the strength of relationships but not whether the underlying relationship is linear, curved, or driven by outliers; always back up a heat map with individual scatter plots for strong correlations.
* **Line charts with a non-zero y-axis exaggerate trends**: starting the y-axis well above zero makes small changes appear dramatic; this is appropriate for some financial contexts but should always be explicitly noted in the axis label or title.

## Next steps

* Implement choices in [Matplotlib basics](3.1-intro-data-viz/matplotlib-basics.md) or your BI lesson in [3.3](3.3-bi-with-tableau/).
* See the [module overview](./) for the full learning path.
