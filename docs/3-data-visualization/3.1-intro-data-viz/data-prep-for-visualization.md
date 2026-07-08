# Preparing Data for Visualization

**After this lesson:** you can reshape, sort, aggregate, and clean data so your charts answer the intended question instead of just plotting raw tables.

> **Note:** This lesson sits between [Visualization principles](visualization-principles.md) and [Matplotlib basics](matplotlib-basics.md). Good charts start with good framing, and good framing usually starts with preparing the data first.

## Helpful video

A practical walkthrough of cleaning and reshaping a real dataset in pandas before visualization.

<iframe width="560" height="315" src="https://www.youtube.com/embed/bDhvCp3_lYw" title="Data Cleaning in Pandas, Alex the Analyst" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Why data prep matters

A chart can be technically correct and still be misleading. Many visualization problems come from plotting the wrong level of detail, leaving categories unsorted, mixing missing values into summaries, or skipping the aggregation step that matches the business question.

Think of data prep as deciding what story the chart is allowed to tell:

- **Granularity:** Are you plotting rows, daily totals, or monthly averages?
- **Ordering:** Does the viewer see a meaningful sequence or a random one?
- **Completeness:** Are missing values handled consistently?
- **Comparability:** Are categories and units aligned before plotting?

## Start with the question

Before writing plotting code, state the question in one sentence.

- "Which product category generated the most revenue last quarter?"
- "How does order value vary by payment method?"
- "Did weekly traffic change after the campaign launch?"

That sentence determines the data shape you need.

{% include mermaid-diagram.html src="3-data-visualization/3.1-intro-data-viz/diagrams/data-prep-for-visualization-1.mmd" %}

## Core preparation patterns

We use one sample dataset throughout all five steps so you can see the data change at each stage.

```python
# no-output
import pandas as pd

sales = pd.DataFrame({
    "order_id":     ["A1",  "A2",  "A3",  "A4",  "A5",  "A6",  "A7",  "A8",  "A9",  "A10"],
    "order_date":   ["2025-03-01","2025-06-15","2025-08-20","2024-11-05","2025-11-12",
                     "2025-01-30","2025-06-15","2025-09-01","2024-12-20","2025-06-15"],
    "status":       ["completed","completed","completed","completed","cancelled",
                     "completed","completed","completed","completed","completed"],
    "category":     ["Electronics","Clothing","Electronics","Clothing","Electronics",
                     "Food","Food","Clothing","Electronics","Clothing"],
    "revenue":      [250.0, 45.0, 180.0, 60.0, None, 30.0, 25.0, 90.0, 200.0, 55.0],
    "product_name": ["Laptop","T-Shirt","Phone","Jeans","Headphones",
                     "Coffee","Tea","Jacket","Tablet","Dress"],
})
sales["order_date"] = pd.to_datetime(sales["order_date"])
```

**Starting dataset, 10 rows, 6 columns:**

| order_id | order_date | status | category | revenue | product_name |
|---|---|---|---|---|---|
| A1 | 2025-03-01 | completed | Electronics | 250.0 | Laptop |
| A2 | 2025-06-15 | completed | Clothing | 45.0 | T-Shirt |
| A3 | 2025-08-20 | completed | Electronics | 180.0 | Phone |
| **A4** | **2024-11-05** | completed | Clothing | 60.0 | Jeans |
| **A5** | 2025-11-12 | **cancelled** | Electronics | **NaN** | Headphones |
| A6 | 2025-01-30 | completed | Food | 30.0 | Coffee |
| A7 | 2025-06-15 | completed | Food | 25.0 | Tea |
| A8 | 2025-09-01 | completed | Clothing | 90.0 | Jacket |
| **A9** | **2024-12-20** | completed | Electronics | 200.0 | Tablet |
| A10 | 2025-06-15 | completed | Clothing | 55.0 | Dress |

Three rows need attention (highlighted): A4 and A9 are from 2024, and A5 is cancelled with missing revenue.

### 1. Inspect the dataset first

**Purpose:** Check column names, data types, missing values, and duplicate rows before plotting.

**Walkthrough:** Use `info`, `isna`, and `duplicated` to catch problems that would later produce broken axes, empty categories, or misleading counts.

```python
def inspect_for_viz(df):
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return summary

for key, val in inspect_for_viz(sales).items():
    print(f"{key}: {val}")
```

```
shape: (10, 6)
columns: ['order_id', 'order_date', 'status', 'category', 'revenue', 'product_name']
missing_values: {'order_id': 0, 'order_date': 0, 'status': 0, 'category': 0, 'revenue': 1, 'product_name': 0}
duplicate_rows: 0
dtypes: {'order_id': 'object', 'order_date': 'datetime64[ns]', 'status': 'object', 'category': 'object', 'revenue': 'float64', 'product_name': 'object'}
```

One missing `revenue` value in row A5. `order_date` is already `datetime64` so grouping by period will work without conversion.

### 2. Filter to the relevant slice

**Purpose:** Remove rows outside the question's scope before summarizing.

**Walkthrough:** Filtering first avoids mixing time periods, regions, or statuses that should not be compared.

```python
sales_2025 = sales.loc[
    (sales["order_date"] >= "2025-01-01")
    & (sales["order_date"] < "2026-01-01")
    & (sales["status"] == "completed")
].copy()

print(sales_2025[["order_id","order_date","status","category","revenue"]].to_string(index=False))
```

```
order_id order_date     status     category  revenue
      A1 2025-03-01  completed  Electronics    250.0
      A2 2025-06-15  completed     Clothing     45.0
      A3 2025-08-20  completed  Electronics    180.0
      A6 2025-01-30  completed         Food     30.0
      A7 2025-06-15  completed         Food     25.0
      A8 2025-09-01  completed     Clothing     90.0
     A10 2025-06-15  completed     Clothing     55.0
```

Down from 10 rows to 7: A4 and A9 (2024 dates) and A5 (cancelled) are gone.

### 3. Aggregate to the right level

**Purpose:** Turn transaction-level rows into chart-ready summaries.

**Walkthrough:** `groupby(...).agg(...)` creates one row per visual mark when the question is about category totals or averages.

```python
category_summary = (
    sales_2025
    .groupby("category", as_index=False)
    .agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        avg_order_value=("revenue", "mean"),
    )
)

print(category_summary.to_string(index=False))
```

```
    category  revenue  orders  avg_order_value
    Clothing    190.0       3    63.333333
 Electronics    430.0       2   215.000000
        Food     55.0       2    27.500000
```

7 transaction rows collapsed to 3 summary rows, one per category. Each row now represents one bar mark on the chart.

### 4. Sort for readability

**Purpose:** Make rankings and comparisons obvious.

**Walkthrough:** Most bar charts are easier to read when sorted by the metric being compared rather than alphabetically.

```python
top_categories = (
    category_summary
    .sort_values("revenue", ascending=False)
    .head(10)
)

print(top_categories.to_string(index=False))
```

```
    category  revenue  orders  avg_order_value
 Electronics    430.0       2   215.000000
    Clothing    190.0       3    63.333333
        Food     55.0       2    27.500000
```

Electronics moves to the top. The chart now reads highest-to-lowest without the viewer having to search.

### 5. Reshape when needed

**Purpose:** Convert wide tables into long form for libraries like Seaborn and Plotly Express.

**Walkthrough:** `melt` is especially useful when several metric columns should become one "metric" column and one "value" column. For example, to plot both `sales` and `returns` as grouped lines on the same chart:

```python
monthly_wide = pd.DataFrame({
    "month":   ["Jan", "Feb", "Mar"],
    "sales":   [120,   150,   170],
    "returns": [5,     7,     6],
})

monthly_long = monthly_wide.melt(
    id_vars="month",
    value_vars=["sales", "returns"],
    var_name="metric",
    value_name="value",
)

print(monthly_wide)
print()
print(monthly_long)
```

```
  month  sales  returns
0   Jan    120        5
1   Feb    150        7
2   Mar    170        6

  month   metric  value
0   Jan    sales    120
1   Feb    sales    150
2   Mar    sales    170
3   Jan  returns      5
4   Feb  returns      7
5   Mar  returns      6
```

Wide form (3 rows × 3 columns) becomes long form (6 rows × 3 columns). Seaborn's `hue` parameter can now split on `metric` to draw two lines from a single column.

## Handling common issues

### Missing values

- Drop rows when missingness is rare and clearly accidental.
- Impute only when the business meaning supports it.
- Leave gaps in time series when showing missingness is informative.

```python
plot_df = orders.dropna(subset=["order_value", "segment"]).copy()
```

### Long category tails

Too many categories create unreadable charts. Group small groups into `"Other"` or show only the top `n`.

```python
top_names = (
    sales_2025["product_name"]
    .value_counts()
    .head(8)
    .index
)

sales_2025["product_group"] = sales_2025["product_name"].where(
    sales_2025["product_name"].isin(top_names),
    "Other"
)
```

### Dates stored as strings

Convert them before grouping or plotting.

```python
traffic["date"] = pd.to_datetime(traffic["date"])
traffic["month"] = traffic["date"].dt.to_period("M").astype(str)
```

## Chart-ready examples

### Comparison chart prep

**Purpose:** Prepare a sorted category table for a horizontal bar chart.

**Walkthrough:** Filter, aggregate, sort, and keep only the columns needed for plotting.

```python
bar_ready = (
    sales.loc[sales["region"] == "Central"]
    .groupby("category", as_index=False)
    .agg(revenue=("revenue", "sum"))
    .sort_values("revenue", ascending=True)
)
```

### Distribution chart prep

**Purpose:** Prepare one clean numeric series for a histogram or box plot.

**Walkthrough:** Remove impossible values and preserve only the variable the distribution chart needs.

```python
distribution_ready = (
    orders.loc[orders["order_value"].between(0, 1000), ["order_value"]]
    .dropna()
)
```

### Relationship chart prep

**Purpose:** Keep only paired numeric observations for a scatter plot.

**Walkthrough:** Scatter plots require matching `x` and `y` values; dropping nulls on both columns prevents broken points.

```python
scatter_ready = (
    marketing[["ad_spend", "revenue", "channel"]]
    .dropna(subset=["ad_spend", "revenue"])
)
```

## A reusable preparation checklist

Use this before every chart:

1. What exact question is the chart answering?
2. What should one mark represent: a row, a group, or a time period?
3. Which rows should be excluded?
4. Which columns need to be transformed or reshaped?
5. Is the result sorted in the order the viewer expects?
6. Are missing values, duplicates, and units handled clearly?

## Common mistakes

- Plotting raw transactional rows when the question needs grouped totals.
- Leaving categories unsorted so rankings are hard to scan.
- Mixing incomplete and complete periods in the same time chart.
- Using wide-form data with a library example that expects long-form data.
- Treating missing values as zero without checking what they mean.

## Practice prompts

1. Prepare a category summary from a sales table and sort it for a bar chart.
2. Convert a daily dataset into monthly totals for a line chart.
3. Reshape a wide KPI table into long form for a Seaborn grouped plot.
4. Clean a numeric column with missing and impossible values before drawing a histogram.

## Gotchas

- **`groupby(...).agg(revenue=("revenue", "sum"))` silently drops rows with `NaN` in the aggregated column**: pandas excludes NaN values from `sum` and `mean` by default, so if A5's missing revenue is in the filtered dataset it simply vanishes from the total without any warning; verify your sums add up to the expected grand total after aggregation.
- **`pd.to_datetime(sales["order_date"])` raises a `ParserError` on inconsistent date formats**: if a column mixes `"2025-03-01"` and `"01/03/2025"`, the conversion will fail or silently coerce bad rows to `NaT`; pass `format=` explicitly or use `errors='coerce'` and then inspect the resulting `NaT` rows before plotting.
- **`.copy()` after `.loc[]` is not optional**: assigning new columns to a slice without `.copy()` raises a `SettingWithCopyWarning` and may not persist the change; always chain `.copy()` when you intend to modify the filtered DataFrame in subsequent steps.
- **`melt` changes the number of rows, which can surprise downstream `.groupby` logic**: after melting a wide table into long form, the same original row now appears multiple times (once per melted column); if you then `groupby` the melted result and count rows, you will overcount observations unless you group by the correct id variable.
- **Sorting before plotting does not persist unless you reset the index**: `sort_values("revenue", ascending=False)` returns a sorted copy but preserves the original integer index; some plotting functions (especially those that accept a DataFrame column by name) may re-sort by index internally; call `.reset_index(drop=True)` after sorting to guarantee the order is preserved.
- **Using `where(..., "Other")` to group a long tail works only if the column dtype is `object`**: if `product_name` is a `Categorical` dtype, the new value `"Other"` may not be in the existing categories and will be set to `NaN` instead; cast the column with `.astype(str)` before applying the `where` pattern.

## Next steps

1. Use [Matplotlib basics](matplotlib-basics.md) to turn prepared tables into clean static charts.
2. Use [Annotations and highlighting](annotations-and-highlighting.md) to direct attention to the most important points in those charts.
3. Use [3.2 Advanced data visualization](../3.2-adv-data-viz/README.md) once your data prep workflow feels natural.
