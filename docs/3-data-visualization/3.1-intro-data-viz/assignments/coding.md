# Assignment: Intro to Data Visualization with Matplotlib

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to create your dataset. No external files are needed.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reproducible random seed
rng = np.random.default_rng(42)

# Monthly sales dataset — 24 months, 3 product lines
months = pd.date_range("2023-01-01", periods=24, freq="MS")
electronics = 120 + np.cumsum(rng.normal(5, 15, 24)).round(1)
clothing     = 80  + np.cumsum(rng.normal(2, 10, 24)).round(1)
food         = 60  + np.cumsum(rng.normal(1,  8, 24)).round(1)

sales = pd.DataFrame({
    "month":       months,
    "Electronics": electronics,
    "Clothing":    clothing,
    "Food":        food,
})

# Category summary (average monthly sales per category)
category_avg = {
    "Electronics": sales["Electronics"].mean(),
    "Clothing":    sales["Clothing"].mean(),
    "Food":        sales["Food"].mean(),
}

# Scatter data — advertising spend vs monthly revenue
ad_spend = rng.uniform(5, 50, 60).round(1)
revenue  = 2.5 * ad_spend + rng.normal(0, 10, 60)

print("Dataset ready.")
print(sales.head())
```

```
Dataset ready.
       month  Electronics  Clothing  Food
0 2023-01-01        129.6      77.7  66.4
1 2023-02-01        119.0      76.2  68.0
2 2023-03-01        135.2      83.5  71.3
3 2023-04-01        154.3      89.2  77.3
4 2023-05-01        130.1      95.3  66.7
```

## Tasks

### 1. Line chart — multi-series trend

- Plot monthly sales for all three product lines (Electronics, Clothing, Food) on a single set of axes using the object-oriented Matplotlib API (`fig, ax = plt.subplots(...)`).
- Give each line a distinct color and a descriptive `label=`.
- Add a title, x-axis label ("Month"), y-axis label ("Sales (units)"), a grid with `alpha=0.4`, and a legend.
- Rotate x-axis tick labels by 45° so dates don't overlap (`ax.tick_params(axis='x', rotation=45)`).

### 2. Bar chart — category comparison

- Using `category_avg`, create a vertical bar chart comparing average monthly sales per category.
- Sort the bars from highest to lowest average before plotting.
- Display the average value as a label on top of each bar (round to one decimal place).
- Apply a consistent color scheme and add axis labels and a title.

### 3. Scatter plot — relationship exploration

- Plot `ad_spend` (x) against `revenue` (y) as a scatter plot.
- Map point color to `revenue` using the `viridis` colormap and add a colorbar labelled "Revenue".
- Set point size to 40 and alpha to 0.6.
- Add axis labels ("Ad Spend ($k)", "Revenue ($k)") and a title.

### 4. Chart annotation and layout

- Reproduce your line chart from Task 1 in a new figure.
- Identify the month where Electronics sales peaked and annotate it with `ax.annotate(...)`. Include an arrow pointing to the peak data point and a text label showing the month name and value.
- Add a horizontal dashed reference line at the Electronics mean value using `ax.axhline(...)` with a neutral grey color and a label in the legend.
- Save the final figure as `sales_trend_annotated.png` at 150 dpi using `fig.savefig(...)`.

## Deliverable

Submit a single Python script or Jupyter notebook that:

1. Runs without errors from top to bottom (all setup code included).
2. Displays or saves each of the four charts.
3. Includes brief inline comments explaining any non-obvious steps.

## Hints

<details>
<summary>Show hints</summary>

### 1. Line chart
- **Where:** [Matplotlib basics](../matplotlib-basics.md) — "Essential plot types — Line Plots".
- **Think:** Use `ax.plot(sales["month"], sales["Electronics"], label="Electronics", ...)` for each series. Three calls, three lines. The `label=` argument feeds `ax.legend()` automatically.
- **Rotation:** `ax.tick_params(axis='x', rotation=45)` or `plt.setp(ax.get_xticklabels(), rotation=45, ha='right')`.

### 2. Bar chart
- **Where:** [Matplotlib basics](../matplotlib-basics.md) — "Essential plot types — Bar Charts", and [Data prep for visualization](../data-prep-for-visualization.md) — "Sort for readability".
- **Think:** Sort the dict by value before passing to `ax.bar`. To add labels on top:
  ```python
  for bar in bars:
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
              f'{height:.1f}', ha='center', va='bottom')
  ```

### 3. Scatter plot
- **Where:** [Matplotlib basics](../matplotlib-basics.md) — "Essential plot types — Scatter Plots".
- **Think:** `ax.scatter(ad_spend, revenue, c=revenue, cmap='viridis', s=40, alpha=0.6)` returns a `PathCollection`. Pass it to `plt.colorbar(scatter, ax=ax, label='Revenue')` for the colorbar.

### 4. Annotation
- **Where:** [Matplotlib basics](../matplotlib-basics.md) — "Styling and Customization", and [Visualization principles](../visualization-principles.md) — "Best Practices — Start with a Clear Purpose".
- **Think:** Find the peak index with `peak_idx = sales["Electronics"].idxmax()`, then get the x and y values. Pass them as `xy=(x_val, y_val)` in `ax.annotate(...)`. Use `xytext` to offset the label so it doesn't overlap the line.
- **Mean line:** `ax.axhline(y=mean_val, color='grey', linestyle='--', label='Electronics mean')` — include it before `ax.legend()` so it appears in the legend.
- **Save:** `fig.savefig('sales_trend_annotated.png', dpi=150, bbox_inches='tight')`.

### Common pitfalls
- Calling `plt.title(...)` instead of `ax.set_title(...)` after creating an explicit `ax` — once you have an Axes reference, use `ax.` methods exclusively to avoid targeting the wrong subplot.
- Forgetting `ax.legend()` after setting `label=` — the labels are stored but not shown until `legend()` is called.
- Passing the colorbar a figure instead of an axes: `plt.colorbar(scatter, ax=ax)` is the safe form.
- `fig.savefig(...)` must be called before `plt.show()` — `show()` clears the figure.

</details>
