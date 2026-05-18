# Assignment: Advanced Data Visualization

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to prepare all datasets. No external files are needed.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

# ── Seaborn tasks: built-in tips dataset ──────────────────────────────────────
tips = sns.load_dataset("tips")

# ── Plotly task: built-in gapminder dataset ───────────────────────────────────
gapminder = px.data.gapminder()
gapminder_2007 = gapminder[gapminder["year"] == 2007].copy()

# ── Time-series task: built-in flights dataset ────────────────────────────────
flights_raw = sns.load_dataset("flights")

month_num = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,
    "May":5,"Jun":6,"Jul":7,"Aug":8,
    "Sep":9,"Oct":10,"Nov":11,"Dec":12,
}
flights_raw["month_num"] = flights_raw["month"].astype(str).map(month_num)
flights_raw["date"] = pd.to_datetime(
    dict(year=flights_raw["year"], month=flights_raw["month_num"], day=1)
)
flights = flights_raw.sort_values("date").reset_index(drop=True)

# Monthly pivot for the heatmap
flights_pivot = flights.pivot_table(
    index="year", columns="month", values="passengers"
)

print("Setup complete.")
print(f"tips: {tips.shape}  |  gapminder_2007: {gapminder_2007.shape}  |  flights: {flights.shape}")
```

## Tasks

### 1. Seaborn distribution charts

Using the `tips` dataset:

- Create a `sns.histplot` of `total_bill` with `kde=True`. Set `bins=20` and color it with a palette of your choice.
- On a separate axes, create a `sns.boxplot` comparing `total_bill` across each `day` of the week. Use `hue="smoker"` to split smokers from non-smokers.
- Arrange both plots side-by-side in a single figure (`plt.subplots(1, 2, figsize=(14, 5))`). Add a descriptive title to the whole figure with `fig.suptitle(...)`.

### 2. Seaborn heatmap — correlation matrix

Using the `flights_pivot` table prepared in setup:

- Compute the correlation matrix of `flights_pivot` (correlations between months across years).
- Draw a `sns.heatmap` with `annot=True`, `fmt=".2f"`, `cmap="coolwarm"`, and `center=0`.
- Add a clear title ("Monthly Passenger Correlations, 1949–1960") and tidy the axis labels so they are readable.

### 3. Plotly interactive scatter

Using `gapminder_2007`:

- Create an interactive scatter plot with `px.scatter`:
  - x = `gdpPercap`, y = `lifeExp`
  - `size="pop"`, `color="continent"`, `hover_name="country"`
  - `log_x=True` (GDP per capita spans many orders of magnitude)
  - `title="GDP vs Life Expectancy (2007)"`
- Use `fig.update_layout(...)` to set `xaxis_title` to `"GDP per Capita (log scale)"` and `yaxis_title` to `"Life Expectancy (years)"`.
- Display the figure with `fig.show()`.

### 4. Time-series line chart with range slider

Using the `flights` DataFrame:

- Aggregate the data to **annual** passenger totals:
  ```python
  annual = flights.groupby("year")["passengers"].sum().reset_index()
  annual["date"] = pd.to_datetime(annual["year"].astype(str))
  ```
- Create an interactive Plotly line chart with `px.line`:
  - x = `"date"`, y = `"passengers"`
  - `title="Annual Airline Passengers (1949–1960)"`
  - `labels={"date": "Year", "passengers": "Total Passengers"}`
- Add a **range slider** below the x-axis with `fig.update_xaxes(rangeslider_visible=True)`.
- Format the y-axis with thousand separators: `fig.update_yaxes(tickformat=",")`.

## Deliverable

Submit a single Python script or Jupyter notebook that:

1. Runs end-to-end without errors.
2. Displays or saves all four visualizations.
3. Includes brief inline comments on any non-obvious configuration choices.

## Hints

<details>
<summary>Show hints</summary>

### 1. Seaborn distribution charts
- **Where:** [Seaborn guide](../seaborn-guide.md) — "Distribution Analysis — Single Variable Distributions" and "Categorical Distributions".
- **Think:** Pass `ax=ax1` and `ax=ax2` when calling Seaborn functions inside a `subplots` grid so each chart lands in the right panel.
- **boxplot with hue:** `sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker", ax=ax2)` — Seaborn handles the color split automatically once `hue=` is set.

### 2. Seaborn heatmap
- **Where:** [Seaborn guide](../seaborn-guide.md) — "Matrix Visualizations — Correlation Analysis".
- **Think:** `flights_pivot.corr()` returns a DataFrame of pairwise correlations between columns (months). `annot=True` requires all values to be numeric — the pivot table already satisfies this.
- **Caution:** If any month column contains NaN (missing year-month combinations), `.corr()` will drop those pairs silently. Check `flights_pivot.isna().sum()` first.

### 3. Plotly interactive scatter
- **Where:** [Plotly guide](../plotly-guide.md) — "Basic Interactive Plots — Enhanced Scatter Plots".
- **Think:** `log_x=True` is a single parameter on `px.scatter` — no manual transformation needed. `hover_name="country"` shows the country name as the tooltip header.
- **Layout update:** `fig.update_layout(xaxis_title="...", yaxis_title="...")` — these override the auto-generated column names shown on the axes.

### 4. Time-series range slider
- **Where:** [Time-series visualization](../time-series-visualization.md) — "Plotly for interactive exploration", and [Plotly guide](../plotly-guide.md) — "Basic Interactive Plots — Time Series Visualization".
- **Think:** `rangeslider_visible=True` goes on `update_xaxes`, not `update_layout`. The range slider is a property of the x-axis, not the figure canvas.
- **Tick format:** `fig.update_yaxes(tickformat=",")` — the comma format adds thousand separators (e.g. 15,000 instead of 15000). For the date axis you do not need a custom format — Plotly auto-formats datetimes.

### Common pitfalls
- Mixing `plt.` and `ax.` calls in Seaborn: once you pass `ax=ax1`, all further customization of that panel should use `ax1.set_title(...)`, not `plt.title(...)`.
- `sns.heatmap` with `fmt='.2f'` throws a `TypeError` if any cell is NaN. Fill NaNs before plotting: `corr_matrix.fillna(0)` or check that the pivot is complete.
- `fig.show()` in a script opens a browser window; in a Jupyter notebook it renders inline. If nothing appears in a script, use `fig.write_html("output.html")` and open the file.
- Forgetting `reset_index()` after `groupby` — the resulting Series has the groupby column as the index, which confuses `px.line` when you reference it by column name.

</details>
