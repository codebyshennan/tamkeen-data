# Time Series Visualization

**After this lesson:** you can prepare and plot time-based data clearly, choose suitable time intervals, and annotate trends, seasonality, and events without misleading the viewer.

> **Note:** Build on [Matplotlib basics](../3.1-intro-data-viz/matplotlib-basics.md) first, then apply the richer styling and interactivity from the [Seaborn guide](seaborn-guide.md) and [Plotly guide](plotly-guide.md).

## What makes time data different?

Most charts treat data points as independent, swap two bars in a bar chart and the story barely changes. Time series data is different: the **order matters**.

Think about tracking monthly sales. If you accidentally sort by value instead of date, you'd see a perfectly smooth curve, and completely miss the seasonal dip in January. Or imagine comparing this month's revenue to last month's when this month isn't over yet. The chart would make things look worse than they are.

These are the specific traps time data sets for you:

- **Order matters**: points have a direction; swapping two values changes the story
- **Gaps matter**: a missing week is not the same as a zero week
- **Aggregation level matters**: daily data looks noisy; monthly data hides spikes
- **Partial periods mislead**: comparing an incomplete month to full months makes recent data look worse

This lesson shows you how to avoid each of these, and how to build charts that communicate time-based trends clearly.

## Setup

The examples below use the `flights` dataset, monthly airline passengers from 1949 to 1960. It is built into Seaborn so there is nothing to download.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

# Load the built-in flights dataset
flights = sns.load_dataset("flights")

# The month column contains abbreviated names like "Jan", "Feb" -
# map them to numbers so pandas can build a proper datetime
month_num = {
    "Jan":1, "Feb":2, "Mar":3, "Apr":4,
    "May":5, "Jun":6, "Jul":7, "Aug":8,
    "Sep":9, "Oct":10,"Nov":11,"Dec":12
}
flights["month_num"] = flights["month"].astype(str).map(month_num)
flights["date"] = pd.to_datetime(
    dict(year=flights["year"], month=flights["month_num"], day=1)
)

# Sort immediately: every time-series operation depends on order
flights = flights.sort_values("date").reset_index(drop=True)

# Roll up to annual totals for the simpler trend examples
annual = flights.groupby("year")["passengers"].sum().reset_index()
annual["date"] = pd.to_datetime(annual["year"].astype(str))
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports and theme</span>
    </div>
    <div class="code-callout__body">
      <p>Set <code>sns.set_theme</code> once at the top so all charts in the session share the same grid, font, and colour palette.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Build a datetime column</span>
    </div>
    <div class="code-callout__body">
      <p>The flights dataset stores months as strings ("Jan", "Feb"…). Map them to integers first, then use <code>pd.to_datetime(dict(...))</code> to build a proper date. This enables resampling, rolling windows, and time-based groupby.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-24" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sort immediately</span>
    </div>
    <div class="code-callout__body">
      <p>Always sort by date right after parsing. Rolling averages, resampling, and line plots all assume chronological order, unsorted data produces wrong results silently.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="26-27" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Annual rollup</span>
    </div>
    <div class="code-callout__body">
      <p>Aggregating to annual totals reduces 144 monthly rows to 12 yearly rows, cleaner for the first trend chart, where the broad direction matters more than monthly detail.</p>
    </div>
  </div>
</aside>
</div>

## Choose the right time grain

Match the aggregation level to the question being asked.

| Question | Grain to use |
|---|---|
| Is the server behaving right now? | Hourly or minute |
| Is this week better than last? | Daily |
| What is the trend this quarter? | Weekly |
| How did we do this year? | Monthly or quarterly |

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Resample monthly data to quarterly sums
quarterly = (
    flights
    .set_index("date")
    .resample("QS")                         # QS = quarter start
    .agg(passengers=("passengers", "sum"))
    .reset_index()
)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">resample</span>
    </div>
    <div class="code-callout__body">
      <p><code>resample</code> requires the date column to be the index. <code>"QS"</code> means quarter-start, each output row represents one quarter. Other useful offsets: <code>"W"</code> (weekly), <code>"MS"</code> (month-start), <code>"A"</code> (annual).</p>
    </div>
  </div>
</aside>
</div>

Two rules before publishing any time chart:

1. Do not compare a partial current period to complete prior periods, either exclude it or label it clearly.
2. Explain missing periods instead of silently skipping them.

## Basic time series patterns

### 1. Trend line

The simplest useful time chart: one line, clear axes, no clutter.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    annual["date"],
    annual["passengers"],
    color="#2b8cbe",
    linewidth=2.5,
    marker="o",     # dot at each data point
    markersize=5
)

ax.set_title("Annual Airline Passengers (1949-1960)")
ax.set_xlabel("Year")
ax.set_ylabel("Total Passengers")
ax.grid(True, alpha=0.3)   # subtle grid - don't fight the data
{% endhighlight %}


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Figure size</span>
    </div>
    <div class="code-callout__body">
      <p><code>figsize=(11, 5)</code> gives a wide, shallow aspect ratio, the standard for time series because it makes horizontal trends easier to follow than a square chart.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="3-10" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Markers on a line</span>
    </div>
    <div class="code-callout__body">
      <p><code>marker="o"</code> places a dot at each data point. With annual data (12 points) the dots help the reader count years. With hundreds of daily points, omit them to avoid clutter.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-15" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Grid opacity</span>
    </div>
    <div class="code-callout__body">
      <p><code>alpha=0.3</code> makes the grid lines faint. The grid should help the reader estimate values, it should never compete with the data line for attention.</p>
    </div>
  </div>
</aside>
</div>

![Time series trend line](assets/ts_trend_line.png)

The upward slope is immediately obvious. A plain line is usually the right starting point, add complexity only when the data demands it.

### 2. Rolling average

Raw data often has short-term noise that hides the longer trend. A rolling average smooths this out.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Add a 3-year rolling average column
annual["rolling_3y"] = (
    annual["passengers"]
    .rolling(window=3, min_periods=1)
    .mean()
)

fig, ax = plt.subplots(figsize=(11, 5))

# Raw series: lighter, thinner, stays visible
ax.plot(annual["date"], annual["passengers"],
        color="#9ecae1", linewidth=1.5, label="Annual passengers")

# Smoothed series: darker, thicker, carries the trend message
ax.plot(annual["date"], annual["rolling_3y"],
        color="#08519c", linewidth=2.5, label="3-year rolling average")

ax.set_title("Smoothing with a Rolling Average")
ax.set_xlabel("Year")
ax.set_ylabel("Total Passengers")
ax.legend()
ax.grid(True, alpha=0.3)
{% endhighlight %}


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">rolling().mean()</span>
    </div>
    <div class="code-callout__body">
      <p><code>window=3</code> averages the current year and the two preceding years. <code>min_periods=1</code> prevents the first two rows from becoming NaN, useful when your series starts at a hard boundary.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-16" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Two-line contrast</span>
    </div>
    <div class="code-callout__body">
      <p>Make the raw series light and thin, the smoothed series dark and thick. The reader immediately understands which carries the main message and which is context.</p>
    </div>
  </div>
</aside>
</div>

![Rolling average](assets/ts_rolling_average.png)

Keep the original series visible, if you only show the smoothed line, you hide information that may matter (like a sudden spike or a missing period).

### 3. Multiple series

Compare groups on the same time axis, but only when the number of lines is manageable (roughly five or fewer). Beyond that, use small multiples.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
fig, ax = plt.subplots(figsize=(11, 5))

# 12 distinct colours: one per month
palette = sns.color_palette("tab10", n_colors=12)

for i, (month_n, month_df) in enumerate(flights.groupby("month_num")):
    label = month_df["month"].iloc[0]      # "Jan", "Feb", …
    ax.plot(
        month_df["year"],
        month_df["passengers"],
        color=palette[i],
        linewidth=1.5,
        label=label,
        marker="o", markersize=3
    )

ax.set_title("Passengers by Month Across Years")
ax.set_xlabel("Year")
ax.set_ylabel("Passengers")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
{% endhighlight %}


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Colour palette</span>
    </div>
    <div class="code-callout__body">
      <p><code>sns.color_palette("tab10", n_colors=12)</code> returns 12 visually distinct colours. Access them by index inside the loop so each month gets a consistent colour throughout the session.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-14" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">groupby loop</span>
    </div>
    <div class="code-callout__body">
      <p>Looping over <code>groupby("month_num")</code> gives one subset per month in numeric order. <code>month_df["month"].iloc[0]</code> retrieves the abbreviated month name for the legend label.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-20" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compact legend</span>
    </div>
    <div class="code-callout__body">
      <p><code>fontsize=7</code> and <code>ncol=2</code> fit 12 month labels into two columns without overflowing the chart. With this many series, the chart is already at the limit, use small multiples if you need more.</p>
    </div>
  </div>
</aside>
</div>

![Multiple series](assets/ts_multiple_series.png)

With 12 months the legend is already crowded. This is the point where small multiples become the better choice.

## Small multiples for clarity

When you have too many series for one chart, split them into separate panels. The viewer can compare patterns across panels without the lines overlapping.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Group months into seasons
season_map = {
    1:"Winter", 2:"Winter",  3:"Spring",
    4:"Spring", 5:"Spring",  6:"Summer",
    7:"Summer", 8:"Summer",  9:"Autumn",
    10:"Autumn",11:"Autumn", 12:"Winter"
}
flights["season"] = flights["month_num"].map(season_map)

season_data = (
    flights
    .groupby(["year","season"])["passengers"]
    .sum()
    .reset_index()
)

sns.relplot(
    data=season_data,
    kind="line",
    x="year",
    y="passengers",
    col="season",      # one panel per season
    col_wrap=2,        # two panels per row
    height=3.5,
    facet_kws={"sharey": False}   # each panel scales independently
)
{% endhighlight %}


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Season mapping</span>
    </div>
    <div class="code-callout__body">
      <p>A dictionary maps month numbers to season names. <code>flights["month_num"].map(season_map)</code> applies it in one step, no loops needed.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Seasonal aggregation</span>
    </div>
    <div class="code-callout__body">
      <p>Group by year and season, then sum passengers. This collapses the 12-month series into four seasonal series, much easier to show in small multiples.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-25" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">relplot with col</span>
    </div>
    <div class="code-callout__body">
      <p><code>col="season"</code> creates one panel per season automatically. <code>col_wrap=2</code> wraps after two panels so you get a 2×2 grid. <code>sharey=False</code> lets each panel use its own y-axis scale, important when summer totals are much larger than winter totals.</p>
    </div>
  </div>
</aside>
</div>

![Small multiples](assets/ts_small_multiples.png)

Each season gets its own panel. The growth trend is visible in every panel, and summer's higher absolute numbers do not visually dominate the others because `sharey=False` lets each panel scale independently.

## Annotating events

A vertical line at an event date often tells more of the story than the data alone.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(annual["date"], annual["passengers"],
        color="#2b8cbe", linewidth=2.5, marker="o", markersize=5)

# Vertical dashed line at the event date
event_date = pd.Timestamp("1955-01-01")
ax.axvline(event_date, color="#636363", linestyle="--", linewidth=1.8)

# Arrow + label annotation
event_y = annual.loc[annual["year"]==1955, "passengers"].values[0]
ax.annotate(
    "Jet age begins\n(1955)",
    xy=(event_date, event_y),          # arrow tip
    xytext=(pd.Timestamp("1956-06-01"), 8500),  # label position
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#636363"),
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="#fff3cd", alpha=0.9)
)

ax.set_title("Annotating a Key Event on a Time Chart")
ax.set_xlabel("Year")
ax.set_ylabel("Total Passengers")
ax.grid(True, alpha=0.3)
{% endhighlight %}


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="6-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">axvline</span>
    </div>
    <div class="code-callout__body">
      <p><code>ax.axvline</code> draws a vertical line across the full y-axis at the given x-position. Use a dashed style and a neutral grey so it reads as context rather than data.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">annotate</span>
    </div>
    <div class="code-callout__body">
      <p><code>xy</code> is where the arrow points; <code>xytext</code> is where the label sits. Separate them so the label does not overlap the line. <code>bbox</code> adds a light background that makes the text readable over the chart grid.</p>
    </div>
  </div>
</aside>
</div>

![Event annotation](assets/ts_event_annotation.png)

The annotation explains *why* growth accelerated, without it the reader has to guess. Always prefer annotation directly on the chart over a caption below it.

## Plotly for interactive exploration

Plotly is useful when your audience needs to zoom in, hover for exact values, or explore a date range themselves.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import plotly.express as px

fig = px.line(
    annual,
    x="date",
    y="passengers",
    title="Annual Airline Passengers",
    labels={"date": "Year", "passengers": "Total Passengers"}
)

# Add a range slider below the chart
fig.update_xaxes(rangeslider_visible=True)

# Format y-axis with comma separators
fig.update_yaxes(tickformat=",")

fig.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">px.line</span>
    </div>
    <div class="code-callout__body">
      <p><code>px.line</code> is the Plotly Express equivalent of Matplotlib's <code>ax.plot</code>. The <code>labels</code> dict replaces column names with readable strings in the axis and hover tooltip.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Range slider</span>
    </div>
    <div class="code-callout__body">
      <p><code>rangeslider_visible=True</code> adds a miniature navigator below the x-axis. The viewer can drag the handles to zoom into any time window without writing any code.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-15" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Tick format</span>
    </div>
    <div class="code-callout__body">
      <p><code>tickformat=","</code> adds thousand-separator commas to large numbers (e.g. 15,000 instead of 15000). For percentages use <code>".1%"</code>; for currency use <code>"$,.0f"</code>.</p>
    </div>
  </div>
</aside>
</div>

<div class="plotly-embed" style="width:100%;height:500px;margin:1.5rem 0;">
<iframe src="assets/ts_plotly_rangeslider.html" width="100%" height="500px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

The range slider at the bottom lets the viewer drag to focus on any window. Use Plotly for dashboards and stakeholder reports where the audience will explore the data, use Matplotlib/Seaborn for static reports and presentations.

## Common mistakes

- Comparing an incomplete current period to complete prior periods, the current period will always look worse.
- Connecting points across missing dates without noting the gap.
- Putting more than ~5 series on one chart, use small multiples instead.
- Adding a second y-axis when normalising or using separate panels would be clearer.
- Over-smoothing and hiding important volatility.

## A practical checklist

Before publishing a time chart:

1. Are the dates parsed and sorted?
2. Is the aggregation level appropriate for the question?
3. Are incomplete periods excluded or clearly labelled?
4. Would a rolling average help, and if so, is the raw series still visible?
5. Are key events marked directly on the chart?

## Practice prompts

1. Turn the monthly `flights` data into quarterly totals and compare readability with the monthly version.
2. Add a 6-month rolling average to the monthly series and keep the raw line visible.
3. Replace the 12-line month chart above with a `sns.relplot` small-multiples version grouped by month instead of season.
4. Pick a year and annotate it with a made-up event label. What does the annotation make the viewer assume?

## Gotchas

- **`resample()` silently produces wrong results if the date column is not the index**: you must call `.set_index("date")` before `.resample("MS")`. If you forget, pandas raises a `TypeError`; but if your index happens to be a DatetimeIndex from a prior operation, resample runs without complaint on the wrong column.
- **`rolling(window=3)` without `min_periods=1` fills the first N-1 rows with NaN**: this is the correct statistical behaviour, but it means your smoothed line starts later than your raw line on the chart. The gap looks like missing data to an audience. Always decide explicitly: fill early NaNs with `min_periods=1` or drop them and annotate the chart.
- **`pd.to_datetime()` with ambiguous formats guesses month/day order based on locale**: the string `"01/02/2025"` becomes January 2nd in US locale and February 1st in European locale. Pass `dayfirst=True` or `format="%d/%m/%Y"` explicitly when your data source is ambiguous.
- **A second y-axis makes percentage-point differences appear identical to absolute differences**: synchronising two y-axes is technically available in both Matplotlib and Plotly, but the visual scale mismatch routinely misleads audiences into thinking two trends are more correlated than they are. Use separate panels or normalise to a common scale instead.
- **Comparing a partial current period to full prior periods makes recency look worse**: if the current month is only half over, its bar or line point will sit below prior complete months even if the pace is identical. Always exclude incomplete periods or label them with "(partial)" directly on the chart.
- **`line_shape='spline'` in Plotly interpolates between data points, inventing values that never existed**: a spline curve implies continuity and can show a dip or peak between two real points that isn't in your data. Use `line_shape='linear'` for factual time series; spline is for aesthetics only.

## Next steps

1. [Plotly guide](plotly-guide.md), richer interactive controls for time exploration.
2. [Real-world case study](real-world-case-study.md), combine time trends with category and distribution views.
3. [3.4 Data storytelling](../3.4-data-storytelling/README.md), turn a time-based analysis into a stakeholder narrative.
