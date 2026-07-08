# Real-World Visualization Case Study

**After this lesson:** you can move from a vague business question to a cleaned dataset, a set of exploratory charts, and a polished final visualization with a clear recommendation.

> **Note:** This lesson is workflow-first. It connects [data prep](../3.1-intro-data-viz/data-prep-for-visualization.md), [Seaborn guide](seaborn-guide.md), and [Plotly guide](plotly-guide.md) into one realistic analysis sequence.

## Scenario

You work for an e-commerce company. The growth team asks:

> "Why did conversion improve in Q2, and which channels should we invest in next?"

You have weekly marketing data with sessions, orders, and revenue broken down by channel and device.

Your job is not just to make charts. Your job is to **answer the question with evidence**.

## Step 1: Define the decision

Break the request into smaller chartable questions before touching any data:

1. Did conversion really improve over time?
2. Which channels contributed most?
3. Was the lift broad-based or concentrated in one device or channel?
4. Is the recommendation about volume, efficiency, or both?

This prevents random chart production and keeps you focused on the answer rather than the aesthetics.

## Step 2: Prepare the data

Aggregate to the right level, week and channel, then compute derived metrics.

Imports and theme

Set the Seaborn theme once at the top so every chart in the session inherits the same look without repeating styling code.

Parse dates immediately

Always convert the date column right after loading. If you skip this, groupby and resample will treat dates as strings and produce incorrect results.

Weekly aggregation

`dt.to_period("W").dt.start_time` snaps each date to its Monday, so every row in the same week gets the same label. Then groupby sums sessions, orders, and revenue per channel per week.

Derived metrics

Conversion rate and revenue per session are computed _after_ aggregation, not before. Computing them at row level and then averaging would give the wrong answer when session counts differ between rows.

## Step 3: Exploratory charts

Use a small set of charts with distinct purposes, one per question from Step 1.

### Chart 1: Overall conversion trend

Question: Did conversion actually improve, or does it just feel that way?

Roll up to overall weekly

Sum across all channels and devices first, then compute the rate. This gives the business-level conversion rate, not a per-channel average.

Line chart

Multiply by 100 to show percentages. `marker="o"` makes individual weeks visible, useful when there are only 20-30 data points.

Event annotation

`axvline` draws the vertical marker; `annotate` adds the label with an arrow. The `bbox` puts a light yellow background behind the text so it reads clearly over the chart.

Percentage formatter

`FuncFormatter` appends the % sign to every y-axis tick automatically, so you never have to hardcode axis labels.

![Overall conversion trend](../../../.gitbook/assets/cs_overall_trend.png)

The dashed line shows when the mobile checkout redesign shipped. Conversion climbed noticeably in the weeks that followed, that is the signal the growth team was asking about.

### Chart 2: Channel comparison

Question: Which channels are strongest on volume and efficiency?

Aggregate across all weeks

Collapse the full period into one row per channel. This gives a stable summary rate, weekly variance would make the comparison hard to read.

Sort before plotting

Always sort a bar chart so the reader's eye moves naturally from shortest to longest (or highest to lowest). An unsorted bar chart forces the reader to do the ranking mentally.

Direct bar labels

Placing the value next to each bar removes the need for the reader to look back at the axis. Use this whenever the exact number matters more than just the relative ranking.

![Channel conversion rate](../../../.gitbook/assets/cs_channel_bar.png)

Email converts at the highest rate despite lower session volume. Paid Search brings the most traffic but at lower efficiency. That tension shapes the recommendation.

### Chart 3: Volume vs efficiency

A bar chart can only show one metric at a time. A scatter plot shows both, volume on one axis, efficiency on the other.

Colour dictionary

Assign colours explicitly to channels so they stay consistent across every chart in the analysis. If Email is green here, it should be green everywhere.

Bubble size as a third dimension

`s=row["orders"] / 3` encodes total orders as bubble size, giving you three variables on one chart: sessions (x), revenue per session (y), and orders (size). Divide by a constant to keep bubbles visually proportionate.

Direct labels

`xytext=(8, 4), textcoords="offset points"` nudges the label a few pixels from the centre of the bubble so it does not overlap the marker.

![Volume vs efficiency scatter](../../../.gitbook/assets/cs_scatter_efficiency.png)

Ideal channels sit in the top-right: high sessions and high revenue per session. Channels in the top-left are efficient but need more traffic investment. Bottom-right means high volume with low return per visit.

### Chart 4: Device and channel heatmap

Question: Was the improvement concentrated in one device-channel combination?

Two-level groupby

Grouping by both channel and device gives one conversion rate per combination, which is exactly what the heatmap needs.

Pivot to matrix form

`pivot` reshapes from long to wide: each device becomes a column, each channel a row. Seaborn's heatmap expects exactly this shape.

Heatmap options

`annot=True` prints the value inside each cell. `fmt=".1f"` limits decimals. `cmap="Blues"` makes darker cells mean higher conversion, intuitive without a legend explanation.

![Channel and device heatmap](../../../.gitbook/assets/cs_device_heatmap.png)

Darker cells = higher conversion. Read across a row to compare devices within a channel. Read down a column to compare channels on the same device.

## Step 4: Identify the actual takeaway

After the four exploratory charts above, the pattern is clear:

* Conversion improved after week 14 (the Q2 redesign)
* Email has the highest conversion rate but limited volume
* Paid Search drives the most sessions but at lower efficiency
* Mobile improved more than Desktop after the redesign

That gives you the skeleton of the recommendation before you touch a final chart.

## Step 5: Build the final visuals

A good final deliverable uses **fewer** charts than the exploration phase. Pick the two or three that make the case most directly.

### Final chart 1: Mobile vs Desktop before and after

Device-level weekly rollup

Same pattern as the overall trend, but grouped by device instead of dropping that dimension. This isolates whether the improvement was device-specific.

Loop over devices

Plotting inside a loop with a colour dictionary keeps Mobile consistently red and Desktop consistently blue across every chart in the deck.

Annotate the event

Placing the annotation at a fixed y-coordinate (`5.5`) keeps it from overlapping either line. Adjust based on the actual value range of your data.

![Mobile vs desktop comparison](../../../.gitbook/assets/cs_device_comparison.png)

Mobile conversion rose sharply after the redesign. Desktop improved only slightly. This is the strongest piece of evidence that the redesign, not external factors, drove the Q2 lift.

### Final chart 2: Interactive Plotly view for stakeholders

When your audience needs to explore exact values by channel, hand them an interactive chart rather than a static one.

px.line with colour grouping

`color="channel"` automatically creates one line per channel and builds the legend. The `labels` dict replaces column names with readable strings in the tooltip and axis.

Unified hover

`hovermode="x unified"` shows all channel values in one tooltip when the cursor is at a given week, much easier to compare than hovering each line individually.

Percentage format

`tickformat=".1%"` tells Plotly to multiply by 100 and append %. If your values are already in percent (e.g. 3.2 not 0.032), use `".1f"` and append the symbol manually in the label.

## How to write the recommendation

Connect each chart to a specific action:

* **"Increase Paid Search budget carefully"**: it drives the most volume, but monitor efficiency (Chart 3 shows it sits below Email on revenue per session).
* **"Protect Email"**: it remains the highest-converting channel even at lower volume (Chart 2).
* **"Continue mobile optimisation"**: the conversion gap between Mobile and Desktop narrowed sharply after the redesign (Chart 5). There is still room to close it further.

That is more useful than "conversion went up."

## Common failure modes

* Starting with a chart type instead of a business question.
* Using more exploratory charts in the final deck than supporting ones.
* Mixing volume and rate metrics without explaining the difference (lots of sessions ≠ high conversion).
* Showing channel performance without normalising for traffic scale.
* Presenting a correlation (redesign → conversion lift) as proof of causation.

## A reusable template

For any real visualization task:

1. Define the decision.
2. Prepare the data at the right level.
3. Explore with several chart types.
4. Choose the 2-3 charts that best support the conclusion.
5. Annotate the final charts.
6. Write the recommendation in plain language.

## Practice prompts

1. Rework this case study with a different dataset, customer support tickets by team and category.
2. Create a final chart set for sales performance by region using the same five-step workflow.
3. Replace the static device comparison with an interactive Plotly version and explain when each format is better.
4. Write a one-paragraph recommendation based on three charts only.

## Gotchas

* **Computing conversion rate before aggregation gives the wrong answer**: if you average per-row rates instead of dividing total orders by total sessions after grouping, high-volume rows get the same weight as low-volume ones. The lesson demonstrates this correctly with derived metrics calculated after `groupby`, but skipping that step is the most common bug in marketing analytics.
* **`dt.to_period("W").dt.start_time` snaps dates to Monday regardless of your actual week definition**: if your business defines weeks starting Sunday or Saturday, the weekly grouping will straddle boundaries and mix data from two calendar weeks. Verify the snap day matches your business calendar before publishing weekly charts.
* **Using the same `redesign_date` variable across multiple code cells requires running them in order**: if a learner runs Step 5's chart cells before Step 2, `redesign_date` is undefined and the annotation call raises a `NameError`. Notebooks don't enforce execution order; always re-run from top when sharing.
* **Bubble size encoded with `s=value / constant` encodes area proportional to orders, but audiences perceive radius**: humans underestimate differences in area and overestimate differences in radius. If the business needs precise comparisons from the bubble chart, add direct labels; don't rely on bubble size alone to convey magnitude.
* **`hovermode="x unified"` in the Plotly channel view flattens all channels to the same tooltip even when they have different values on the same week**: for channels with very similar rates, the unified tooltip shows them stacked in the order traces were added, which may not match the legend order. Confirm the tooltip order matches what you describe in the recommendation.
* **Annotating an event with `axvline` does not prove causation**: the chart visually implies the redesign caused the Q2 lift, but external factors (seasonality, a concurrent campaign) could also explain it. Always note the limitation in the recommendation text, as the "Common failure modes" section of this lesson already flags.

## Next steps

1. [3.4 Data storytelling](../3.4-data-storytelling/), turn case-study evidence into a polished narrative.
2. [Module assignment](../assignments/module-assignment.md), a fuller end-to-end practice task.
3. [Annotations and highlighting](../3.1-intro-data-viz/annotations-and-highlighting.md), if your final charts still need too much verbal explanation.
