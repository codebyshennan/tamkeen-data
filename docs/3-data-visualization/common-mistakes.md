# Common Data Visualization Mistakes and How to Fix Them

**After this lesson:** you can explain Common Data Visualization Mistakes and How to Fix Them and try the examples in your own notebook.

## Helpful video

Orientation for the course visualization materials.

## Overview

Use this page as a **checklist**: each section names a mistake, gives a fix, and includes a figure placeholder you can replace with your own before/after screenshot.

## 1. Choosing the Wrong Chart Type

### The Mistake

Using an inappropriate chart type that doesn't effectively communicate your data's story. For example:

* Using a pie chart to show changes over time
* Using a line chart for unrelated categories
* Using a 3D chart when 2D would be clearer

### The Solution

Match your chart type to your data and goal:

* **Time-based data**: Use line charts
* **Comparing categories**: Use bar charts
* **Parts of a whole**: Use pie charts (but only for 2-6 segments)
* **Relationships**: Use scatter plots

### Example

![Before vs after: fixing a cluttered chart](../../.gitbook/assets/before_after_example.png) _Left: Raw scattered data points. Right: Same data with a trend line that helps tell the story._

## 2. Overwhelming Your Audience

### The Mistake

* Cramming too much information into one visualization
* Using too many colors or patterns
* Including unnecessary decimal places
* Adding distracting chart elements

### The Solution

* Focus on one main message per visualization
* Use colors purposefully and sparingly
* Round numbers appropriately
* Remove chart junk (gridlines, borders, etc.)

### Example

> **Figure (add screenshot or diagram):** A clean line chart of daily step counts over 30 days, one trend line, no gridlines, a single clear title "Daily Steps (March)", and a minimalist y-axis with rounded tick labels; illustrating the "less is more" principle.

## 3. Poor Color Choices

### The Mistake

* Using too many colors
* Choosing colors that clash
* Not considering color-blind viewers
* Using colors with no meaning

### The Solution

* Stick to a simple color palette
* Use contrasting colors that work well together
* Choose colorblind-friendly palettes
* Make colors meaningful (e.g., red for negative, green for positive)

### Example

> **Figure (add screenshot or diagram):** A pie chart of ice cream flavor preferences using a colorblind-friendly palette (5 distinct colors from the Okabe-Ito or ColorBrewer set), each slice clearly differentiated with no jarring clashes, demonstrating purposeful color use.

## 4. Missing Context

### The Mistake

* No title or unclear title
* Missing axis labels
* No units of measurement
* No source attribution

### The Solution

* Use clear, descriptive titles
* Label all axes and include units
* Add necessary context in subtitles
* Cite your data sources

### Example

> **Figure (add screenshot or diagram):** A donut chart of daily activities (Sleep, Work, Exercise, Leisure) with a descriptive title "How I Spend My Day", each segment labeled with activity name and percentage, and a subtitle citing the data source, demonstrating proper chart context.

## 5. Misleading Scales

### The Mistake

* Starting y-axis at non-zero
* Using inconsistent scales
* Manipulating aspect ratio
* Using deceptive comparisons

### The Solution

* Start y-axis at zero for bar charts
* Use consistent scales when comparing
* Maintain appropriate aspect ratios
* Make fair comparisons

### Example

> **Figure (add screenshot or diagram):** Side-by-side bar charts of "Study Hours vs Grade", left chart with y-axis starting at 60 (misleading, exaggerating differences) and right chart with y-axis starting at 0 (honest scale); illustrating how axis manipulation distorts perception.

## 6. Poor Data-to-Ink Ratio

### The Mistake

* Too much decoration
* Unnecessary 3D effects
* Redundant elements
* Excessive gridlines

### The Solution

* Remove unnecessary elements
* Keep it simple and clean
* Use space effectively
* Include only essential gridlines

## 7. Not Considering Your Audience

### The Mistake

* Using technical jargon
* Assuming domain knowledge
* Not explaining complex concepts
* Ignoring audience needs

### The Solution

* Use plain language
* Provide necessary context
* Explain complex terms
* Consider audience expertise level

## 8. Inconsistent Formatting

### The Mistake

* Different fonts in one visualization
* Inconsistent color schemes
* Varying chart styles
* Mixed formatting

### The Solution

* Use consistent typography
* Maintain a color scheme
* Stick to one style
* Create a style guide

## Tips for Success

1. **Start Simple**
   * Begin with basic charts
   * Add elements gradually
   * Test with your audience
   * Iterate based on feedback
2. **Focus on Clarity**
   * Make it easy to understand
   * Highlight important information
   * Remove distractions
   * Tell a clear story
3. **Be Honest**
   * Present data accurately
   * Don't manipulate scales
   * Show uncertainty
   * Cite your sources
4. **Test Your Visualization**
   * Get feedback from others
   * View on different devices
   * Check for accessibility
   * Verify accuracy

## Gotchas

* **A truncated y-axis is not always dishonest, but omitting a note about it is**: starting a bar chart y-axis at a non-zero value exaggerates differences, but starting a line chart y-axis at a non-zero value is often the right call for time series with small variance; the mistake is not the scale itself but failing to label or note it so viewers do not assume the axis starts at zero.
* **Removing gridlines entirely can make value estimation harder, not easier**: this guide correctly calls out excessive gridlines as clutter, but removing all reference lines forces viewers to guess values; keep one set of faint horizontal gridlines to support reading bar or line heights.
* **"One main message per chart" does not mean one data series**: a scatter plot comparing two variables still has one message (their relationship); the mistake is plotting unrelated questions on the same axes, not plotting multiple related series.
* **Inconsistent formatting across a multi-panel figure is harder to spot than inconsistency across separate charts**: when four subplots share a `fig.suptitle`, it is easy to overlook that one axis uses a different font size or that one bar chart has a grid while others do not; check all panels in the final figure together, not panel by panel.
* **Color meanings that feel "obvious" (red = bad, green = good) create confusion in contexts where both are neutral**: a chart showing revenue growth in green and cost reduction in red reads correctly in English-speaking business contexts but inverts in other settings; always add a label or legend note rather than relying on color convention alone.
* **3D effects on pie or bar charts do not just "look bad", they actively distort values**: a 3D pie chart makes the front slices look larger than rear slices of equal value due to perspective foreshortening; this is not a style preference, it is a perceptual error that misleads viewers.

## Next steps

* Cross-check with [Best practices](best-practices-guide.md) and [Choosing the right visualization](choosing-the-right-visualization.md).
* Return to the [module README](./) for the next lesson in your path.
