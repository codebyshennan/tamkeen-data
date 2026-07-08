# Data Visualization: A Beginner's Guide

**After this lesson:** you can explain Data Visualization: A Beginner's Guide and try the examples in your own notebook.

## Helpful video

Orientation for the course visualization materials.

## Prerequisites

* You can run short Python snippets or read charts in a slide deck; no advanced statistics required.
* Optional: [Quick start](quick-start.md) if you want a hands-on Matplotlib warm-up first.

## What is Data Visualization?

Think of data visualization like turning numbers into pictures. Just like how a photograph can tell a story better than a description, a good visualization helps us understand data better than looking at rows of numbers.

### Real-World Analogy

Imagine you're planning a road trip. You could read a list of distances between cities, or you could look at a map. The map (visualization) makes it instantly clear which route is shortest, where the mountains are, and which cities are close together. That's exactly what data visualization does for numbers!

### Why This Matters

* **Quick Understanding**: Spot patterns in seconds that might take hours to find in spreadsheets
* **Better Decisions**: Make informed choices by seeing the big picture
* **Clear Communication**: Share insights with others who might not be comfortable with raw data
* **Problem Solving**: Identify issues and opportunities more easily

## Your First Steps in Data Visualization

### 1. Understanding Your Data

Before you start visualizing, ask yourself:

* What story do you want to tell?
* Who is your audience?
* What type of data do you have? (numbers, categories, time-based, etc.)

### 2. Choosing the Right Chart

Think of charts like different types of maps:

* **Line Charts** are like road maps showing how things change over time
* **Bar Charts** are like comparing heights of buildings
* **Pie Charts** are like slicing a pizza to show portions
* **Scatter Plots** are like plotting stars on a night sky map

## Basic Chart Types (With Real Examples)

### 1. Line Chart

**Purpose:** Plot an ordered category (weekday) against a numeric measure (steps) to see day-to-day variation.

**Walkthrough:** `plot` with `marker='o'` emphasizes discrete days; grid and title explain units.

<figure><img src="../../.gitbook/assets/beginners-guide_fig_2.png" alt="beginners-guide"><figcaption><p>Figure 2: My Daily Steps This Week</p></figcaption></figure>

Import

`matplotlib.pyplot` is the only dependency for this basic chart, no additional libraries needed.

Data Setup

Parallel lists for weekday labels and step counts, the simplest way to define x/y data for Matplotlib.

Styled Line Chart

`marker='o'` adds dots at each day; hex color and `linewidth=2` improve readability over the default thin grey line.

<figure><img src="../../.gitbook/assets/beginners-guide_fig_3.png" alt="beginners-guide"><figcaption><p>Figure 3: My Daily Steps This Week</p></figcaption></figure>

**When to use:**

* Tracking daily activities
* Monitoring progress over time
* Comparing trends

### 2. Bar Chart

**Purpose:** Compare counts across unordered categories (flavors) with bar height as the encoding.

**Walkthrough:** `bar` takes parallel lists of labels and values; per-bar `color` is optional; `xticks(rotation=45)` avoids label overlap.

<figure><img src="../../.gitbook/assets/beginners-guide_fig_3.png" alt="beginners-guide"><figcaption><p>Figure 3: Favorite Ice Cream Flavors</p></figcaption></figure>

Import

Only `matplotlib.pyplot` is required for basic categorical bar charts.

Category Data

Parallel lists of flavor names and preference counts-`plt.bar` maps each name to a bar height.

Colored Bars

A list of hex colors assigns a distinct hue to each bar; `xticks(rotation=45)` prevents label overlap on narrow charts.

<figure><img src="../../.gitbook/assets/beginners-guide_fig_4.png" alt="beginners-guide"><figcaption><p>Figure 4: Favorite Ice Cream Flavors</p></figcaption></figure>

**When to use:**

* Comparing quantities
* Showing rankings
* Displaying survey results

### 3. Pie Chart

**Purpose:** Show how a day divides into parts that sum to 100%-appropriate when "share of total" is the question.

**Walkthrough:** `pie` takes magnitudes (hours); `autopct` prints percentages; `colors` overrides default palette.

```python
# A basic pie chart - like showing how you spend your day
import matplotlib.pyplot as plt

# Time spent during the day
activities = ['Sleep', 'Work', 'Free Time', 'Other']
hours = [8, 8, 5, 3]

# Create the chart
plt.figure(figsize=(10, 6))
plt.pie(hours, labels=activities, autopct='%1.1f%%',
        colors=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'])
plt.title('How I Spend My Day', fontsize=14, pad=20)
plt.show()
```

<figure><img src="../../.gitbook/assets/beginners-guide_fig_1.png" alt="beginners-guide"><figcaption><p>Figure 1: How I Spend My Day</p></figcaption></figure>

**When to use:**

* Showing parts of a whole
* Displaying percentages
* Simple comparisons

## Common Mistakes to Avoid

1. **Too Much Information**
   * Don't try to show everything in one chart
   * Keep it simple and focused
   * Like trying to read a map with too many details
2. **Wrong Chart Type**
   * Don't use a pie chart for trends over time
   * Don't use a line chart for unrelated categories
   * Like using a road map when you need a star chart
3. **Missing Labels**
   * Always label your axes
   * Include a clear title
   * Explain what the numbers mean
   * Like a map without street names

## Making Your Charts Better

### 1. Add Colors

**Purpose:** Differentiate bars with fill and outline color instead of relying on the default single color.

**Walkthrough:** `edgecolor` outlines each bar; assumes `months` and `expenses` exist like earlier examples.

```python
# Instead of plain bars
plt.bar(months, expenses, color='skyblue', edgecolor='navy')
```

### 2. Add Some Style

**Purpose:** Apply a bundled Matplotlib style sheet so typography and colors stay consistent across figures.

**Walkthrough:** `plt.style.use('seaborn-v0_8-whitegrid')` selects a named style; run once per session or notebook.

```python
# Make it look nicer
plt.style.use('seaborn-v0_8-whitegrid')  # Uses a pre-made style
```

<figure><img src="../../.gitbook/assets/beginners-guide_fig_1.png" alt="beginners-guide"><figcaption><p>Figure 1: Generated visualization</p></figcaption></figure>

### 3. Add Explanations

**Purpose:** Anchor the chart with a small data-source note in figure coordinates, common in reports.

**Walkthrough:** `figtext` uses 0-1 figure coordinates; `ha='right'` aligns the caption to the bottom-right margin.

```python
# Add a note about the data
plt.figtext(0.99, 0.01, 'Data source: My Budget App',
            ha='right', va='bottom', fontsize=8)
```

<figure><img src="../../.gitbook/assets/beginners-guide_fig_2.png" alt="beginners-guide"><figcaption><p>Figure 2: Generated visualization</p></figcaption></figure>

```
Text(0.99, 0.01, 'Data source: My Budget App')
```

**Captured output (notebook):** The last line may print the `Text` artist returned by `figtext`-that is normal; the annotation still appears on the figure.

## Tips for Beginners

1. **Start Simple**
   * Begin with basic charts
   * Add features one at a time
   * Practice with small datasets
   * Like learning to draw before painting
2. **Use Good Data**
   * Make sure your numbers are correct
   * Keep your data organized
   * Know what your numbers mean
   * Like using accurate measurements in cooking
3. **Tell a Story**
   * What do you want to show?
   * Why is it important?
   * What should people learn?
   * Like writing a good story with a clear message
4. **Get Feedback**
   * Show your charts to others
   * Ask if they understand
   * Make improvements based on feedback
   * Like testing a recipe before serving

## Next steps

1. **Practice With Real Data**
   * Use your own expenses
   * Track daily activities
   * Monitor habits or goals
   * Like keeping a diary of your progress
2. **Learn More Tools**
   * Try different Python libraries
   * Experiment with interactive charts
   * Learn about data cleaning
   * Like learning new cooking techniques
3. **Share Your Work**
   * Create a portfolio
   * Help others visualize their data
   * Join online communities
   * Like sharing your recipes with friends

## Resources for Learning

1. **Free Datasets**
   * Weather data
   * Sports statistics
   * Population data
   * Economic indicators
2. **Online Tools**
   * Google Colab (free Python environment)
   * Kaggle (for practice datasets)
   * DataCamp (for interactive learning)
3. **Books and Courses**
   * "Storytelling with Data" by Cole Nussbaumer Knaflic
   * "The Visual Display of Quantitative Information" by Edward Tufte
   * Coursera's "Data Visualization and Communication with Tableau"

## Common Questions

1. **Which chart should I use?**
   * For trends over time: Line chart
   * For comparing quantities: Bar chart
   * For parts of a whole: Pie chart
   * For relationships: Scatter plot
2. **How do I make my charts look professional?**
   * Use consistent colors
   * Add clear labels
   * Keep it simple
   * Tell a clear story
3. **What tools should I start with?**
   * Begin with matplotlib in Python
   * Try Google Colab for free practice
   * Move to more advanced tools as you grow

Remember: The best visualization is one that helps your audience understand the data quickly and clearly. Start simple, practice often, and don't be afraid to experiment!

## Gotchas

* **`plt.style.use('seaborn-v0_8-whitegrid')` must be called before any plotting**: if you call it after `plt.figure()` or `plt.plot()`, the style applies to the next figure, not the one already open; move style calls to the top of your setup cell.
* **Pie charts with `hours = [8, 8, 5, 3]` hide that slices are unequal even when they look similar**: 8 hours of "Sleep" and 8 hours of "Work" produce identical wedges, making it hard to spot difference; when two categories are close in value, a bar chart communicates the gap far more clearly.
* **Color lists must match the number of bars exactly**: passing `color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']` to `plt.bar` works only when there are exactly 4 bars; adding or removing a category without updating the color list raises a silent mismatch or a `ValueError`.
* **`plt.figtext` coordinates are figure-relative (0-1), not data or axes coordinates**: a note placed at `(0.99, 0.01)` sits at the bottom-right of the whole figure canvas, not the chart area; if your figure has a large bottom margin, the note may appear far below the chart.
* **No x-axis label on bar charts creates ambiguity**: the bar chart examples here label only the y-axis (`'Number of People'`) but omit an x-axis label; viewers who see the chart without the title cannot tell what the categories represent.
* **`marker='o'` on line charts works for small datasets but clutters dense time series**: once you have more than \~20 points on a line, markers overlap and obscure the trend; omit the marker or reduce its size with `markersize=3` for denser data.
* Structured follow-on: [3.1 Intro to data visualization](3.1-intro-data-viz/) and [Choosing the right visualization](choosing-the-right-visualization.md).
