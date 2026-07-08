# Frequently Asked Questions (FAQ)

**After this lesson:** you can explain Frequently Asked Questions (FAQ) and try the examples in your own notebook.

Quick answers about libraries, chart choice, Jupyter quirks, and performance. For structured lessons, start with the [module README](README.md) and [3.1 Intro to data visualization](3.1-intro-data-viz/README.md).

## Helpful video

Orientation for the course visualization materials.

<iframe width="560" height="315" src="https://www.youtube.com/embed/RBSUwFGa6Fk" title="What is Data Science?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## General Questions

### Q: Which visualization library should I use?

**A:** Choose based on your needs:
- **Matplotlib**: Basic plotting, complete control, static visualizations
- **Seaborn**: Statistical visualizations, better defaults, built on Matplotlib
- **Plotly**: Interactive visualizations, web integration, dashboards
- **Tableau**: Business intelligence, drag-and-drop interface, enterprise features

### Q: How do I choose the right chart type?

**A:** Consider your data and purpose:
1. **Comparison**:
   - Bar charts for categories
   - Line charts for trends
   - Scatter plots for relationships

2. **Distribution**:
   - Histograms for single variables
   - Box plots for multiple categories
   - Violin plots for detailed distributions

3. **Composition**:
   - Pie charts for parts of a whole
   - Stacked bars for changes over time
   - Treemaps for hierarchical data

### Q: How can I make my visualizations more accessible?

**A:** Follow these guidelines:
1. Use colorblind-friendly palettes
2. Include alternative text descriptions
3. Maintain sufficient contrast
4. Add clear labels and legends
5. Use patterns or shapes alongside colors
6. Ensure text is readable at different sizes

## Technical Questions

### Q: Why is my plot not showing in Jupyter?

**A:** Common solutions:
1. Add `%matplotlib inline` at the start
2. Call `plt.show()` after plotting
3. Check if data is empty or invalid
4. Verify plot commands are executed in order

Example:

**Purpose:** Minimal notebook cell that proves the inline backend and `show()` display a line plot.

**Walkthrough:** Magic first, then imports; `plot` then `show()`-order matches execution.

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 2, 3])
plt.show()
```

### Q: How do I save high-quality plots for publication?

**A:** Use these settings:

**Purpose:** Pick DPI and format for print (vector/PDF, high-DPI PNG) versus web (smaller PNG, optional SVG).

**Walkthrough:** `bbox_inches='tight'` trims margins; `optimize` for PNG where supported.

```python
# For print (PDF)
plt.savefig('plot.pdf', dpi=300, bbox_inches='tight')

# For web (PNG)
plt.savefig('plot.png', dpi=150, bbox_inches='tight', optimize=True)

# For vector graphics (SVG)
plt.savefig('plot.svg', bbox_inches='tight')
```

### Q: How do I handle large datasets in visualizations?

**A:** Several approaches:
1. **Sampling**:

**Purpose:** Plot a random subset of rows so the scatter stays readable and fast.

**Walkthrough:** `np.random.choice` indices without replacement; index both columns consistently.

```python
sample_size = 1000
sample_idx = np.random.choice(len(data), sample_size)
plt.scatter(data[sample_idx, 0], data[sample_idx, 1])
```

2. **Aggregation**:

**Purpose:** Summarize massive `(x,y)` pairs into a 2D heatmap of counts per bin.

**Walkthrough:** `histogram2d` returns counts and bin edges; `pcolormesh` plots the grid.

```python
binned_data = np.histogram2d(x, y, bins=50)
plt.pcolormesh(binned_data[1], binned_data[2], binned_data[0].T)
```

3. **Streaming**:

**Purpose:** Process a CSV in chunks when the full table does not fit in memory.

**Walkthrough:** `chunksize` in `read_csv` yields DataFrames; replace `process_and_plot` with your aggregation.

```python
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    process_and_plot(chunk)
```

## Design Questions

### Q: How do I choose colors for my visualization?

**A:** Follow these principles:
1. **Sequential Data**: Use single color gradient
2. **Categorical Data**: Use distinct colors
3. **Diverging Data**: Use two contrasting colors
4. **Highlight Data**: Use bright color against neutral

Example:

**Purpose:** See how Seaborn named palettes support sequential, categorical, and diverging data.

**Walkthrough:** `color_palette` returns RGB tuples; pass to `scatter`/`bar` or use as `cmap` when appropriate.

```python
# Sequential
colors = sns.color_palette("Blues", n_colors=5)

# Categorical
colors = sns.color_palette("Set2")

# Diverging
colors = sns.color_palette("RdBu", n_colors=11)
```


<figure>
<img src="assets/FAQ_fig_1.png" alt="FAQ" />
<figcaption>Figure 1: Generated visualization</figcaption>
</figure>


### Q: How do I handle overlapping data points?

**A:** Several solutions:
1. **Transparency**:

**Purpose:** Reveal density where many points stack at the same coordinates.

**Walkthrough:** lower `alpha` stacks visually; tune until structure appears without washing out.

```python
plt.scatter(x, y, alpha=0.1)
```

2. **Jittering**:

**Purpose:** Break ties on discrete axes (e.g. Likert scales) so individual points don't sit on one pixel.

**Walkthrough:** Small Gaussian noise on x only (or y); keep sigma small relative to spacing.

```python
x_jitter = x + np.random.normal(0, 0.1, len(x))
plt.scatter(x_jitter, y)
```

3. **2D Histogram**:

**Purpose:** When jitter is not enough, show counts per bin as a heatmap.

**Walkthrough:** `hist2d` is the same family as `histogram2d` but plots directly.

```python
plt.hist2d(x, y, bins=50)
```

### Q: How do I create effective dashboards?

**A:** Key principles:
1. Organize related information together
2. Use consistent styling
3. Provide clear navigation
4. Include interactive filters
5. Optimize performance
6. Test on target devices

## Performance Questions

### Q: Why is my visualization slow?

**A:** Common issues and solutions:
1. **Too much data**:
   - Sample or aggregate data
   - Use appropriate plot types
   - Consider data streaming

2. **Inefficient code**:
   - Use vectorized operations
   - Minimize redundant calculations
   - Cache intermediate results

3. **Resource constraints**:
   - Reduce plot complexity
   - Optimize image resolution
   - Use appropriate file formats

### Q: How do I optimize Tableau dashboards?

**A:** Best practices:
1. Use extracts instead of live connections
2. Limit the number of filters
3. Aggregate data appropriately
4. Use efficient calculations
5. Test with production-size data
6. Monitor performance metrics

## Learning Resources

### Q: Where can I learn more?

**A:** Recommended resources:
1. **Documentation**:
   - Matplotlib, Seaborn, Plotly docs
   - Tableau help center
   - Online tutorials

2. **Books**:
   - "Fundamentals of Data Visualization"
   - "Storytelling with Data"
   - "Python for Data Analysis"

3. **Online Courses**:
   - Coursera Data Visualization
   - DataCamp
   - Tableau Training

4. **Communities**:
   - Stack Overflow
   - GitHub Discussions
   - Tableau Community

### Q: How do I practice visualization skills?

**A:** Suggested approaches:
1. Work with public datasets
2. Participate in visualization challenges
3. Recreate interesting visualizations
4. Contribute to open source projects
5. Create personal data projects
6. Join visualization communities

## Troubleshooting

### Q: Common Error Messages

1. **"No display name and no $DISPLAY environment variable"**:
   - Use `plt.switch_backend('agg')`
   - Configure appropriate backend

2. **"Figure includes Axes that are not compatible with tight_layout"**:
   - Adjust figure size
   - Modify subplot parameters
   - Use `constrained_layout`

3. **"Clipping input data to the valid range"**:
   - Check data ranges
   - Verify calculations
   - Handle outliers appropriately

### Q: Version Compatibility

1. **Library Versions**:
   - Check compatibility matrix
   - Use virtual environments
   - Document dependencies

2. **Operating Systems**:
   - Test on target platforms
   - Use appropriate backends
   - Handle path differences

## Gotchas

- **`%matplotlib inline` must appear before `import matplotlib.pyplot as plt`**: if the magic runs after the import, it does not switch the already-loaded backend; restart the kernel and put `%matplotlib inline` in the first cell to guarantee inline rendering.
- **`plt.savefig('plot.pdf', dpi=300)`, `dpi` is ignored for PDF exports**: PDFs are vector formats and have no native DPI; the `dpi` argument only affects raster elements (e.g. images embedded inside the figure); use `bbox_inches='tight'` to control margins, not DPI.
- **`np.random.choice(len(data), sample_size)` samples indices, not values**: the call returns integer positions that you must then use to index `data`; passing the returned array directly into a plot function will plot the index numbers, not your original data values.
- **`plt.hist2d(x, y, bins=50)` returns a tuple, not a figure**: the function plots the heatmap and also returns `(counts, xedges, yedges, image)`; if you store the return value expecting a figure object and then call `.savefig()` on it, you will get an `AttributeError`.
- **Seaborn palettes used as `cmap=` expect a Colormap object, not a list**: `sns.color_palette("Blues", n_colors=5)` returns a list of RGB tuples; passing it to `cmap=` in `scatter` or `imshow` will raise a `TypeError`; use `sns.color_palette("Blues", as_cmap=True)` or `matplotlib.colors.ListedColormap(...)` to wrap the list.
- **The `"Clipping input data to the valid range"` warning from `imshow` is silent data loss**: it means values outside `[0, 1]` (float) or `[0, 255]` (int) are being clamped before display; normalize your array first with `(arr - arr.min()) / (arr.max() - arr.min())` rather than letting the clip happen invisibly.
