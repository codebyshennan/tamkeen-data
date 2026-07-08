# Interactive Visualization with Plotly

**After this lesson:** you can explain Interactive Visualization with Plotly and try the examples in your own notebook.

> **Note:** This lesson is **code-first**. Comfortable **pandas** plots or [Seaborn](seaborn-guide.md) help, but you can follow along if you know Matplotlib basics.

## Introduction

Plotly transforms static visualizations into dynamic, interactive web experiences. Think of it as giving your audience a visualization they can explore, not just view.

### Video Tutorial: Plotly Interactive Visualization

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/GGL6U0k8WYA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*Plotly Express - Interactive Visualization made easy in Python*

```yaml
Key Features:
┌─────────────────────────┐
│ Interactivity         │ → Zoom, pan, hover, click
├─────────────────────────┤
│ Web Integration       │ → HTML/JavaScript output
├─────────────────────────┤
│ Real-time Updates     │ → Dynamic data handling
└─────────────────────────┘
```

## Getting Started

{% include mermaid-diagram.html src="3-data-visualization/3.2-adv-data-viz/diagrams/plotly-guide-1.mmd" %}

*Rule of thumb: always try `px` first. Drop to `go` only when Express can't express what you need.*

### Professional Setup

> **Level:** Beginner, run this cell once per notebook session.

Import Plotly Express and Graph Objects, set a default template and size, and initialize notebook mode for inline HTML.

`pio.templates.default` sets global styling. `init_notebook_mode(connected=True)` loads JS from CDN in Jupyter.

```python
import plotly.express as px      # High-level interface
import plotly.graph_objects as go # Low-level interface
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def setup_plotly_environment():
    """Configure professional Plotly defaults"""
    import plotly.io as pio

    # Set template
    pio.templates.default = "plotly_white"

    # Configure default width and height
    pio.defaults.width = 900
    pio.defaults.height = 500

    # For Jupyter notebooks
    from plotly.offline import init_notebook_mode
    init_notebook_mode(connected=True)

setup_plotly_environment()
```

### Data Loading & Inspection

Load Plotly's bundled Gapminder dataset, drop nulls, and add a log GDP column.

`px.data.gapminder()` is bundled with Plotly, no download needed. The same pattern works for `px.data.tips()`, `px.data.iris()`, and others.

```python
def load_and_prepare_data(dataset_name="gapminder"):
    """Load and prepare dataset for visualization"""
    # Load dataset
    if dataset_name == "gapminder":
        df = px.data.gapminder()
    else:
        df = px.data.__getattribute__(dataset_name)()

    # Basic cleaning
    df = df.dropna()

    # Add derived columns if needed
    if "gdpPercap" in df.columns:
        df["log_gdp"] = np.log10(df["gdpPercap"])

    return df

# Example usage
df = load_and_prepare_data()
```



## Basic Interactive Plots

> **Level:** Beginner, these are the core Plotly Express patterns you will use most often.

### 1. Enhanced Scatter Plots

Build a scatter with size and color encodings, optional animation by year, formatted hovers, and a centered title.

`px.scatter` handles all encoding in one call. `update_layout` controls the surrounding canvas, title position, background color, gridline style.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def create_interactive_scatter(data, x_col, y_col,
                             size_col=None, color_col=None,
                             animation_col=None):
    """Create professional interactive scatter plot"""

    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        animation_frame=animation_col,
        # Enhanced hover information
        hover_data={
            x_col: ':.2f',
            y_col: ':.2f',
            size_col: ':.0f' if size_col else False
        },
        # Professional labels
        labels={
            x_col: x_col.replace('_', ' ').title(),
            y_col: y_col.replace('_', ' ').title(),
            color_col: color_col.replace('_', ' ').title() if color_col else None
        }
    )

    # Enhance layout
    fig.update_layout(
        title={
            'text': f'{y_col.title()} vs {x_col.title()}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        # Clean, professional look
        paper_bgcolor='white',
        plot_bgcolor='white',
        # Add subtle grid
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
    )

    return fig

# Example usage
scatter_fig = create_interactive_scatter(
    df,
    x_col='gdpPercap',
    y_col='lifeExp',
    size_col='pop',
    color_col='continent',
    animation_col='year'
)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Function signature</span>
    </div>
    <div class="code-callout__body">
      <p>The function accepts a DataFrame plus column name strings for each visual channel. Default <code>None</code> for optional channels means they're skipped when not supplied, the same function works for a simple two-axis scatter or a fully-animated bubble chart.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Multiple encoding channels</span>
    </div>
    <div class="code-callout__body">
      <p><code>px.scatter</code> maps data columns to visual channels: <code>x</code>/<code>y</code> for position, <code>size</code> for bubble area, <code>color</code> for hue, and <code>animation_frame</code> for a time slider. Each additional channel encodes another variable, <em>without</em> extra code.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-25" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Hover and axis labels</span>
    </div>
    <div class="code-callout__body">
      <p><code>hover_data</code> controls which columns appear on hover and how they're formatted (<code>':.2f'</code> = two decimal places). <code>labels</code> renames columns for display, replacing underscores and title-casing so axes read cleanly without touching the data.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-51" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">update_layout: title + background + grid</span>
    </div>
    <div class="code-callout__body">
      <p><code>update_layout</code> takes a dict for <code>title</code> (position, font) and separate <code>paper_bgcolor</code> (outer canvas) vs <code>plot_bgcolor</code> (inner axes area). Grid styling goes inside <code>xaxis=dict(...)</code>, Plotly's fine-grained axis control.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="52-63" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Return and example call</span>
    </div>
    <div class="code-callout__body">
      <p>Returning the <code>fig</code> object lets callers save, display, or chain further <code>update_*</code> calls. The example call demonstrates the full parameter set: Gapminder GDP vs life expectancy with population size, continent color, and an animated year slider.</p>
    </div>
  </div>
</aside>
</div>

**Output (Animation Frames):**
<div class="plotly-embed" style="width:100%;height:500px;margin:1.5rem 0;">
<iframe src="assets/plotly_animated_scatter.html" width="100%" height="500px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

### 2. Time Series Visualization

Interactive line chart with optional range slider and quick "YTD vs full range" relayout buttons.

`update_xaxes(rangeslider_visible=True)` adds the mini navigator below the chart. `updatemenus` injects `relayout` buttons that update the x-axis range without reloading the page.

```python
def create_time_series_plot(data, x_col, y_col,
                           group_col=None, add_range_slider=True):
    """Create interactive time series visualization"""

    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        color=group_col,
        # Line styling
        line_shape='spline',  # Smooth lines
        render_mode='svg',    # Crisp lines
        # Enhanced hover
        hover_data={
            x_col: True,
            y_col: ':.2f'
        }
    )

    # Add range slider if requested
    if add_range_slider:
        fig.update_xaxes(rangeslider_visible=True)

    # Add helpful buttons
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                # YTD button
                {'args': [{'xaxis.range': [
                    data[x_col].max() - pd.Timedelta(days=365),
                    data[x_col].max()
                ]}],
                 'label': 'YTD',
                 'method': 'relayout'},
                # Reset button
                {'args': [{'xaxis.range': [None, None]}],
                 'label': 'All',
                 'method': 'relayout'}
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.1
        }]
    )

    return fig

# Example usage
time_fig = create_time_series_plot(
    df,
    x_col='year',
    y_col='lifeExp',
    group_col='continent'
)
```

**Output (with Range Slider):**
<div class="plotly-embed" style="width:100%;height:500px;margin:1.5rem 0;">
<iframe src="assets/plotly_timeseries_rangeslider.html" width="100%" height="500px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

> **Try it**
>
> Using `px.data.gapminder()`:
>
> 1. Create a scatter of `gdpPercap` vs `lifeExp` for the year 2007 only (filter first: `df[df.year == 2007]`). Add `size="pop"` and `color="continent"`. What does bubble size show that the axes alone don't?
> 2. Add `animation_frame="year"` to the same chart. Run the animation, which continent shows the most change between 1952 and 2007?
> 3. Try adding `log_x=True` to the scatter. Does the relationship look more linear? Why might that be?

## Statistical Visualizations

> **Level:** Intermediate, uses `go` (Graph Objects) directly for finer control over subplot layouts.

### 1. Distribution Analysis

Four-cell dashboard: histogram, box, violin, and normal QQ plot using Graph Objects subplots.

`make_subplots` builds the grid. `go.Histogram`/`Box`/`Violin`/`Scatter` add traces. `scipy.stats.probplot` feeds QQ points, ensure `scipy` is installed (`pip install scipy`).

```python
def create_distribution_dashboard(data, numeric_col,
                                group_col=None):
    """Create comprehensive distribution analysis"""

    # Create subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Histogram with KDE',
            'Box Plot',
            'Violin Plot',
            'QQ Plot'
        )
    )

    # 1. Histogram with KDE
    fig.add_trace(
        go.Histogram(
            x=data[numeric_col],
            name='Histogram',
            nbinsx=30,
            showlegend=False
        ),
        row=1, col=1
    )

    # 2. Box Plot
    fig.add_trace(
        go.Box(
            x=data[group_col] if group_col else None,
            y=data[numeric_col],
            name='Box Plot',
            boxpoints='outliers'
        ),
        row=1, col=2
    )

    # 3. Violin Plot
    fig.add_trace(
        go.Violin(
            x=data[group_col] if group_col else None,
            y=data[numeric_col],
            name='Violin Plot',
            box_visible=True,
            meanline_visible=True
        ),
        row=2, col=1
    )

    # 4. QQ Plot
    from scipy import stats
    qq_x, qq_y = stats.probplot(data[numeric_col], dist='norm')
    fig.add_trace(
        go.Scatter(
            x=qq_x[0],
            y=qq_y[0],
            mode='markers',
            name='QQ Plot'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"Distribution Analysis: {numeric_col}",
        title_x=0.5
    )

    return fig

# Example usage
dist_fig = create_distribution_dashboard(
    df,
    numeric_col='lifeExp',
    group_col='continent'
)
```

**Output:**
<div class="plotly-embed" style="width:100%;height:700px;margin:1.5rem 0;">
<iframe src="assets/plotly_distribution_dashboard.html" width="100%" height="700px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

### Additional Interactive Features

**3D Scatter Plot:**
<div class="plotly-embed" style="width:100%;height:500px;margin:1.5rem 0;">
<iframe src="assets/plotly_3d_scatter.html" width="100%" height="500px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

**Dashboard Layout:**
<div class="plotly-embed" style="width:100%;height:700px;margin:1.5rem 0;">
<iframe src="assets/plotly_dashboard_layout.html" width="100%" height="700px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

**Interactive Hover Information:**
<div class="plotly-embed" style="width:100%;height:500px;margin:1.5rem 0;">
<iframe src="assets/plotly_hover_example.html" width="100%" height="500px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

**Hierarchical Visualization (Sunburst-style):**
<div class="plotly-embed" style="width:100%;height:500px;margin:1.5rem 0;">
<iframe src="assets/plotly_sunburst.html" width="100%" height="500px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

**Theme Comparison:**
<div class="plotly-embed" style="width:100%;height:500px;margin:1.5rem 0;">
<iframe src="assets/plotly_theme_comparison.html" width="100%" height="500px" frameborder="0" style="border:none;border-radius:4px;"></iframe>
</div>

> **Try it**
>
> Using `px.data.tips()`:
>
> 1. Run `create_distribution_dashboard` on `total_bill`. Compare the histogram and violin, what does each one tell you that the other doesn't?
> 2. Use `px.box(tips, x="day", y="total_bill", color="time", points="outliers")`. Which day has the most outliers? Does time of day (lunch vs dinner) change the pattern?
> 3. Add `facet_col="sex"` to the box plot. Does the day-of-week pattern hold for both sexes?

## Advanced Features

> **Level:** Advanced, custom themes and interactive controls. Use these when presentation quality matters or when your audience needs to filter data.

### 1. Custom Themes

Toggle light vs dark presentation themes by updating paper/plot background, font, and gridline colors.

```python
def apply_custom_theme(fig, theme='modern'):
    """Apply professional custom theme"""

    themes = {
        'modern': {
            'bgcolor': 'white',
            'font_family': 'Arial',
            'grid_color': 'lightgray',
            'colorscale': 'Viridis'
        },
        'dark': {
            'bgcolor': '#1f2630',
            'font_family': 'Helvetica',
            'grid_color': '#3b4754',
            'colorscale': 'Plasma'
        }
    }

    theme_settings = themes.get(theme, themes['modern'])

    fig.update_layout(
        # Background
        paper_bgcolor=theme_settings['bgcolor'],
        plot_bgcolor=theme_settings['bgcolor'],

        # Font
        font=dict(
            family=theme_settings['font_family']
        ),

        # Grid
        xaxis=dict(
            gridcolor=theme_settings['grid_color'],
            zerolinecolor=theme_settings['grid_color']
        ),
        yaxis=dict(
            gridcolor=theme_settings['grid_color'],
            zerolinecolor=theme_settings['grid_color']
        )
    )

    return fig
```



### 2. Interactive Features

Standardize hover behavior, modebar styling, and add simple restyle buttons for trace visibility.

`hovermode='closest'` snaps the tooltip to the nearest point. `updatemenus` with `method='restyle'` toggles `visible` arrays, align the indices with `fig.data`.

```python
def add_interactive_features(fig):
    """Add professional interactive features"""

    fig.update_layout(
        # Hover mode
        hovermode='closest',

        # Modebar
        modebar=dict(
            bgcolor='rgba(0,0,0,0)',
            color='gray',
            activecolor='black'
        ),

        # Add buttons
        updatemenus=[
            # Zoom buttons
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True, True]}],
                        label="Show All",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Hide Trend",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )

    return fig
```

## Best Practices

### 1. Performance Optimization

Downsample traces that exceed a point budget and prefer WebGL markers for large scatters.

This mutates `trace.x`/`y` in place, copy the figure first if you need the full data elsewhere.

```python
def optimize_for_web(fig, max_points=1000):
    """Optimize figure for web performance"""

    # Reduce number of points if necessary
    if len(fig.data[0].x) > max_points:
        step = len(fig.data[0].x) // max_points
        for trace in fig.data:
            trace.x = trace.x[::step]
            trace.y = trace.y[::step]

    # Enable WebGL for large datasets
    fig.update_traces(
        mode='webgl',
        marker=dict(
            size=6,
            opacity=0.7
        )
    )

    return fig
```

### 2. Export Settings

Export self-contained HTML (CDN JS) and static images when `kaleido` is available.

`write_html` for sharing interactively. `write_image` needs the `kaleido` package (`pip install kaleido`) for PNG/SVG/PDF.

```python
def export_figure(fig, filename, formats=None):
    """Export figure in multiple formats"""

    if formats is None:
        formats = ['html', 'png', 'svg']

    for fmt in formats:
        if fmt == 'html':
            fig.write_html(
                f"{filename}.html",
                include_plotlyjs='cdn',
                include_mathjax='cdn'
            )
        elif fmt in ['png', 'svg', 'pdf']:
            fig.write_image(
                f"{filename}.{fmt}",
                width=1200,
                height=800,
                scale=2  # Retina display
            )
```

Remember:

- Start with Plotly Express for quick results
- Use Graph Objects for fine-grained control
- Consider performance with large datasets
- Test interactivity across browsers
- Include clear documentation for users

## Gotchas

- **`fig.write_image()` fails silently without `kaleido` installed**: Plotly's static image export depends on the `kaleido` package (`pip install kaleido`). Without it the call raises a `ValueError` that looks like a Plotly bug, not a missing dependency. Install `kaleido` before expecting PNG/SVG/PDF output.
- **`animation_frame` requires the column to be pre-sorted**: `px.scatter` with `animation_frame="year"` will animate frames in the order the values appear in the DataFrame, not numerically. If your data isn't sorted by year, frames can jump backwards in time. Sort before calling `px.scatter`.
- **`pio.templates.default` is a global setting that persists across all cells in the session**: setting `pio.templates.default = "plotly_white"` in a setup cell means every subsequent figure uses that template even in unrelated notebooks. If you share notebook cells, the template affects colleagues' charts too unless they reset it.
- **`update_traces(mode='webgl')` silently ignores incompatible trace types**: `mode='webgl'` is only valid for `Scattergl` traces. Applying it to a `Bar` or `Histogram` trace via `update_traces` doesn't raise an error but has no effect; you need `go.Scattergl` from the start for WebGL rendering.
- **`hovermode="x unified"` collapses tooltips in a way that hides individual trace names for closely spaced points**: when multiple traces share the same x value, unified hover can truncate or merge labels unpredictably. Test with your real data density before shipping to stakeholders.
- **`make_subplots` row/col indexing is 1-based, not 0-based**: passing `row=0` raises a `ValueError`. Learners coming from NumPy arrays consistently start at 0, which fails here. Always start subplot coordinates at `row=1, col=1`.

## Next steps

- Pair this with [Seaborn](seaborn-guide.md) when you need strong statistical defaults in Python.
- Move to [3.3 BI with Tableau](../3.3-bi-with-tableau/README.md) for drag-and-drop dashboards, or [3.4 Data storytelling](../3.4-data-storytelling/README.md) for narrative structure.
