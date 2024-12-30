# Interactive Visualization with Plotly 🎨

## 🎯 Introduction

Plotly transforms static visualizations into dynamic, interactive web experiences. Think of it as giving your audience a visualization they can explore, not just view.

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

## 🚀 Getting Started

### Professional Setup
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

## 📊 Basic Interactive Plots

### 1. Enhanced Scatter Plots
```python
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
```

### 2. Time Series Visualization
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

## 📈 Statistical Visualizations

### 1. Distribution Analysis
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

## 🎨 Advanced Features

### 1. Custom Themes
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

## 🎯 Best Practices

### 1. Performance Optimization
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
