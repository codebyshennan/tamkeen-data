# Getting Started with Matplotlib 📊

## 🎯 Introduction to Matplotlib

Matplotlib is Python's foundational data visualization library, offering a powerful and flexible system for creating publication-quality plots. Think of it as your digital canvas where data comes to life through visual representation.

## 🚀 Basic Plotting

### Setting Up Your Environment
```python
# Essential imports
import matplotlib.pyplot as plt
import numpy as np

# For Jupyter notebooks
%matplotlib inline

# Set style for better-looking plots
plt.style.use('seaborn')
```

### Anatomy of a Matplotlib Plot
```
Figure Hierarchy:
┌─────────────────────────┐
│ Figure                  │
│  ┌─────────────────┐   │
│  │     Axes        │   │
│  │  ┌─────────┐   │   │
│  │  │ Title   │   │   │
│  │  ├─────────┤   │   │
│  │  │ Plot    │   │   │
│  │  │ Area    │   │   │
│  │  └─────────┘   │   │
│  │ X-Label        │   │
│  └─────────────────┘   │
└─────────────────────────┘
```

### Your First Plot
```python
def create_simple_plot():
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data with styling
    ax.plot(x, y, 
           color='#2ecc71',    # Emerald green
           linewidth=2,        # Thicker line
           linestyle='-',      # Solid line
           label='sin(x)')     # Legend label
    
    # Customize the plot
    ax.set_title('My First Plot', 
                fontsize=14, 
                pad=15)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return fig, ax
```

## 🎨 The Two Interfaces

### 1. MATLAB-style (pyplot)
```python
# Quick and simple plots
def pyplot_example():
    plt.figure(figsize=(10, 6))
    plt.plot([1, 2, 3], [1, 2, 3], 'ro-')
    plt.title('Pyplot Interface')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
```

### 2. Object-Oriented
```python
# More control and better for complex plots
def object_oriented_example():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3], [1, 2, 3], 'bo-')
    ax.set_title('Object-Oriented Interface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    return fig, ax
```

## 📊 Essential Plot Types

### 1. Line Plots
```python
def create_line_plot(x, y1, y2):
    """Create a professional line plot with multiple series"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot multiple lines
    ax.plot(x, y1, 
           color='#3498db',  # Blue
           label='Series 1', 
           linewidth=2)
    ax.plot(x, y2, 
           color='#e74c3c',  # Red
           label='Series 2', 
           linewidth=2)
    
    # Customize
    ax.set_title('Multi-Series Line Plot')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    
    return fig, ax
```

### 2. Scatter Plots
```python
def create_scatter_plot(x, y, colors, sizes):
    """Create an informative scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(x, y,
                        c=colors,           # Color mapping
                        s=sizes,            # Size mapping
                        alpha=0.6,          # Transparency
                        cmap='viridis')     # Color scheme
    
    # Add colorbar
    plt.colorbar(scatter, label='Value')
    
    # Customize
    ax.set_title('Scatter Plot with Color and Size Encoding')
    ax.set_xlabel('X Variable')
    ax.set_ylabel('Y Variable')
    
    return fig, ax
```

### 3. Bar Charts
```python
def create_bar_chart(categories, values, errors=None):
    """Create a professional bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars = ax.bar(categories, 
                  values,
                  yerr=errors,          # Error bars
                  capsize=5,            # Error bar caps
                  color='#2ecc71',      # Bar color
                  alpha=0.8)            # Transparency
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:,.0f}',
                ha='center', va='bottom')
    
    # Customize
    ax.set_title('Bar Chart with Error Bars')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    
    return fig, ax
```

## 🎨 Styling and Customization

### Color Palettes
```python
# Professional color schemes
color_schemes = {
    'main_colors': ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6'],
    'pastel_colors': ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5', '#ff8b94'],
    'grayscale': ['#212121', '#424242', '#616161', '#757575', '#9e9e9e']
}

# Example usage
def apply_color_scheme(ax, scheme='main_colors'):
    """Apply a professional color scheme to a plot"""
    for i, line in enumerate(ax.lines):
        line.set_color(color_schemes[scheme][i % len(color_schemes[scheme])])
```

### Text Styling
```python
def style_text(ax, title_size=14, label_size=12):
    """Apply professional text styling"""
    ax.set_title(ax.get_title(), 
                fontsize=title_size,
                pad=15,
                fontweight='bold')
    
    ax.set_xlabel(ax.get_xlabel(),
                 fontsize=label_size,
                 labelpad=10)
    
    ax.set_ylabel(ax.get_ylabel(),
                 fontsize=label_size,
                 labelpad=10)
    
    ax.tick_params(labelsize=10)
```

## 📐 Subplots and Layouts

### Creating Multiple Plots
```python
def create_dashboard():
    """Create a dashboard with multiple plots"""
    # Create figure with grid
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Add subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Line plot
    ax2 = fig.add_subplot(gs[0, 1])  # Scatter plot
    ax3 = fig.add_subplot(gs[1, :])  # Bar plot spanning bottom
    
    # Style each subplot
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle=':', alpha=0.7)
        style_text(ax)
    
    return fig, (ax1, ax2, ax3)
```

## 💾 Saving and Exporting

### High-Quality Exports
```python
def save_plot(fig, filename, dpi=300):
    """Save plot in multiple formats with best practices"""
    # Save as PNG for web
    fig.savefig(f'{filename}.png',
                dpi=dpi,
                bbox_inches='tight',
                transparent=True)
    
    # Save as PDF for print
    fig.savefig(f'{filename}.pdf',
                bbox_inches='tight')
    
    # Save as SVG for editing
    fig.savefig(f'{filename}.svg',
                bbox_inches='tight')
```

## ✅ Best Practices

### 1. Setup Template
```python
def setup_plot_template():
    """Create a template for consistent plotting"""
    # Set style
    plt.style.use('seaborn')
    
    # Create figure with reasonable size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set DPI for clear display
    plt.rcParams['figure.dpi'] = 100
    
    # Set font family
    plt.rcParams['font.family'] = 'sans-serif'
    
    return fig, ax
```

### 2. Memory Management
```python
def plot_with_memory_management():
    """Plot with proper memory cleanup"""
    try:
        # Create plot
        fig, ax = plt.subplots()
        # ... plotting code ...
        
        # Save or display
        plt.show()
        
    finally:
        # Clean up
        plt.close('all')
```

Remember: The key to effective visualization with Matplotlib is to start simple and gradually add complexity as needed. Focus on clarity and purpose in your visualizations, and always consider your audience when making design decisions.
