# Matplotlib Troubleshooting Guide

## Common Issues and Solutions

### 1. Display Problems

#### Plot Not Showing

Think of this as your TV not turning on:

```python
#  Problem: Your plot is invisible
plt.plot([1, 2, 3], [1, 2, 3])
# Nothing appears

#  Solution 1: Add plt.show() - like pressing the power button
plt.plot([1, 2, 3], [1, 2, 3])
plt.show()

#  Solution 2: For Jupyter - like setting up your TV
%matplotlib inline
plt.plot([1, 2, 3], [1, 2, 3])
```

#### Backend Issues

Think of this as your TV not being connected properly:

```python
#  Error: No display name and no $DISPLAY environment variable
#  Solution: Switch to non-interactive backend - like using a different TV input
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot
import matplotlib.pyplot as plt
```

### 2. Layout Problems

#### Overlapping Elements

Think of this as trying to fit too many things in a small room:

```python
#  Problem: Cramped layout - like a crowded room
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(data)
ax.set_xlabel('Very Long X Label')
ax.set_ylabel('Very Long Y Label')

#  Solution: Adjust layout - like rearranging furniture
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data)
ax.set_xlabel('Very Long X Label', labelpad=10)
ax.set_ylabel('Very Long Y Label', labelpad=10)
plt.tight_layout(pad=1.5)
```

#### Subplots Spacing

Think of this as arranging pictures on a wall:

```python
#  Problem: Overlapping subplots - like pictures too close together
fig, (ax1, ax2) = plt.subplots(2, 1)

#  Solution: Add spacing - like adding space between pictures
fig, (ax1, ax2) = plt.subplots(2, 1, 
                              height_ratios=[1, 1],
                              gridspec_kw={'hspace': 0.3})
```

### 3. Data Handling

#### Missing Data

Think of this as having gaps in your story:

```python
#  Problem: NaN values breaking plot - like missing pages in a book
data = [1, 2, np.nan, 4, 5]

#  Solution 1: Filter NaN - like skipping missing pages
clean_data = [x for x in data if not np.isnan(x)]

#  Solution 2: Interpolate - like filling in the missing parts
def handle_missing(data):
    """Handle missing values with interpolation"""
    data = np.array(data)
    mask = np.isnan(data)
    data[mask] = np.interp(
        np.flatnonzero(mask), 
        np.flatnonzero(~mask), 
        data[~mask]
    )
    return data
```

#### Scale Issues

Think of this as trying to compare very different things:

```python
#  Problem: Different scales making plot unreadable - like comparing inches and miles
x = np.linspace(0, 1, 100)
y1 = x
y2 = 1000 * x

#  Solution 1: Secondary Y-axis - like using two different rulers
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'b-', label='y1')
ax2.plot(x, y2, 'r-', label='y2')

#  Solution 2: Normalize data - like converting everything to the same unit
def normalize(data):
    """Normalize data to [0, 1] range"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))
```

### 4. Memory Management

#### Memory Leaks

Think of this as leaving too many windows open on your computer:

```python
#  Problem: Memory growing with multiple plots - like leaving windows open
for i in range(100):
    plt.figure()
    plt.plot(data)
    plt.show()

#  Solution: Proper cleanup - like closing windows when done
def plot_with_cleanup(data):
    """Plot with proper memory management"""
    try:
        plt.figure()
        plt.plot(data)
        plt.show()
    finally:
        plt.close('all')
```

#### Large Dataset Handling

Think of this as trying to show too many stars in the sky:

```python
#  Problem: Slow with large datasets - like trying to show every star
x = np.random.randn(1_000_000)
y = np.random.randn(1_000_000)

#  Solution: Data reduction strategies - like showing constellations instead
def plot_large_dataset(x, y, max_points=10_000):
    """Plot large datasets efficiently"""
    if len(x) > max_points:
        # Random sampling - like choosing representative stars
        idx = np.random.choice(
            len(x), 
            max_points, 
            replace=False
        )
        x = x[idx]
        y = y[idx]
    
    # Use scatter with transparency - like showing star density
    plt.scatter(x, y, alpha=0.1, rasterized=True)
```

### 5. Style and Formatting

#### Font Problems

Think of this as trying to use a font that's not installed:

```python
#  Problem: Font not found - like trying to use a font you don't have
plt.rcParams['font.family'] = 'NonExistentFont'

#  Solution: Robust font handling - like having backup fonts
def set_font_safely():
    """Set fonts with fallbacks"""
    try:
        plt.rcParams['font.family'] = 'Arial'
    except:
        plt.rcParams['font.family'] = 'sans-serif'
```

#### Color Issues

Think of this as trying to read yellow text on a white background:

```python
#  Problem: Poor color visibility - like hard-to-read colors
plt.plot(data1, color='yellow')  # Hard to see
plt.plot(data2, color='lime')    # Too bright

#  Solution: Professional color palette - like using readable colors
professional_colors = {
    'blue': '#2E86C1',
    'red': '#E74C3C',
    'green': '#2ECC71',
    'purple': '#8E44AD',
    'orange': '#E67E22'
}
```

### 6. Export and Saving

#### Resolution Problems

Think of this as taking a blurry photo:

```python
#  Problem: Blurry exports - like a low-resolution photo
plt.savefig('plot.png')

#  Solution: High-quality export settings - like using a better camera
def save_high_quality(fig, filename):
    """Save figure with high quality settings"""
    fig.savefig(filename,
                dpi=300,                # High DPI - like high resolution
                bbox_inches='tight',    # No cutoff - like proper framing
                pad_inches=0.1,         # Small padding - like a small border
                transparent=True)       # Transparent background - like a PNG
```

## Debugging Tools

### 1. Plot Information

Think of this as checking your car's dashboard:

```python
def print_plot_info():
    """Print current plot information"""
    fig = plt.gcf()
    ax = plt.gca()
    
    info = {
        'Figure Size': fig.get_size_inches(),
        'DPI': fig.dpi,
        'Axis Limits': {
            'X': ax.get_xlim(),
            'Y': ax.get_ylim()
        },
        'Number of Artists': len(ax.get_children()),
        'Memory Usage (MB)': (
            psutil.Process().memory_info().rss / 
            1024 / 
            1024
        )
    }
    
    return info
```

### 2. Performance Monitoring

Think of this as timing how long something takes:

```python
import time
import functools

def plot_timer(func):
    """Decorator to time plotting functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end-start:.2f} seconds')
        return result
    return wrapper
```

## Best Practices

### 1. Setup Template

Think of this as having a checklist before starting:

```python
def setup_professional_plot():
    """Setup template for professional plots"""
    plt.style.use('seaborn')
    
    # Figure size and DPI
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    
    # Grid settings
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Return figure and axes
    return plt.gcf(), plt.gca()
```

### 2. Common Mistakes to Avoid

- Not closing figures when done
- Using inappropriate chart types
- Poor color choices
- Missing labels or context

### 3. Tips for Success

- Start with a clear purpose
- Keep it simple
- Test your visualizations
- Get feedback from others

## Next Steps

1. Practice with different plot types
2. Experiment with customization
3. Learn from others' code
4. Share your visualizations
5. Join the community

Remember: The best way to learn is by doing. Start with simple plots and gradually add complexity as you become more comfortable with Matplotlib.

## Additional Resources

### Documentation Links

```
