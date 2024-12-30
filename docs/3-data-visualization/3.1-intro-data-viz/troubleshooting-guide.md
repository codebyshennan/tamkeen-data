# Matplotlib Troubleshooting Guide 🔧

## 🚨 Common Issues and Solutions

### 1. Display Problems

#### Plot Not Showing
```python
# ❌ Problem
plt.plot([1, 2, 3], [1, 2, 3])
# Nothing appears

# ✅ Solution 1: Add plt.show()
plt.plot([1, 2, 3], [1, 2, 3])
plt.show()

# ✅ Solution 2: For Jupyter
%matplotlib inline
plt.plot([1, 2, 3], [1, 2, 3])
```

#### Backend Issues
```python
# ❌ Error: No display name and no $DISPLAY environment variable
# ✅ Solution: Switch to non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot
import matplotlib.pyplot as plt
```

### 2. Layout Problems

#### Overlapping Elements
```python
# ❌ Problem: Cramped layout
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(data)
ax.set_xlabel('Very Long X Label')
ax.set_ylabel('Very Long Y Label')

# ✅ Solution: Adjust layout
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data)
ax.set_xlabel('Very Long X Label', labelpad=10)
ax.set_ylabel('Very Long Y Label', labelpad=10)
plt.tight_layout(pad=1.5)
```

#### Subplots Spacing
```python
# ❌ Problem: Overlapping subplots
fig, (ax1, ax2) = plt.subplots(2, 1)

# ✅ Solution: Add spacing
fig, (ax1, ax2) = plt.subplots(2, 1, 
                              height_ratios=[1, 1],
                              gridspec_kw={'hspace': 0.3})
```

### 3. Data Handling

#### Missing Data
```python
# ❌ Problem: NaN values breaking plot
data = [1, 2, np.nan, 4, 5]

# ✅ Solution 1: Filter NaN
clean_data = [x for x in data if not np.isnan(x)]

# ✅ Solution 2: Interpolate
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
```python
# ❌ Problem: Different scales making plot unreadable
x = np.linspace(0, 1, 100)
y1 = x
y2 = 1000 * x

# ✅ Solution 1: Secondary Y-axis
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'b-', label='y1')
ax2.plot(x, y2, 'r-', label='y2')

# ✅ Solution 2: Normalize data
def normalize(data):
    """Normalize data to [0, 1] range"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))
```

### 4. Memory Management

#### Memory Leaks
```python
# ❌ Problem: Memory growing with multiple plots
for i in range(100):
    plt.figure()
    plt.plot(data)
    plt.show()

# ✅ Solution: Proper cleanup
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
```python
# ❌ Problem: Slow with large datasets
x = np.random.randn(1_000_000)
y = np.random.randn(1_000_000)

# ✅ Solution: Data reduction strategies
def plot_large_dataset(x, y, max_points=10_000):
    """Plot large datasets efficiently"""
    if len(x) > max_points:
        # Random sampling
        idx = np.random.choice(
            len(x), 
            max_points, 
            replace=False
        )
        x = x[idx]
        y = y[idx]
    
    # Use scatter with transparency
    plt.scatter(x, y, alpha=0.1, rasterized=True)
```

### 5. Style and Formatting

#### Font Problems
```python
# ❌ Problem: Font not found
plt.rcParams['font.family'] = 'NonExistentFont'

# ✅ Solution: Robust font handling
def set_font_safely():
    """Set fonts with fallbacks"""
    try:
        plt.rcParams['font.family'] = 'Arial'
    except:
        plt.rcParams['font.family'] = 'sans-serif'
```

#### Color Issues
```python
# ❌ Problem: Poor color visibility
plt.plot(data1, color='yellow')  # Hard to see
plt.plot(data2, color='lime')    # Too bright

# ✅ Solution: Professional color palette
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
```python
# ❌ Problem: Blurry exports
plt.savefig('plot.png')

# ✅ Solution: High-quality export settings
def save_high_quality(fig, filename):
    """Save figure with high quality settings"""
    fig.savefig(filename,
                dpi=300,                # High DPI
                bbox_inches='tight',    # No cutoff
                pad_inches=0.1,         # Small padding
                transparent=True)       # Transparent background
```

## 🛠️ Debugging Tools

### 1. Plot Information
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

## ✅ Best Practices

### 1. Setup Template
```python
def setup_professional_plot():
    """Setup template for professional plots"""
    plt.style.use('seaborn')
    
    # Figure size and DPI
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Font settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # Grid settings
    plt.grid(True, linestyle=':', alpha=0.7)
```

### 2. Error Handling
```python
def safe_plot(func):
    """Decorator for safe plotting"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            plt.close('all')  # Cleanup
            raise
    return wrapper
```

## 📚 Additional Resources

### Documentation Links
```yaml
Official Docs:
  - matplotlib.org/stable/users/index.html
  - matplotlib.org/stable/api/index.html
  - matplotlib.org/stable/gallery/index.html

Community:
  - stackoverflow.com/questions/tagged/matplotlib
  - github.com/matplotlib/matplotlib/issues
```

### Debugging Tools
```yaml
Tools:
  - Memory Profiler: memory_profiler
  - Line Profiler: line_profiler
  - Debugger: pdb/ipdb
  
Visualization:
  - Debug Viewer: mplcursors
  - Interactive Tools: mpldatacursor
```

Remember: When troubleshooting, start with the simplest possible example that reproduces your issue. This makes it easier to identify and fix the problem.
