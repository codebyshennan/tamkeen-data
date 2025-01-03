# Data Visualization Principles 📊

## 🧠 The Science of Visual Perception

Understanding how humans process visual information is crucial for creating effective visualizations. Our brains process visual information in specific ways that we can leverage for better communication.

### Pre-attentive Processing
```
Processing Time:
┌─────────────────────┐
│ Pre-attentive       │ → < 250ms
├─────────────────────┤
│ Conscious           │ → > 250ms
└─────────────────────┘
```

### Against Attentive Processing
Pre-attentive Processing:
- Fast, automatic, parallel
- Detects visual properties
- Identifies patterns


Attentive Processing:
- Slow, effortful, serial
- Focuses on details
- Interprets complex information


#### 1. Form Attributes
```yaml
Visual Elements:
  Length:
    - Bar charts
    - Line length
    - Progress bars
    
  Size:
    - Bubble plots
    - Tree maps
    - Icon size
    
  Shape:
    - Markers
    - Icons
    - Symbols
    
  Enclosure:
    - Boundaries
    - Containers
    - Groups
```

#### 2. Color Attributes
```yaml
Properties:
  Hue:
    - Categories
    - Distinct groups
    - Qualitative data
    
  Intensity:
    - Sequential data
    - Heat maps
    - Density plots
    
  Position:
    - Coordinates
    - Placement
    - Alignment
```

### Gestalt Principles
```
Visual Organization:
     ┌─────────────┐
     │ Proximity   │ → Close = Related
     ├─────────────┤
     │ Similarity  │ → Similar = Group
     ├─────────────┤
     │ Continuity  │ → Flow = Connection
     ├─────────────┤
     │ Closure     │ → Complete Shapes
     ├─────────────┤
     │ Figure/Ground│ → Focus vs Context
     └─────────────┘
```

Key Principles:
1. Simple
- maximize impact, minimize noise
- if it doesn't add value or serve a purpose, get rid of it

2. Narrative
- don't just show it; tell a story with your data
- communicate key insights clearly, quickly and powerfully

3. Balance betrween design & function
- selecting the right chart type, color scheme, and layout
- beautiful is good, functional is better, both is best

> "The goal of a visualization is insight, not pictures." - Ben Shneiderman

### The 10-Second Rule
- If your audience can't understand your visualization in 10 seconds, it's not effective
- Keep it simple, clear, and focused on the key message

## 📊 Chart Selection Framework

### 1. Comparison
```python
# Example: Bar Chart for Category Comparison
def create_comparison_chart(data):
    plt.figure(figsize=(10, 6))
    
    # Create bars with different colors
    bars = plt.bar(data['category'], 
                   data['value'],
                   color=sns.color_palette("husl", 
                                         len(data)))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.title('Category Comparison')
    plt.xlabel('Categories')
    plt.ylabel('Values')
```

### 2. Distribution
```yaml
Chart Types:
  Single Variable:
    Primary: Histogram
    Alternatives:
      - Density plot
      - Box plot
      - Violin plot
    
  Multiple Variables:
    Primary: Box plots
    Alternatives:
      - Violin plots
      - Ridge plots
      - Bean plots
```

### 3. Relationship
```python
# Example: Enhanced Scatter Plot
def create_relationship_plot(data):
    plt.figure(figsize=(10, 6))
    
    # Create scatter with size and color encoding
    plt.scatter(data['x'], data['y'],
               s=data['size']*100,  # Size encoding
               c=data['color'],     # Color encoding
               alpha=0.6,           # Transparency
               cmap='viridis')      # Color map
    
    # Add trend line
    z = np.polyfit(data['x'], data['y'], 1)
    p = np.poly1d(z)
    plt.plot(data['x'], p(data['x']), 
             "r--", alpha=0.8)
    
    plt.title('Relationship Analysis')
    plt.xlabel('Variable X')
    plt.ylabel('Variable Y')
```

## 🎨 Color Theory

### 1. Color Schemes
```yaml
Sequential:
  Use Case: Ordered data
  Examples:
    - Light to dark blue
    - Yellow to red
    - Single hue progression
    
Diverging:
  Use Case: Data with midpoint
  Examples:
    - Red → White → Blue
    - Purple → White → Green
    - Diverging from neutral
    
Qualitative:
  Use Case: Categories
  Examples:
    - Distinct hues
    - Equal brightness
    - Maximum contrast
```

### 2. Accessibility Guidelines
```python
# Example: Colorblind-friendly palette
def get_colorblind_palette():
    """Return a colorblind-friendly palette"""
    return {
        'blue': '#0077BB',    # Blue
        'orange': '#EE7733',  # Orange
        'cyan': '#00CCBB',    # Cyan
        'magenta': '#EE3377', # Magenta
        'red': '#CC3311',     # Red
        'teal': '#009988',    # Teal
        'grey': '#BBBBBB'     # Grey
    }
```

## 📐 Layout and Composition

### 1. Visual Hierarchy
```
Information Flow:
     ┌─────────────┐
     │ Primary     │ → Key message/visual
     ├─────────────┤
     │ Secondary   │ → Supporting info
     ├─────────────┤
     │ Tertiary    │ → Details/context
     └─────────────┘
```

### 2. Grid Systems
```yaml
Dashboard Layout:
  12-Column Grid:
    Full Width: 12 columns
    Half Width: 6 columns
    Third Width: 4 columns
    Quarter Width: 3 columns
    
  Spacing:
    Margins: 24px
    Gutters: 16px
    Padding: 16px
```

## ⚠️ Common Pitfalls

### 1. Chart Junk
```yaml
Avoid:
  Decorative Elements:
    - 3D effects
    - Gradients
    - Shadows
    
  Unnecessary Components:
    - Redundant labels
    - Extra gridlines
    - Decorative icons
```

### 2. Data Distortion
```python
# Example: Proper Axis Handling
def create_proper_bar_chart(data):
    plt.figure(figsize=(10, 6))
    
    # Always start y-axis at zero for bars
    plt.bar(data['category'], data['value'])
    
    # Set proper y-axis limits
    plt.ylim(bottom=0)  # Force y-axis to start at 0
    
    # Add proper labels
    plt.title('Sales by Category')
    plt.xlabel('Category')
    plt.ylabel('Sales ($)')
```

## ✅ Best Practices Checklist

### 1. Data Integrity
```yaml
Verification:
  - Data accuracy
  - Proper scales
  - Clear sources
  - Context provided
```

### 2. Visual Clarity
```yaml
Elements:
  - Clear purpose
  - Appropriate chart
  - Clean design
  - Logical flow
```

### 3. Technical Excellence
```python
# Example: Professional Plot Setup
def create_professional_visualization():
    # Set style
    plt.style.use('seaborn')
    
    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set DPI for clarity
    plt.rcParams['figure.dpi'] = 300
    
    # Add proper spacing
    plt.tight_layout()
    
    return fig, ax
```

Remember: The goal of data visualization is to enhance understanding, not just to make things look pretty. Every visual element should serve a purpose in communicating your data story effectively.
