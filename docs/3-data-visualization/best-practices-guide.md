# Data Visualization Best Practices Guide

## Core Principles

### 1. Clarity First

#### Clear Purpose
- Each visualization should answer a specific question
- Focus on one main message
- Remove unnecessary elements
- Guide viewer's attention

#### Example
```python
# Bad: Too much information
plt.plot(data1, label='Sales')
plt.plot(data2, label='Revenue')
plt.plot(data3, label='Costs')
plt.plot(data4, label='Profits')

# Good: Focus on key message
plt.plot(data1, label='Sales', color='blue')
plt.plot(data4, label='Profits', color='green')
```

### 2. Know Your Audience

#### Audience Considerations
- Technical expertise
- Domain knowledge
- Time constraints
- Decision needs

#### Example Adaptations
```python
# Technical Audience
plt.plot(data, label='Revenue Growth')
plt.title('Revenue Growth Rate (YoY)')
plt.xlabel('Time (Quarters)')
plt.ylabel('Growth Rate (%)')

# General Audience
plt.bar(categories, values)
plt.title('Sales Performance')
plt.ylabel('Sales ($M)')
```

### 3. Choose the Right Chart

#### Data Type Considerations
- Temporal: Line charts, area charts
- Categorical: Bar charts, pie charts
- Numerical: Histograms, box plots
- Relational: Scatter plots, bubble charts

#### Examples for Different Data Types
```python
# Time Series
plt.plot(dates, values)

# Categories
plt.bar(categories, values)

# Distribution
plt.hist(values, bins=30)

# Correlation
plt.scatter(x, y)
```

## Design Principles

### 1. Color Usage

#### Color Purpose
- Highlight important data
- Show categories
- Represent values
- Create hierarchy

#### Color Best Practices
```python
# Bad: Rainbow colors
plt.scatter(x, y, c=np.random.rand(100))

# Good: Sequential palette
import seaborn as sns
palette = sns.color_palette("Blues", n_colors=5)
plt.scatter(x, y, c=values, cmap=sns.color_palette("Blues", as_cmap=True))
```

### 2. Typography

#### Text Hierarchy
- Clear titles
- Readable labels
- Appropriate font sizes
- Consistent styling

#### Example
```python
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Revenue Growth', fontsize=16, pad=20)
plt.xlabel('Time Period', fontsize=12)
plt.ylabel('Revenue ($M)', fontsize=12)
plt.tick_params(labelsize=10)
```

### 3. Layout

#### Space Usage
- Maintain white space
- Align elements
- Group related items
- Use consistent spacing

#### Example
```python
# Create grid layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Sales Analysis Dashboard', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
```

## Interactive Features

### 1. Tooltips

#### Content Guidelines
- Show relevant details
- Use clear formatting
- Maintain consistency
- Avoid clutter

#### Example (Plotly)
```python
import plotly.express as px

fig = px.scatter(data, x='x', y='y',
                hover_data=['category', 'value'],
                hover_name='name',
                title='Sales by Region')
```

### 2. Filters

#### Implementation
- Clear controls
- Instant feedback
- Multiple options
- Reset capability

#### Example (Plotly)
```python
fig = px.scatter(data, x='x', y='y',
                color='category',
                animation_frame='year',
                range_x=[0, 100],
                range_y=[0, 100])
```

## Performance Optimization

### 1. Data Preparation

#### Best Practices
- Aggregate when possible
- Remove unnecessary data
- Use appropriate data types
- Cache results

#### Example
```python
# Bad: Plot all points
plt.scatter(large_dataset_x, large_dataset_y)

# Good: Aggregate or sample
bins = np.histogram2d(large_dataset_x, large_dataset_y, bins=50)
plt.pcolormesh(bins[1], bins[2], bins[0].T)
```

### 2. Rendering Optimization

#### Techniques
- Use appropriate formats
- Optimize resolution
- Minimize elements
- Consider file size

#### Example
```python
# Export for web
plt.savefig('plot.png', dpi=72, optimize=True)

# Export for print
plt.savefig('plot.pdf', dpi=300)
```

## Accessibility

### 1. Color Blindness

#### Considerations
- Use colorblind-friendly palettes
- Include patterns/shapes
- Maintain contrast
- Test with simulators

#### Example
```python
# Use colorblind-friendly palette
plt.style.use('seaborn')
colors = sns.color_palette("colorblind")
plt.plot(data1, color=colors[0], linestyle='-', marker='o')
plt.plot(data2, color=colors[1], linestyle='--', marker='s')
```

### 2. Text Readability

#### Guidelines
- Sufficient font size
- High contrast
- Clear hierarchy
- Alternative text

#### Example
```python
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Sales Growth', fontsize=16, color='black')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
```

## Documentation

### 1. Code Comments

#### Best Practices
- Explain complex logic
- Document assumptions
- Note data sources
- Include references

#### Example
```python
# Calculate moving average for smoothing
window = 7  # 7-day window for weekly patterns
smoothed_data = data.rolling(window=window, center=True).mean()

# Plot original and smoothed data
plt.plot(data, alpha=0.3, label='Original')
plt.plot(smoothed_data, label='7-day Average')
```

### 2. Visualization Documentation

#### Elements to Include
- Data sources
- Processing steps
- Calculation methods
- Update frequency

#### Example
```python
# Add source annotation
plt.figtext(0.99, 0.01, 'Source: Sales Database (Updated Daily)',
            ha='right', va='bottom', fontsize=8, style='italic')
```

## Quality Assurance

### 1. Testing

#### Check Points
- Data accuracy
- Visual accuracy
- Performance
- Accessibility

#### Example
```python
# Verify data ranges
assert data.min() >= 0, "Negative values found"
assert data.max() <= 100, "Values exceed maximum"

# Test plot generation
try:
    plt.plot(data)
    plt.savefig('test.png')
except Exception as e:
    print(f"Plot generation failed: {e}")
```

### 2. Review Process

#### Steps
- Peer review
- User testing
- Performance testing
- Documentation review

#### Checklist
1. Data accuracy verified
2. Visualization clarity checked
3. Performance tested
4. Accessibility confirmed
5. Documentation complete
