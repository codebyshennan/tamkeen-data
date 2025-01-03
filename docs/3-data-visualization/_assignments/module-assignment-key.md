# Module 3: Data Visualization Assignment Solution Guide

This document provides guidance and example solutions for the comprehensive data visualization assignment.

## Dataset Preparation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load and prepare the dataset
df = pd.read_csv('global_temperature.csv')
df['date'] = pd.to_datetime(df['date'])
df['decade'] = (df['date'].dt.year // 10) * 10
```

## Part 1: Basic Visualization Solutions

### 1. Line Plot Solution

```python
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['temperature'], 
         color='#FF5733', 
         linewidth=2, 
         label='Global Temperature')

plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Global Temperature Change Over Time', 
          fontsize=14, 
          pad=20)
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()

# Save for web
plt.savefig('temperature_line_web.png', dpi=72, bbox_inches='tight')
# Save for print
plt.savefig('temperature_line_print.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

### 2. Bar Chart Solution

```python
decade_avg = df.groupby('decade')['temperature'].mean().sort_values()

plt.figure(figsize=(12, 6))
bars = plt.bar(decade_avg.index.astype(str), 
               decade_avg.values,
               color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(decade_avg))))

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}°C',
             ha='center', va='bottom')

plt.title('Average Temperature Change by Decade')
plt.xlabel('Decade')
plt.ylabel('Temperature Change (°C)')
plt.xticks(rotation=45)

plt.savefig('temperature_bars.png', dpi=72, bbox_inches='tight')
plt.savefig('temperature_bars.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

### 3. Multi-panel Figure Solution

```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Line plot
ax1.plot(df['date'], df['temperature'])
ax1.set_title('Temperature Change Over Time')
ax1.set_xlabel('Year')
ax1.set_ylabel('Temperature (°C)')

# Histogram
ax2.hist(df['temperature'], bins=30, color='skyblue', edgecolor='black')
ax2.set_title('Temperature Distribution')
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Frequency')

# Box plot by season
df['season'] = df['date'].dt.quarter
ax3.boxplot([df[df['season']==i]['temperature'] for i in range(1,5)])
ax3.set_title('Seasonal Temperature Variations')
ax3.set_xticklabels(['Winter', 'Spring', 'Summer', 'Fall'])
ax3.set_ylabel('Temperature (°C)')

# Density plot
sns.kdeplot(data=df, x='temperature', ax=ax4)
ax4.set_title('Temperature Density')
ax4.set_xlabel('Temperature (°C)')

plt.suptitle('Global Temperature Analysis', fontsize=16, y=1.02)
plt.tight_layout()

plt.savefig('multi_panel.png', dpi=72, bbox_inches='tight')
plt.savefig('multi_panel.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

## Part 2: Advanced Visualization Solutions

### 1. Seaborn Visualizations

```python
# Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[['temperature', 'uncertainty', 'latitude', 'longitude']].corr()
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f')
plt.title('Temperature Correlation Heatmap')
plt.show()

# Pair Plot
sns.pairplot(df[['temperature', 'uncertainty', 'latitude', 'longitude']],
             diag_kind='kde',
             plot_kws={'alpha': 0.6})
plt.show()
```

### 2. Plotly Interactive Visualizations

```python
# Interactive Time Series
fig = px.line(df, 
              x='date', 
              y='temperature',
              title='Interactive Global Temperature Change')

fig.update_layout(
    xaxis_rangeslider_visible=True,
    hovermode='x unified',
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": [True, False]}],
                    label="Temperature",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [False, True]}],
                    label="Trend",
                    method="restyle"
                )
            ]),
        )
    ]
)

# Add trend line
fig.add_scatter(x=df['date'], 
                y=df['temperature'].rolling(window=365).mean(),
                name='Trend',
                line=dict(color='red'))

fig.write_html('interactive_temperature.html')
```

## Part 3: Tableau Dashboard Solution

### Dashboard Components

1. Global Map:
```
// Tableau Calculation for Temperature Color
IF [Temperature] > 0 THEN 'Above Average'
ELSEIF [Temperature] < 0 THEN 'Below Average'
ELSE 'Average'
END
```

2. Time Series:
```
// Tableau Calculation for Moving Average
WINDOW_AVG(AVG([Temperature]), -12, 0)
```

3. Custom Tooltip:
```
Temperature: <[Temperature]>°C
Change from Baseline: <[Temperature] - LOOKUP(AVG([Temperature]), -120)>°C
Uncertainty: ±<[Uncertainty]>°C
```

## Part 4: Data Storytelling Solution

### Executive Summary Example

"Global temperatures have shown a significant warming trend since 1850, with the rate of warming accelerating in recent decades. Our analysis reveals:

1. An average temperature increase of 0.8°C since 1850
2. Accelerated warming post-1980
3. Significant regional variations in warming patterns
4. Seasonal temperature changes affecting various regions differently

Key recommendations include:
- Continue monitoring temperature trends
- Focus on regions showing rapid change
- Implement additional measurement stations in underrepresented areas"

### Presentation Structure

1. Introduction Slide:
   - Title: "Global Temperature Change Analysis"
   - Key Question: "How has Earth's temperature changed since 1850?"

2. Methodology Slide:
   - Data source and quality
   - Analysis methods
   - Visualization choices

3. Findings Slides:
   - Long-term trends
   - Regional patterns
   - Seasonal variations
   - Uncertainty analysis

4. Implications Slide:
   - Scientific significance
   - Policy implications
   - Future projections

5. Recommendations Slide:
   - Monitoring suggestions
   - Research priorities
   - Data collection improvements

## Grading Guidelines

### Technical Execution (40%)
- Full marks require:
  * Clean, well-documented code
  * Efficient data processing
  * Proper use of libraries
  * Error handling

### Visual Design (30%)
- Full marks require:
  * Clear visual hierarchy
  * Appropriate color usage
  * Consistent styling
  * Professional presentation

### Storytelling (20%)
- Full marks require:
  * Clear narrative flow
  * Compelling insights
  * Effective use of data
  * Audience-appropriate content

### Documentation (10%)
- Full marks require:
  * Complete code documentation
  * Clear instructions
  * Organized submission
  * Thorough explanation

## Common Issues and Solutions

1. Data Loading:
```python
# Handle missing values
df = df.fillna(method='ffill')

# Convert data types
df['temperature'] = df['temperature'].astype(float)
```

2. Performance Optimization:
```python
# Reduce memory usage
df['date'] = pd.to_datetime(df['date'])
df = df.astype({'temperature': 'float32', 'uncertainty': 'float32'})
```

3. Interactive Features:
```python
# Add caching for better performance
@st.cache_data
def load_data():
    return pd.read_csv('global_temperature.csv')
```

## Bonus Points Solutions

### Advanced Features Example
```python
# Custom visualization combining multiple chart types
fig = go.Figure()

# Add temperature line
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['temperature'],
    name='Temperature'
))

# Add uncertainty range
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['temperature'] + df['uncertainty'],
    fill=None,
    mode='lines',
    line_color='gray',
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['temperature'] - df['uncertainty'],
    fill='tonexty',
    mode='lines',
    line_color='gray',
    name='Uncertainty Range'
))

fig.update_layout(title='Temperature with Uncertainty Range')
```

### Extended Analysis Example
```python
# Predictive analysis using simple forecasting
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare data
X = (df['date'] - df['date'].min()).dt.total_seconds().values.reshape(-1, 1)
y = df['temperature'].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Make future predictions
future_dates = pd.date_range(start=df['date'].max(), periods=120, freq='M')
future_X = (future_dates - df['date'].min()).total_seconds().values.reshape(-1, 1)
future_temps = model.predict(future_X)
```

## Additional Resources

1. Data Processing Templates:
```python
def prepare_data(df):
    """
    Prepare the dataset for visualization
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = df['date'].dt.quarter
    return df
```

2. Visualization Templates:
```python
def create_base_figure(figsize=(12, 6)):
    """
    Create a base figure with common settings
    """
    plt.figure(figsize=figsize)
    plt.style.use('seaborn')
    return plt
```

3. Documentation Template:
```python
"""
Global Temperature Analysis
--------------------------
Author: [Name]
Date: [Date]

This script analyzes global temperature data and creates visualizations
for the comprehensive data visualization assignment.

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- plotly

Usage:
python temperature_analysis.py
"""
