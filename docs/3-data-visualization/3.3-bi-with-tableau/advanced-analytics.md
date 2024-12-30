# Advanced Analytics in Tableau 📊

## 🎯 Overview

Advanced analytics in Tableau transforms your visualizations from simple charts into powerful analytical tools. Think of it as upgrading from a basic calculator to a sophisticated statistical engine.

```yaml
Analytics Hierarchy:
┌─────────────────────────┐
│ Descriptive            │ → What happened?
├─────────────────────────┤
│ Diagnostic            │ → Why did it happen?
├─────────────────────────┤
│ Predictive           │ → What might happen?
└─────────────────────────┘
```

## 📊 Table Calculations

### Understanding Table Calculations
```yaml
Calculation Types:
  Quick Table Calculations:
    - Running total
    - Difference
    - Percent difference
    - Percent of total
    - Rank
    - Percentile
    
  Custom Table Calculations:
    - Window functions
    - Lookup functions
    - Aggregations
    - Complex logic
```

### Basic Table Calculations

#### 1. Running Totals
```sql
-- Simple Running Total
RUNNING_SUM(SUM([Sales]))

-- Running Total with Partitioning
RUNNING_SUM(SUM([Sales]))
PARTITION BY [Category]

-- Running Total with Direction
RUNNING_SUM(SUM([Sales]), 'first()') -- Forward
RUNNING_SUM(SUM([Sales]), 'last()') -- Backward
```

#### 2. Moving Calculations
```sql
-- Moving Average (3-period)
WINDOW_AVG(SUM([Sales]), -1, 1)

-- Moving Sum (4-period)
WINDOW_SUM(SUM([Sales]), -3, 0)

-- Moving Maximum
WINDOW_MAX(SUM([Sales]), -2, 2)

-- Example: Smoothing Seasonal Data
SCRIPT_REAL("
    import numpy as np
    return np.convolve(x, np.ones(3)/3, mode='valid')
", SUM([Sales]))
```

### Advanced Table Calculations

#### 1. Nested Calculations
```sql
-- Year-over-Year Growth with Moving Average
WINDOW_AVG(
    ([Sales] - LOOKUP([Sales], -1)) / 
    LOOKUP([Sales], -1),
    -2, 2
)

-- Cumulative Growth Rate
(RUNNING_SUM(SUM([Sales])) / 
 FIRST(RUNNING_SUM(SUM([Sales])))) - 1
```

#### 2. Complex Aggregations
```sql
-- Weighted Average
SUM([Sales] * [Weight]) / SUM([Weight])

-- Moving Weighted Average
WINDOW_SUM(
    SUM([Sales] * [Weight]), -2, 0
) / 
WINDOW_SUM(SUM([Weight]), -2, 0)
```

## 🔍 Level of Detail (LOD) Expressions

### Understanding LOD Expressions
```yaml
LOD Types:
┌─────────────────────────┐
│ FIXED                  │ → Independent of view
├─────────────────────────┤
│ INCLUDE               │ → Add to view level
├─────────────────────────┤
│ EXCLUDE               │ → Remove from view
└─────────────────────────┘
```

### Basic LOD Patterns

#### 1. FIXED LOD
```sql
-- Basic Customer Metrics
{FIXED [Customer ID] : 
    SUM([Sales])} -- Total customer sales

{FIXED [Customer ID] : 
    MIN([Order Date])} -- First purchase

{FIXED [Customer ID] : 
    COUNT(DISTINCT [Order ID])} -- Order count
```

#### 2. INCLUDE LOD
```sql
-- Sales with Additional Detail
{INCLUDE [Product]: 
    SUM([Sales])} -- Product level sales

-- Nested Customer Metrics
{INCLUDE [Customer ID]: 
    AVG(
        {FIXED [Product]: SUM([Sales])}
    )}
```

#### 3. EXCLUDE LOD
```sql
-- Remove Dimension
{EXCLUDE [Region]: 
    AVG([Sales])} -- Overall average

-- Complex Exclusion
{EXCLUDE [Date]: 
    MAX(
        {FIXED [Customer]: AVG([Sales])}
    )}
```

### Advanced LOD Applications

#### 1. Cohort Analysis
```sql
-- Define Cohort
{FIXED [Customer ID]: 
    MIN(DATETRUNC('month', [First Purchase Date]))}

-- Cohort Metrics
{FIXED [Cohort], [Months Since First]: 
    AVG([Retention Rate])}
```

#### 2. Market Analysis
```sql
-- Market Share
SUM([Sales]) / 
{EXCLUDE [Product]: SUM([Sales])}

-- Relative Performance
([Sales] - {EXCLUDE [Region]: AVG([Sales])}) /
{EXCLUDE [Region]: STDEV([Sales])}
```

## 📈 Statistical Analysis

### Clustering

#### 1. K-Means Configuration
```yaml
Parameters:
  Number of Clusters:
    - Use elbow method
    - Consider business context
    - Test different values
    
  Variables:
    - Standardize numeric fields
    - Handle categorical data
    - Weight features appropriately
```

#### 2. Cluster Analysis
```sql
-- Cluster Quality
{FIXED [Cluster] : 
    VAR([Standardized Value])}

-- Silhouette Score
(
    {FIXED [Point]: MIN(
        {EXCLUDE [Current Cluster]: 
            AVG([Distance to Center])
        }
    )} -
    {FIXED [Point]: 
        AVG([Distance to Center])
    }
) /
GREATEST(
    {FIXED [Point]: MIN(
        {EXCLUDE [Current Cluster]: 
            AVG([Distance to Center])
        }
    )},
    {FIXED [Point]: 
        AVG([Distance to Center])
    }
)
```

### Forecasting

#### 1. Time Series Components
```yaml
Components:
  Trend:
    - Linear
    - Exponential
    - Polynomial
    
  Seasonality:
    - Additive
    - Multiplicative
    - Multiple cycles
    
  Noise:
    - Random variation
    - Outlier handling
    - Confidence intervals
```

#### 2. Advanced Forecasting
```sql
-- Exponential Smoothing
SCRIPT_REAL("
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(
        x, 
        seasonal_periods=12, 
        trend='add', 
        seasonal='add'
    ).fit()
    return model.forecast(12)
", SUM([Sales]))

-- Custom Forecast Model
SCRIPT_REAL("
    import pmdarima as pm
    model = pm.auto_arima(x)
    return model.predict(12)
", SUM([Sales]))
```

## 🎯 Best Practices

### 1. Performance Optimization
```yaml
Calculation Strategy:
  - Use appropriate LOD scope
  - Minimize calculation complexity
  - Pre-aggregate when possible
  - Cache intermediate results
  
Data Strategy:
  - Extract and filter early
  - Index key fields
  - Partition large datasets
  - Monitor query performance
```

### 2. Development Workflow
```yaml
Process:
  1. Prototype:
     - Start simple
     - Test core logic
     - Validate results
     
  2. Optimize:
     - Improve performance
     - Refine calculations
     - Add documentation
     
  3. Deploy:
     - Test thoroughly
     - Monitor performance
     - Gather feedback
```

### 3. Documentation
```yaml
Required Elements:
  Calculation Logic:
    - Purpose
    - Dependencies
    - Assumptions
    - Limitations
    
  Performance Notes:
    - Expected volume
    - Optimization tips
    - Known issues
    
  Usage Guide:
    - Example usage
    - Common pitfalls
    - Best practices
```

Remember: Advanced analytics in Tableau is about finding the right balance between analytical power and performance. Start with clear business requirements, build incrementally, and always validate your results.
