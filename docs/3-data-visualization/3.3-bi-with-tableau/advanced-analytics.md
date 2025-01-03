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

### Segmentation Analysis

#### Understanding Customer Segments
```yaml
Segmentation Methods:
  Demographic:
    - Age groups
    - Income levels
    - Geographic regions
    
  Behavioral:
    - Purchase frequency
    - Average order value
    - Product preferences
    
  Value-based:
    - Customer lifetime value
    - Profitability
    - Loyalty status
```

#### Implementing Segmentation
```sql
-- RFM Segmentation
{FIXED [Customer ID]:
    MAX(DATEDIFF('day', MAX([Order Date]), TODAY()))} -- Recency

{FIXED [Customer ID]:
    COUNT([Order ID])} -- Frequency

{FIXED [Customer ID]:
    AVG([Total Amount])} -- Monetary

-- Custom Segment Scoring
CASE 
    WHEN [Recency Score] >= 4 AND [Frequency Score] >= 4 
    THEN 'High Value'
    WHEN [Recency Score] <= 2 AND [Frequency Score] <= 2 
    THEN 'At Risk'
    ELSE 'Medium Value'
END
```

### Groups vs Sets

#### Understanding the Difference
```yaml
Groups:
  Characteristics:
    - Static member list
    - Based on discrete values
    - Simple to create and use
    - Cannot be combined with calculations
    
Sets:
  Characteristics:
    - Dynamic membership
    - Based on conditions
    - Can use complex logic
    - Can be combined with other sets
```

#### Working with Sets
```sql
-- Dynamic Set Based on Performance
[Sales] > {FIXED [Region]: AVG([Sales])}

-- Combining Sets (Using Set Actions)
[High Value Customers] AND [Recent Purchasers]

-- Set Size Analysis
SIZE([My Set]) / TOTAL(COUNT([Customer ID]))
```

### Combining Sets

#### Set Operations
```yaml
Operations:
  Union:
    - Combines members from both sets
    - Removes duplicates
    
  Intersection:
    - Keeps only common members
    - Useful for finding overlap
    
  Difference:
    - Removes members of one set from another
    - Identifies unique segments
```

#### Advanced Set Combinations
```sql
-- Complex Set Logic
([High Value Set] AND [Active Set]) 
OR 
([Medium Value Set] AND [Recent Purchase Set])

-- Nested Set Operations
{FIXED [Region]: 
    MIN(
        [Premium Customers] 
        AND [Loyalty Program Members]
    )}
```

### Binning Strategies

#### Types of Binning
```yaml
Binning Methods:
  Fixed-width:
    - Equal intervals
    - Good for uniform distributions
    
  Quantile:
    - Equal number of records
    - Better for skewed data
    
  Custom:
    - Business-defined ranges
    - Domain-specific breaks
```

#### Implementation Examples
```sql
-- Custom Bins with LOD
{FIXED [Customer ID]: 
    CASE 
        WHEN SUM([Sales]) < 1000 THEN 'Low'
        WHEN SUM([Sales]) < 5000 THEN 'Medium'
        ELSE 'High'
    END
}

-- Dynamic Binning
PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY [Value])
PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY [Value])
```

### Investment Scenario Analysis

#### Portfolio Analysis
```yaml
Analysis Components:
  Risk Assessment:
    - Historical volatility
    - Value at risk (VaR)
    - Beta calculation
    
  Return Analysis:
    - Total return
    - Risk-adjusted return
    - Attribution analysis
```

#### Calculations
```sql
-- Portfolio Return
SUM([Return] * [Weight]) / SUM([Weight])

-- Risk-Adjusted Metrics
([Return] - [Risk Free Rate]) / 
WINDOW_STDEV([Return], -12, 0)

-- Attribution Analysis
{FIXED [Asset Class]: 
    SUM([Return] * [Weight]) / SUM([Weight])}
```

### What-If Analysis

#### Parameter-Based Scenarios
```yaml
Scenario Types:
  Sensitivity:
    - Single variable changes
    - Impact assessment
    
  Multi-variable:
    - Combined effects
    - Interaction analysis
    
  Monte Carlo:
    - Random sampling
    - Distribution of outcomes
```

#### Implementation
```sql
-- Basic What-If
[Base Value] * (1 + [Growth Parameter])

-- Complex Scenario
CASE [Scenario Parameter]
    WHEN 'Optimistic' THEN [Value] * 1.2
    WHEN 'Pessimistic' THEN [Value] * 0.8
    ELSE [Value]
END

-- Sensitivity Testing
([Value] * POWER(1 + [Growth Rate], [Years])) *
(1 + [Adjustment Parameter])
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

## 📚 Resources and References

### Official Documentation
- [Tableau Help: Advanced Analytics](https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_advanced.htm)
- [Tableau Sets Guide](https://help.tableau.com/current/pro/desktop/en-us/sortgroup_sets_create.htm)
- [Level of Detail Expressions](https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_lod.htm)

### Community Resources
- [Tableau Community Forums](https://community.tableau.com/)
- [Tableau Public Gallery](https://public.tableau.com/gallery)
- [Tableau Blog: Analytics](https://www.tableau.com/learn/articles/advanced-analytics)

### Books and Publications
- "Visual Analytics with Tableau" by Alexander Loth
- "Practical Tableau" by Ryan Sleeper
- "Advanced Analytics with Tableau" by Jen Stirrup

Remember: Advanced analytics in Tableau is about finding the right balance between analytical power and performance. Start with clear business requirements, build incrementally, and always validate your results.
