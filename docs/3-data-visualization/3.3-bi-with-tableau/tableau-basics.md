# Tableau Basics

## 🎯 Getting Started with Tableau

What is Tableau?

* Tableau is a powerful data visualization tool that allows users to create interactive and shareable dashboards.
* It enables users to connect to various data sources, transform data, and create visualizations without writing code.

### Tableau Product Line

| Product                 | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| Tableau Prep            | Visually combine, shape, clean data, automate data prep flow    |
| Tableau Desktop         | Visual analysis, data exploration, data-driven decision-making  |
| Tableau Public          | Free version of Tableau Desktop with some limitations           |
| Tableau Server & Online | Share the tableau report across organisation on server or cloud |
| Tableau Mobile          | Access KPI on Mobile App for iOS and Android                    |

### Understanding the Interface

<figure><img src="../../.gitbook/assets/Tableau Public 2025-01-04 00.54.33.png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/CleanShot 2025-01-04 at 00.58.00.png" alt=""><figcaption></figcaption></figure>

```yaml
Interface Components:
┌─────────────────────────────────────────────────────────────┐
│                      Toolbar & Menu                         │
├────────┬────────────────────────────┬───────────────────────┤
│        │                            │                       │
│  Data  │     Visualization          │       Show Me         │
│  Panel │         Canvas             │        Panel          │
│        │                            │                       │
├────────┤                            │                       │
│        │                            ├───────────────────────┤
│ Marks  │                            │                       │
│ Card   │                            │       Legends &       │
│        │                            │        Filters.       │
│        │                            │                       │
├────────┴────────────────────────────┴───────────────────────┤
│                  Worksheet Tabs                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components Explained

#### 1. Data Panel

```yaml
Structure:
  Dimensions (Blue):
    - Categorical data
    - Dates
    - Geographic fields
    
  Measures (Green):
    - Numeric values
    - Aggregated data
    - Calculated fields
    
  Other Elements:
    - Parameters
    - Sets
    - Calculated fields
```

#### 2. Marks Card

```yaml
Visual Properties:
  - Color: Encode categories/values
  - Size: Represent magnitude
  - Label: Show data points
  - Detail: Add information
  - Tooltip: Interactive info
  - Path: Line/polygon order
```

## 📊 Data Connection Fundamentals

### Connecting to Data Sources

#### 1. File Connections

```yaml
Local Files:
  Excel:
    - Multiple sheets
    - Named ranges
    - Custom SQL
    
  CSV/Text:
    - Delimiter options
    - Character encoding
    - First row headers
    
  JSON/XML:
    - Schema detection
    - Path specification
    - Data flattening
```

#### 2. Database Connections

```sql
-- Example: SQL Connection Settings
Server: database.example.com
Port: 5432
Database: sales_data
Authentication:
  - Username/Password
  - Integrated
  - SSL certificates

-- Example: Custom SQL Query
SELECT 
    date_trunc('month', order_date) as month,
    product_category,
    SUM(sales) as total_sales,
    COUNT(DISTINCT customer_id) as unique_customers
FROM sales_table
WHERE order_date >= '2023-01-01'
GROUP BY 1, 2
```

### Data Preparation

#### 1. Data Source Filters

```yaml
Filter Types:
  Extract Filters:
    - Reduce data size
    - Improve performance
    - Focus analysis
    
  Connection Filters:
    - Limit initial load
    - Improve query speed
    - Reduce server load
```

#### 2. Data Cleaning

```yaml
Common Tasks:
  Field Renaming:
    - Clear naming
    - Remove spaces
    - Add prefixes
    
  Data Type Setting:
    - String → Date
    - Number → Currency
    - String → Geographic
    
  Null Handling:
    - Filter nulls
    - Replace values
    - Special indicators
```

## 🎨 Building Basic Visualizations

### Essential Chart Types

#### 1. Bar Charts

```yaml
Types:
  Vertical Bars:
    Use: Category comparison
    Example: Sales by Product
    
  Horizontal Bars:
    Use: Long category names
    Example: Customer Rankings
    
  Stacked Bars:
    Use: Part-to-whole
    Example: Sales by Region & Category
```

#### 2. Line Charts

```yaml
Variations:
  Single Line:
    Steps:
      1. Date to Columns
      2. Measure to Rows
      3. Set date level
      4. Add reference lines
      
  Multiple Lines:
    Steps:
      1. Base line chart
      2. Color by dimension
      3. Add legends
      4. Customize tooltips
```

#### 3. Scatter Plots

```yaml
Construction:
  Basic Plot:
    1. Measure X → Columns
    2. Measure Y → Rows
    3. Change mark to Circle
    
  Enhancements:
    - Size by measure
    - Color by dimension
    - Add trend lines
    - Show clusters
```

### Advanced Chart Features

#### 1. Dual Axis Charts

```sql
-- Creating Dual Axis
1. Create first measure
2. Drag second measure to right
3. Right-click → Dual Axis
4. Synchronize if needed

-- Example: Sales and Profit
Axis 1: SUM([Sales])
Axis 2: SUM([Profit])
Mark Types: Bar and Line
```

#### 2. Calculated Fields

```sql
-- Basic Calculations
Profit Ratio = [Profit] / [Sales]

-- Date Calculations
Days Since First Purchase = 
DATEDIFF('day', {FIXED [Customer ID] : 
  MIN([Order Date])}, [Order Date])

-- Window Calculations
Running Total = 
RUNNING_SUM(SUM([Sales]))
```

## 🔧 Formatting and Style

### Visual Formatting

#### 1. Color Usage

```yaml
Color Types:
  Sequential:
    - Single hue progression
    - Intensity variation
    - Quantitative data
    
  Diverging:
    - Two-color scale
    - Neutral midpoint
    - Above/below threshold
    
  Categorical:
    - Distinct hues
    - Clear separation
    - Qualitative data
```

#### 2. Typography

```yaml
Text Elements:
  Titles:
    - Clear hierarchy
    - Consistent size
    - Informative text
    
  Labels:
    - Readable size
    - Strategic placement
    - Minimal overlap
    
  Tooltips:
    - Relevant info
    - Clear formatting
    - Custom content
```

### Layout Best Practices

#### 1. Grid System

```yaml
Alignment:
  - Use snap to grid
  - Maintain margins
  - Consistent spacing
  - Aligned elements

Distribution:
  - Even spacing
  - Balanced layout
  - Visual hierarchy
  - Clear grouping
```

#### 2. White Space

```yaml
Usage:
  - Separate sections
  - Frame content
  - Improve readability
  - Guide attention
```

## 🎯 Practical Tips

### 1. Performance Optimization

```yaml
Data Strategy:
  - Use extracts
  - Filter early
  - Aggregate data
  - Index key fields

Calculation Tips:
  - Minimize complexity
  - Use context filters
  - Pre-aggregate
  - Cache results
```

### 2. Common Pitfalls

```yaml
Avoid:
  - Too many filters
  - Complex calculations
  - Large extracts
  - Visual clutter

Instead:
  - Strategic filtering
  - Simple measures
  - Optimized data
  - Clean design
```

### 3. Development Workflow

```yaml
Process:
  1. Plan visualization
  2. Prepare data
  3. Build basic view
  4. Add interactivity
  5. Format and polish
  6. Test performance
```

## 📚 Next Steps

### 1. Advanced Features

```yaml
Learn:
  - Table calculations
  - Level of Detail (LOD)
  - Parameters
  - Actions
```

### 2. Dashboard Design

```yaml
Explore:
  - Layout techniques
  - Interactivity
  - Mobile design
  - Performance
```

Remember: The key to mastering Tableau is practice and experimentation. Start with simple visualizations and gradually increase complexity as you become more comfortable with the tool.
