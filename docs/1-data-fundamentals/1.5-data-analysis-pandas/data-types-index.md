# Understanding Data Types and Index in Pandas

**After this lesson:** you can explain Understanding Data Types and Index in Pandas and try the examples in your own notebook.

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/zmdjNSmRXF4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*Corey Schafer, Python pandas tutorial (part 2): DataFrame and Series basics*

## Overview

**Prerequisites:** [Series](./series.md) and [DataFrame](./dataframe.md) basics.

**Why this lesson:** Every column has a **dtype**; wrong dtypes break math (strings that look like numbers), waste memory (`int64` where `int8` would do), or hide bugs (`object` columns that should be categorical). The **index** labels rows, misunderstanding it breaks joins and alignment. This page connects dtypes and index to everyday fixes.

## Data types (dtypes)

---

### What are Data Types?

In Pandas, each column in a DataFrame (or each value in a Series) has a specific data type (dtype). Understanding data types is important for:

- Memory efficiency
- Better performance
- Correct calculations
- Proper data handling

{% include mermaid-diagram.html src="1-data-fundamentals/1.5-data-analysis-pandas/diagrams/data-types-index-1.mmd" %}

*After loading a CSV, always call `df.dtypes` and `df.info()`. Columns that should be numeric but show `object` usually contain stray text (e.g. `"$1,200"` or `"N/A"`). Fix those before doing any math.*

Common data types include:

- **numbers**:
  - `int64` (whole numbers: age, count)
  - `float64` (decimal numbers: price, temperature)
- **text**:
  - `object` or `string` (names, categories)
  - Use `string` when possible (more efficient than `object`)
- **boolean**:
  - `bool` (True/False: is_active, has_subscription)
- **dates**:
  - `datetime64` (timestamps, calendar dates)
  - `timedelta64` (time differences)
- **categorical**:
  - For limited unique values (status, grade)
  - More memory efficient than strings

we will look at them in action:

**Mixed dtypes in one DataFrame**

- **Purpose:** Inspect `dtypes`, per-column memory, `describe()`, and `info()` on a toy table mixing numeric, string, bool, datetime, and categorical.
- **Walkthrough:** `pd.Categorical`, `pd.date_range`, `memory_usage(deep=True)` for object-heavy columns.

```python
import pandas as pd
import numpy as np

# Create a DataFrame with different data types
df = pd.DataFrame({
    'ID': [1, 2, 3],                                    # integer
    'Name': ['Alice', 'Bob', 'Charlie'],                # string
    'Height': [1.75, 1.80, 1.65],                       # float
    'IsStudent': [True, False, True],                   # boolean
    'BirthDate': pd.date_range('2000-01-01', periods=3),# datetime
    'Grade': pd.Categorical(['A', 'B', 'A'])            # categorical
})

# Check data types and memory usage
print("Data types in our DataFrame:")
print(df.dtypes)
print("\nMemory usage per column:")
print(df.memory_usage(deep=True))

# Basic statistics (only works on numeric columns)
print("\nNumerical Statistics:")
print(df.describe())

# Info about the DataFrame
print("\nDataFrame Info:")
df.info()
```

```
Data types in our DataFrame:
ID                    int64
Name                    str
Height              float64
IsStudent              bool
BirthDate    datetime64[us]
Grade              category
dtype: object

Memory usage per column:
Index        132
ID            24
Name         162
Height        24
IsStudent      3
BirthDate     24
Grade        103
dtype: int64

Numerical Statistics:
        ID    Height            BirthDate
count  3.0  3.000000                    3
mean   2.0  1.733333  2000-01-02 00:00:00
min    1.0  1.650000  2000-01-01 00:00:00
25%    1.5  1.700000  2000-01-01 12:00:00
50%    2.0  1.750000  2000-01-02 00:00:00
75%    2.5  1.775000  2000-01-02 12:00:00
max    3.0  1.800000  2000-01-03 00:00:00
std    1.0  0.076376                  NaN

DataFrame Info:
<class 'pandas.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 6 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   ID         3 non-null      int64
 1   Name       3 non-null      str
 2   Height     3 non-null      float64
 3   IsStudent  3 non-null      bool
 4   BirthDate  3 non-null      datetime64[us]
 5   Grade      3 non-null      category
dtypes: bool(1), category(1), datetime64[us](1), float64(1), int64(1), str(1)
memory usage: 250.0 bytes
```

Real-world example - Sales data:

**Categorical product + vectorized revenue**

- **Purpose:** Use `category` for low-cardinality product names and compute total sales with element-wise `Price * Quantity`.
- **Walkthrough:** `sales_df['Product'].unique()` respects category order; sum the multiplied Series.

```python
# Create sales data with appropriate types
sales_df = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=5),
    'Product': pd.Categorical(['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse']),
    'Price': [1200.50, 25.99, 1100.00, 85.50, 20.99],
    'InStock': [True, True, False, True, True],
    'Quantity': [5, 10, 3, 8, 15]
})

print("Sales Data Types:")
print(sales_df.dtypes)
print("\nUnique Products:", sales_df['Product'].unique())
print("Total Sales:", (sales_df['Price'] * sales_df['Quantity']).sum())
```

```
Sales Data Types:
Date        datetime64[us]
Product           category
Price              float64
InStock               bool
Quantity             int64
dtype: object

Unique Products: ['Laptop', 'Mouse', 'Keyboard']
Categories (3, str): ['Keyboard', 'Laptop', 'Mouse']
Total Sales: 10561.25
```

---

### Checking and Converting Data Types

You can check and change data types easily:

**`astype` for numeric Series**

- **Purpose:** Convert string digits to `int64` so you can do math without Python loops.
- **Walkthrough:** `numbers.astype('int64')` returns a new Series, assign back to replace.

```python
# Create a Series with numbers as strings
numbers = pd.Series(['1', '2', '3'])
print("Original data type:", numbers.dtype)

# Convert to integers
numbers = numbers.astype('int64')
print("New data type:", numbers.dtype)
print(numbers)
```

```
Original data type: str
New data type: int64
0    1
1    2
2    3
dtype: int64
```

Common type conversions:

**String ↔ float, int ↔ string, parse dates**

- **Purpose:** Remember three frequent casts: decimals as text → float, integers → pandas `string` dtype, ISO-like strings → `datetime64` via `to_datetime`.
- **Walkthrough:** `pd.to_datetime` is flexible with string Series.

```python
# String to number
text_numbers = pd.Series(['1.5', '2.5', '3.5'])
float_numbers = text_numbers.astype('float64')

# Number to string
numbers = pd.Series([1, 2, 3])
text = numbers.astype('string')

# String to datetime
dates = pd.Series(['2023-01-01', '2023-01-02'])
dates = pd.to_datetime(dates)
```

---

### Selecting Columns by Data Type

You can select columns based on their data type:

**`select_dtypes` for numeric vs text**

- **Purpose:** Pull only **number** columns for modeling or only **object** columns for cleaning, avoids manual column lists.
- **Walkthrough:** `include=['number']` picks both int and float; `include=['object']` matches this frame's strings.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [1.1, 2.2, 3.3],
    'C': ['x', 'y', 'z'],
    'D': [True, False, True]
})

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['number'])
print("Numeric columns:")
print(numeric_cols)

# Select string columns
text_cols = df.select_dtypes(include=['object'])
print("\nText columns:")
print(text_cols)
```

```
Numeric columns:
   A    B
0  1  1.1
1  2  2.2
2  3  3.3

Text columns:
   C
0  x
1  y
2  z
```

## Understanding Index

---

### What is an Index?

Think of an index as the "row labels" in your DataFrame or Series. It's like the row numbers in Excel, but more powerful because:

- It can contain any immutable type (numbers, strings, dates)
- It helps align data when performing operations
- It makes accessing data more intuitive

**Series with month labels**

- **Purpose:** Practice label-based lookup (`sales['Feb']`) as the mental model for row alignment later.
- **Walkthrough:** Index is the first argument to `pd.Series` after values.

```python
# Create a Series with a custom index
sales = pd.Series([100, 120, 140, 160],
                 index=['Jan', 'Feb', 'Mar', 'Apr'])
print("Monthly sales:")
print(sales)

# Access data using index
print("\nFebruary sales:", sales['Feb'])
```

```
Monthly sales:
Jan    100
Feb    120
Mar    140
Apr    160
dtype: int64

February sales: 120
```

---

### Working with Index

You can perform various operations with index:

**Custom row labels and index metadata**

- **Purpose:** Read `df.index` as row names and check uniqueness before using the index as a join key.
- **Walkthrough:** `index.is_unique` is quick validation for identifiers.

```python
# Create a DataFrame with custom index
df = pd.DataFrame({
    'Temperature': [20, 25, 22],
    'Humidity': [50, 45, 55]
}, index=['Day 1', 'Day 2', 'Day 3'])

print("DataFrame with custom index:")
print(df)

# Get index information
print("\nIndex values:", df.index.tolist())
print("Is index unique?", df.index.is_unique)
```

```
DataFrame with custom index:
       Temperature  Humidity
Day 1           20        50
Day 2           25        45
Day 3           22        55

Index values: ['Day 1', 'Day 2', 'Day 3']
Is index unique? True
```

---

### Setting and Resetting Index

You can change the index of your DataFrame:

**`set_index` and `reset_index`**

- **Purpose:** Move a column into the index for tidy lookup, then flatten back to a default `RangeIndex` when you need a column again.
- **Walkthrough:** `set_index('City')` drops that column from columns; `reset_index()` promotes it back.

```python
# Create a DataFrame
df = pd.DataFrame({
    'City': ['London', 'Paris', 'Tokyo'],
    'Population': [9M, 2.2M, 37M]
})

# Set 'City' as index
df_indexed = df.set_index('City')
print("After setting City as index:")
print(df_indexed)

# Reset index back to numbers
df_reset = df_indexed.reset_index()
print("\nAfter resetting index:")
print(df_reset)
```

## Best Practices for Data Types and Index

---

### Data Type Best Practices

1. **Choose Appropriate Types**:
   - Use `int64` for whole numbers
   - Use `float64` for decimal numbers
   - Use `string` for text (better than `object`)
   - Use `datetime64` for dates

2. **Memory Efficiency**:
   - Use smaller number types when possible (e.g., `int32` instead of `int64`)
   - Convert object columns to more specific types when possible

3. **Type Consistency**:
   - Keep data types consistent within columns
   - Convert mixed-type columns to appropriate types

---

### Index Best Practices

1. **Choose Meaningful Index**:
   - Use business-relevant identifiers
   - Ensure index values are unique when needed
   - Consider using multiple index levels for complex data

2. **Index Operations**:
   - Sort index for better performance
   - Use index for faster data lookup
   - Reset index when needed for calculations

**DateRange index for time series**

- **Purpose:** Use a **DatetimeIndex** when rows are ordered in time, helps resampling and joins later.
- **Walkthrough:** `pd.date_range('2023-01-01', periods=3)` as `index=` sets daily timestamps.

```python
# Good index practice
sales_data = pd.DataFrame({
    'Revenue': [100, 200, 300],
    'Expenses': [50, 100, 150]
}, index=pd.date_range('2023-01-01', periods=3))

print("Well-structured DataFrame with date index:")
print(sales_data)
```

```
Well-structured DataFrame with date index:
            Revenue  Expenses
2023-01-01      100        50
2023-01-02      200       100
2023-01-03      300       150
```

## Common Pitfalls and Solutions

1. **Mixed Data Types**:
   - Problem: Column contains mix of numbers and strings
   - Solution: Clean data and convert to appropriate type
   - **Purpose (snippet):** Coerce messy `Amount` strings to numeric; non-parsable values become `NaN` when using `errors='coerce'`.

   ```python
   # Fix mixed types
   df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
   ```

2. **Wrong Date Format**:
   - Problem: Dates stored as strings
   - Solution: Convert to datetime
   - **Purpose (snippet):** Parse a whole date column to `datetime64` for sorting and time-based logic.

   ```python
   # Convert to datetime
   df['Date'] = pd.to_datetime(df['Date'])
   ```

3. **Duplicate Index Values**:
   - Problem: Non-unique index causing data access issues
   - Solution: Ensure index uniqueness or use multi-index
   - **Purpose (snippet):** Quick boolean check before using the index as a key.

   ```python
   # Check for duplicates
   print("Duplicate index values:", df.index.duplicated().any())
   ```

```
Duplicate index values: False
```

Remember: Understanding data types and index is important for efficient data analysis. Take time to set up your data structure correctly at the beginning of your analysis!

## Next steps

Continue to [Reindexing and dropping](./reindexing-dropping.md), then [Function mapping](./function-mapping.md) and the remaining lessons in [Data analysis with pandas](./README.md).
