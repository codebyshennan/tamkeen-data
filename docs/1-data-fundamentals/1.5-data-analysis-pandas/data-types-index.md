# Understanding Data Types and Index in Pandas

## Data Types (dtypes)

{% stepper %}
{% step %}

### What are Data Types?

In Pandas, each column in a DataFrame (or each value in a Series) has a specific data type (dtype). Understanding data types is crucial for:

- Memory efficiency
- Better performance
- Correct calculations
- Proper data handling

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

Let's explore them in action:

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

Real-world example - Sales data:

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

{% endstep %}

{% step %}

### Checking and Converting Data Types

You can check and change data types easily:

```python
# Create a Series with numbers as strings
numbers = pd.Series(['1', '2', '3'])
print("Original data type:", numbers.dtype)

# Convert to integers
numbers = numbers.astype('int64')
print("New data type:", numbers.dtype)
print(numbers)
```

Common type conversions:

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

{% endstep %}

{% step %}

### Selecting Columns by Data Type

You can select columns based on their data type:

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

{% endstep %}
{% endstepper %}

## Understanding Index

{% stepper %}
{% step %}

### What is an Index?

Think of an index as the "row labels" in your DataFrame or Series. It's like the row numbers in Excel, but more powerful because:

- It can contain any immutable type (numbers, strings, dates)
- It helps align data when performing operations
- It makes accessing data more intuitive

```python
# Create a Series with a custom index
sales = pd.Series([100, 120, 140, 160],
                 index=['Jan', 'Feb', 'Mar', 'Apr'])
print("Monthly sales:")
print(sales)

# Access data using index
print("\nFebruary sales:", sales['Feb'])
```

{% endstep %}

{% step %}

### Working with Index

You can perform various operations with index:

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

{% endstep %}

{% step %}

### Setting and Resetting Index

You can change the index of your DataFrame:

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

{% endstep %}
{% endstepper %}

## Best Practices for Data Types and Index

{% stepper %}
{% step %}

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
{% endstep %}

{% step %}

### Index Best Practices

1. **Choose Meaningful Index**:
   - Use business-relevant identifiers
   - Ensure index values are unique when needed
   - Consider using multiple index levels for complex data

2. **Index Operations**:
   - Sort index for better performance
   - Use index for faster data lookup
   - Reset index when needed for calculations

Example:

```python
# Good index practice
sales_data = pd.DataFrame({
    'Revenue': [100, 200, 300],
    'Expenses': [50, 100, 150]
}, index=pd.date_range('2023-01-01', periods=3))

print("Well-structured DataFrame with date index:")
print(sales_data)
```

{% endstep %}
{% endstepper %}

## Common Pitfalls and Solutions

1. **Mixed Data Types**:
   - Problem: Column contains mix of numbers and strings
   - Solution: Clean data and convert to appropriate type

   ```python
   # Fix mixed types
   df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
   ```

2. **Wrong Date Format**:
   - Problem: Dates stored as strings
   - Solution: Convert to datetime

   ```python
   # Convert to datetime
   df['Date'] = pd.to_datetime(df['Date'])
   ```

3. **Duplicate Index Values**:
   - Problem: Non-unique index causing data access issues
   - Solution: Ensure index uniqueness or use multi-index

   ```python
   # Check for duplicates
   print("Duplicate index values:", df.index.duplicated().any())
   ```

Remember: Understanding data types and index is crucial for efficient data analysis. Take time to set up your data structure correctly at the beginning of your analysis!
