# Reindexing and Dropping Data in Pandas

## Understanding Reindexing

{% stepper %}
{% step %}

### What is Reindexing?

Think of reindexing like reorganizing your data to match a new set of labels. It's a powerful tool for:

- Rearranging data in a specific order
- Adding new index labels (with placeholder values)
- Removing unwanted index labels
- Aligning multiple datasets
- Restructuring data hierarchies

Real-world applications:

- Aligning financial data from different sources
- Filling in missing dates in time series
- Standardizing country codes/names
- Matching customer records across systems

Let's explore with examples:
{% endstep %}

{% step %}

### Basic Reindexing

Let's start with practical examples:

```python
import pandas as pd
import numpy as np

# Example 1: Student Grades
grades = pd.Series([85, 92, 78], 
                  index=['Alice', 'Bob', 'Charlie'])
print("Original grades:")
print(grades)

# Add a new student and reorder alphabetically
new_index = ['Alice', 'Bob', 'Charlie', 'David']
new_grades = grades.reindex(new_index)
print("\nAfter reindexing (added David):")
print(new_grades)

# Example 2: Sales Data
sales = pd.DataFrame({
    'Revenue': [1000, 1500, 1200],
    'Units': [50, 75, 60]
}, index=['Jan', 'Mar', 'Jun'])
print("\nOriginal sales data:")
print(sales)

# Fill in missing months
all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
complete_sales = sales.reindex(all_months)
print("\nComplete sales data (with missing months):")
print(complete_sales)
```

Notice how 'David' was added with a NaN (Not a Number) value since we didn't have data for them.
{% endstep %}

{% step %}

### Filling Missing Values

When reindexing, you can specify how to handle missing values:

```python
# Create a Series with missing days
temps = pd.Series([20, 22, 25], 
                 index=['Mon', 'Wed', 'Fri'])
print("Original temperatures:")
print(temps)

# Reindex to include all weekdays
all_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# Fill missing values with the previous day's temperature
temps_ffill = temps.reindex(all_days, method='ffill')
print("\nFilled forward:")
print(temps_ffill)

# Fill missing values with the next day's temperature
temps_bfill = temps.reindex(all_days, method='bfill')
print("\nFilled backward:")
print(temps_bfill)
```

{% endstep %}
{% endstepper %}

## Working with DataFrames

{% stepper %}
{% step %}

### Reindexing DataFrame Rows

You can reindex both rows and columns in a DataFrame:

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'temp': [20, 22, 25],
    'humidity': [50, 55, 45]
}, index=['Mon', 'Wed', 'Fri'])

print("Original DataFrame:")
print(df)

# Reindex with all weekdays
all_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
df_reindexed = df.reindex(all_days)
print("\nAfter reindexing rows:")
print(df_reindexed)
```

{% endstep %}

{% step %}

### Reindexing DataFrame Columns

You can also reindex columns to add or rearrange them:

```python
# Reindex columns to add 'precipitation'
new_columns = ['temp', 'humidity', 'precipitation']
df_new_cols = df.reindex(columns=new_columns)
print("After adding new column:")
print(df_new_cols)

# Reindex to rearrange columns
df_rearranged = df.reindex(columns=['humidity', 'temp'])
print("\nAfter rearranging columns:")
print(df_rearranged)
```

{% endstep %}
{% endstepper %}

## Dropping Data

{% stepper %}
{% step %}

### Understanding Drop Operations

Dropping is like removing items from your dataset. You can drop:

- Specific rows or columns
- Rows or columns that meet certain conditions
- Missing values

The dropped data is removed from the result but your original data remains unchanged unless you use `inplace=True`.
{% endstep %}

{% step %}

### Dropping Rows

Here's how to drop rows from your data:

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'grade': [85, 92, 78, 95],
    'attendance': [100, 95, None, 90]
})
print("Original DataFrame:")
print(df)

# Drop a specific row by index
df_dropped = df.drop(1)  # Drops Bob's row
print("\nAfter dropping row 1:")
print(df_dropped)

# Drop rows with missing values
df_clean = df.dropna()
print("\nAfter dropping rows with missing values:")
print(df_clean)
```

{% endstep %}

{% step %}

### Dropping Columns

You can also drop columns you don't need:

```python
# Drop a single column
df_no_attendance = df.drop('attendance', axis=1)
print("After dropping attendance column:")
print(df_no_attendance)

# Drop multiple columns
df_names_only = df.drop(['grade', 'attendance'], axis=1)
print("\nAfter dropping multiple columns:")
print(df_names_only)
```

{% endstep %}
{% endstepper %}

## Best Practices and Tips

{% stepper %}
{% step %}

### When to Use Reindex

Use reindex when you want to:

1. Align data with a specific order or structure
2. Add new index entries with placeholder values
3. Reorganize columns in a specific order
4. Match the structure of another DataFrame

Example of aligning two DataFrames:

```python
# Two DataFrames with different indexes
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'A': [4, 5, 6]}, index=['b', 'c', 'd'])

# Align df2 to match df1's index
df2_aligned = df2.reindex(df1.index)
print("Aligned DataFrame:")
print(df2_aligned)
```

{% endstep %}

{% step %}

### When to Use Drop

Use drop when you want to:

1. Remove unnecessary columns
2. Clean data by removing rows with missing values
3. Filter out specific rows or columns
4. Create a subset of your data

Example of smart dropping:

```python
# Drop rows where more than 50% of values are missing
df_clean = df.dropna(thresh=df.shape[1]//2)

# Drop duplicate rows
df_unique = df.drop_duplicates()

# Drop rows based on a condition
df_filtered = df.drop(df[df['grade'] < 60].index)
```

{% endstep %}
{% endstepper %}

## Common Pitfalls and Solutions

1. **Forgetting to Assign Results**:

   ```python
   # Wrong: original df unchanged
   df.drop('column_name', axis=1)
   
   # Right: save result or use inplace=True
   df = df.drop('column_name', axis=1)
   # or
   df.drop('column_name', axis=1, inplace=True)
   ```

2. **Wrong Axis**:

   ```python
   # Remember:
   # axis=0 or 'index' for rows
   # axis=1 or 'columns' for columns
   
   # Drop a column
   df.drop('column_name', axis=1)  # or axis='columns'
   
   # Drop a row
   df.drop(0, axis=0)  # or axis='index'
   ```

3. **Chaining Operations**:

   ```python
   # More efficient way to drop multiple items
   df_clean = (df
               .drop('unnecessary_col', axis=1)
               .dropna()
               .drop_duplicates())
   ```

Remember: Always make a copy of your data before dropping or reindexing if you want to preserve the original data structure!
