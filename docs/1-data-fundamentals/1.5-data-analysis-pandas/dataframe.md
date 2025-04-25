# Understanding DataFrames

## What is a DataFrame?

Think of a DataFrame as an Excel spreadsheet in Python! It's a 2-dimensional table with rows and columns, where each column can hold different types of data (numbers, text, dates, etc.). DataFrames are perfect for:

- Analyzing structured data (sales records, customer info)
- Time series analysis (stock prices, weather data)
- Data cleaning and preparation
- Complex data analysis and statistics

Real-world applications:

- Sales analytics
- Financial reporting
- Customer demographics analysis
- Survey data analysis
- Medical research data

![dataframe](./assets/dataframe.png)

{% stepper %}
{% step %}

### Creating Your First DataFrame

Let's explore different ways to create a DataFrame:

```python
import pandas as pd
import numpy as np

# 1. From a dictionary
student_data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [20, 22, 21],
    "Grade": [85, 92, 78],
    "Pass": [True, True, False]
}
df = pd.DataFrame(student_data)
print("Student Database:")
print(df)
print("\nDataFrame Info:")
print(df.info())  # Shows data types and missing values

# 2. From a list of dictionaries
transactions = pd.DataFrame([
    {"date": "2023-01-01", "item": "Laptop", "price": 1200},
    {"date": "2023-01-02", "item": "Mouse", "price": 25},
    {"date": "2023-01-02", "item": "Keyboard", "price": 100}
])
print("\nTransaction Records:")
print(transactions)

# 3. From a NumPy array
array_data = np.random.rand(3, 2)  # 3 rows, 2 columns of random numbers
df_array = pd.DataFrame(array_data, 
                       columns=['Value 1', 'Value 2'],
                       index=['Row 1', 'Row 2', 'Row 3'])
print("\nFrom NumPy Array:")
print(df_array)
```

Notice how Pandas automatically adds numbered row labels (0, 1, 2) called the index!
{% endstep %}

{% step %}

### Understanding DataFrame Structure

A DataFrame has several key components:

1. **Columns**: Named fields (like "Name", "Age", "Grade")
2. **Index**: Row labels (0, 1, 2 by default)
3. **Values**: The actual data

Check these components:

```python
# Column names
print("Columns:", df.columns.tolist())

# Index
print("Index:", df.index.tolist())

# Shape (rows, columns)
print("Shape:", df.shape)
```

{% endstep %}
{% endstepper %}

## Basic DataFrame Operations

{% stepper %}
{% step %}

### Viewing Your Data

Pandas provides several ways to peek at your data:

```python
# View first few rows
print("First 2 rows:")
print(df.head(2))

# View basic information
print("\nDataFrame Info:")
print(df.info())

# View quick statistics
print("\nNumerical Statistics:")
print(df.describe())
```

{% endstep %}

{% step %}

### Accessing Columns

You can access columns in two ways:

1. Dictionary-style with square brackets
2. Attribute-style with dot notation

```python
# Get the 'Name' column
print("Using square brackets:")
print(df['Name'])

print("\nUsing dot notation:")
print(df.Name)

# Get multiple columns
print("\nMultiple columns:")
print(df[['Name', 'Grade']])
```

{% endstep %}

{% step %}

### Adding and Modifying Data

You can easily add or modify columns:

```python
# Add a new column
df['Pass'] = df['Grade'] >= 80
print("Added Pass/Fail column:")
print(df)

# Modify existing column
df['Age'] = df['Age'] + 1
print("\nAfter increasing everyone's age:")
print(df)
```

{% endstep %}
{% endstepper %}

## Working with Rows

{% stepper %}
{% step %}

### Accessing Rows

Use `loc` for label-based indexing or `iloc` for position-based indexing:

```python
# Get row by position using iloc
print("First row:")
print(df.iloc[0])

# Get row by label using loc
print("\nRow with index 1:")
print(df.loc[1])
```

{% endstep %}

{% step %}

### Filtering Rows

You can filter rows based on conditions:

```python
# Get all students who passed
passing_students = df[df['Grade'] >= 80]
print("Passing students:")
print(passing_students)

# Multiple conditions
good_grades_young = df[(df['Grade'] >= 85) & (df['Age'] < 22)]
print("\nYoung students with good grades:")
print(good_grades_young)
```

{% endstep %}
{% endstepper %}

## Handling Missing Data

{% stepper %}
{% step %}

### Understanding Missing Values

Real-world data often has missing values (shown as `NaN` in Pandas):

```python
# Create DataFrame with missing data
student_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [20, None, 21],
    'Grade': [85, 92, None]
})
print(student_data)
```

{% endstep %}

{% step %}

### Dealing with Missing Values

Pandas provides several ways to handle missing data:

```python
# Check for missing values
print("Missing values in each column:")
print(student_data.isna().sum())

# Drop rows with any missing values
print("\nDrop rows with missing values:")
print(student_data.dropna())

# Fill missing values
print("\nFill missing values with 0:")
print(student_data.fillna(0))
```

{% endstep %}
{% endstepper %}

## Best Practices and Tips

1. **Start Simple**: Begin with a small DataFrame while learning
2. **Check Your Data**:
   - Use `info()` to see data types and missing values
   - Use `describe()` for numerical summaries
   - Use `head()` to preview your data
3. **Keep Track of Changes**:
   - Make copies before big changes: `df_backup = df.copy()`
   - Chain operations thoughtfully
4. **Handle Missing Data Early**:
   - Decide on a strategy (drop or fill)
   - Document your decisions

## Common Gotchas to Avoid

1. **Chained Indexing**: Avoid `df['column'][condition]`, use `df.loc[condition, 'column']` instead
2. **Copy vs View**: Be aware when you're working with a view vs a copy of your data
3. **Missing Data**: Don't forget to check for and handle missing values
4. **Data Types**: Make sure columns have the correct data types for your analysis

Remember: DataFrames are powerful tools for data analysis. Take time to experiment with these examples and you'll be a Pandas pro in no time!
