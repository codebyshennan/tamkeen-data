# Arithmetic and Data Alignment in Pandas

## Understanding Data Alignment

{% stepper %}
{% step %}
### What is Data Alignment?
Data alignment is one of Pandas' most powerful features! It automatically matches up data by their index labels when performing operations. Think of it like:
- 📝 Two people comparing shopping lists
- 🔄 Matching employee records from different departments
- 📊 Combining sales data from multiple stores
- 📈 Merging financial data from different sources

Key benefits:
- ✨ Automatic matching by index
- 🔍 Safe handling of missing data
- 🛡️ Prevention of data misalignment errors
- 🔄 Flexible data combination options

Real-world applications:
- 💰 Financial data reconciliation
- 📊 Sales data comparison across regions
- 📈 Stock portfolio analysis
- 🏢 Company performance metrics
{% endstep %}

{% step %}
### Basic Alignment Example
Let's explore alignment with practical examples:

```python
import pandas as pd
import numpy as np

# Example 1: Sales Data Comparison
store1_sales = pd.Series({
    'Mon': 100,
    'Tue': 120,
    'Wed': 150
}, name='Store 1')

store2_sales = pd.Series({
    'Tue': 110,
    'Wed': 140,
    'Thu': 130
}, name='Store 2')

print("Store 1 Sales:")
print(store1_sales)
print("\nStore 2 Sales:")
print(store2_sales)

# Compare sales
sales_diff = store1_sales - store2_sales
print("\nSales Difference (Store 1 - Store 2):")
print(sales_diff)

# Example 2: Product Inventory
inventory_start = pd.Series({
    'Laptop': 50,
    'Mouse': 100,
    'Keyboard': 75
})

units_sold = pd.Series({
    'Laptop': 20,
    'Keyboard': 30,
    'Monitor': 15
})

# Calculate remaining inventory
remaining = inventory_start - units_sold
print("\nRemaining Inventory:")
print(remaining)
```

Notice how:
- Labels that exist in both Series ('b' and 'c') get added together
- Labels that exist in only one Series ('a' and 'd') get NaN values
{% endstep %}
{% endstepper %}

## DataFrame Arithmetic

{% stepper %}
{% step %}
### Basic DataFrame Operations
Let's see how arithmetic works with DataFrames:

```python
# Create two DataFrames
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=['row1', 'row2', 'row3'])

df2 = pd.DataFrame({
    'B': [7, 8, 9],
    'C': [10, 11, 12]
}, index=['row1', 'row2', 'row4'])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Add them together
result = df1 + df2
print("\nResult of addition:")
print(result)
```

Notice how:
- Only column 'B' exists in both DataFrames
- Column 'A' only exists in df1
- Column 'C' only exists in df2
- Row 'row4' only exists in df2
{% endstep %}

{% step %}
### Filling Missing Values
You can specify a fill value for missing data during operations:

```python
# Add with fill_value
result = df1.add(df2, fill_value=0)
print("Result with fill_value=0:")
print(result)
```

This is like saying "if a value is missing in one DataFrame, treat it as 0 for the calculation"
{% endstep %}
{% endstepper %}

## Arithmetic Methods

{% stepper %}
{% step %}
### Available Methods
Pandas provides several arithmetic methods:
- `add()` or `+`: Addition
- `sub()` or `-`: Subtraction
- `mul()` or `*`: Multiplication
- `div()` or `/`: Division

```python
# Create sample DataFrames
prices = pd.DataFrame({
    'Item': ['Apple', 'Banana', 'Orange'],
    'Price': [0.5, 0.3, 0.6]
})
quantity = pd.DataFrame({
    'Item': ['Apple', 'Orange', 'Mango'],
    'Quantity': [10, 8, 5]
})

print("Prices:")
print(prices)
print("\nQuantities:")
print(quantity)
```
{% endstep %}

{% step %}
### Using Arithmetic Methods
Methods give you more control over operations:

```python
# Set Item as index for both DataFrames
prices.set_index('Item', inplace=True)
quantity.set_index('Item', inplace=True)

# Calculate total cost
total = prices['Price'].mul(quantity['Quantity'], fill_value=0)
print("\nTotal cost per item:")
print(total)
```

The `fill_value` parameter helps handle missing data more gracefully than the default NaN values.
{% endstep %}
{% endstepper %}

## Combining Overlapping Data

{% stepper %}
{% step %}
### Using combine_first()
`combine_first()` is perfect when you have two datasets and want to:
- Use values from the first dataset where available
- Fill in missing values from the second dataset

```python
# Create two DataFrames with some overlapping data
primary_data = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [np.nan, 5, 6]
}, index=['row1', 'row2', 'row3'])

secondary_data = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [40, 50, 60]
}, index=['row1', 'row2', 'row4'])

print("Primary data:")
print(primary_data)
print("\nSecondary data:")
print(secondary_data)

# Combine the data
combined = primary_data.combine_first(secondary_data)
print("\nCombined data:")
print(combined)
```
{% endstep %}

{% step %}
### Real-World Example
Here's a practical example using sales data:

```python
# Create two sources of sales data
store1_sales = pd.DataFrame({
    'Product': ['Apple', 'Banana', 'Orange'],
    'Sales': [100, 150, np.nan]
}).set_index('Product')

store2_sales = pd.DataFrame({
    'Product': ['Apple', 'Orange', 'Grape'],
    'Sales': [80, 120, 90]
}).set_index('Product')

print("Store 1 Sales:")
print(store1_sales)
print("\nStore 2 Sales:")
print(store2_sales)

# Combine sales data, preferring store1's data
combined_sales = store1_sales.combine_first(store2_sales)
print("\nCombined Sales Data:")
print(combined_sales)
```
{% endstep %}
{% endstepper %}

## Best Practices and Tips

1. **Always Check Your Data**:
   ```python
   # Before operations, check for:
   print("Missing values in df1:", df1.isna().sum())
   print("Missing values in df2:", df2.isna().sum())
   ```

2. **Use Appropriate Fill Values**:
   ```python
   # Choose fill_value based on your data:
   # 0 for additive operations
   df1.add(df2, fill_value=0)
   
   # 1 for multiplicative operations
   df1.mul(df2, fill_value=1)
   ```

3. **Handle Index Alignment**:
   ```python
   # Make sure indexes match when needed
   df1 = df1.reindex(df2.index)
   # or
   df2 = df2.reindex(df1.index)
   ```

4. **Document Your Choices**:
   ```python
   # Add comments explaining your decisions
   # Example: Combining sales data, preferring recent data
   combined = recent_data.combine_first(historical_data)  # Recent data takes precedence
   ```

Remember: Data alignment is automatic in Pandas, but understanding how it works helps you handle missing or mismatched data effectively!
