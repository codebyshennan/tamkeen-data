# Arithmetic and Data Alignment in Pandas

**After this lesson:** you can explain Arithmetic and Data Alignment in Pandas and try the examples in your own notebook.

### Video

_Corey Schafer, Python pandas tutorial (part 2): DataFrame and Series basics_

## Overview

**Prerequisites:** [Series](series.md) with labeled index; comfort with element-wise operations.

**Why this lesson:** Pandas **aligns** Series and DataFrames by **index label** before it adds, subtracts, or multiplies. That prevents silent position-wise mistakes, but it also introduces **NaN** where labels do not match. Learning alignment is what makes pandas feel "smart" instead of "broken."

## Understanding Data Alignment

***

### What is Data Alignment?

Data alignment is one of Pandas' most powerful features! It automatically matches up data by their index labels when performing operations. Think of it like:

_Pandas matches by **label**, not position. Unmatched labels become `NaN`. Always check `.isna()` after arithmetic on two Series with potentially different indexes._

* Two people comparing shopping lists
* Matching employee records from different departments
* Combining sales data from multiple stores
* Merging financial data from different sources

Key benefits:

* Automatic matching by index
* Safe handling of missing data
* Prevention of data misalignment errors
* Flexible data combination options

Real-world applications:

* Financial data reconciliation
* Sales data comparison across regions
* Stock portfolio analysis
* Company performance metrics

***

### Basic Alignment Example

we will look at alignment with practical examples:

**Series subtraction with overlapping indexes**

* **Purpose:** See how pandas aligns Series by **index label** when you subtract, introducing NaN where a label exists in only one Series.
* **Walkthrough:** Compare `store1_sales - store2_sales` (day labels) and `inventory_start - units_sold` (product labels); unmatched rows/columns become `NaN`.

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

```
Store 1 Sales:
Mon    100
Tue    120
Wed    150
Name: Store 1, dtype: int64

Store 2 Sales:
Tue    110
Wed    140
Thu    130
Name: Store 2, dtype: int64

Sales Difference (Store 1 - Store 2):
Mon     NaN
Thu     NaN
Tue    10.0
Wed    10.0
dtype: float64

Remaining Inventory:
Keyboard    45.0
Laptop      30.0
Monitor      NaN
Mouse        NaN
dtype: float64
```

Notice how:

* For **stores**, shared labels (`Tue`, `Wed`) subtract; labels only on one side (`Mon`, `Thu`) become `NaN`.
* For **inventory**, `units_sold` has no `Mouse` and `inventory_start` has no `Monitor`, so those rows show `NaN` after subtraction.

## DataFrame Arithmetic

***

### Basic DataFrame Operations

Look at how arithmetic works with DataFrames:

**Addition with aligned index and columns**

* **Purpose:** Learn that `+` on DataFrames aligns on **both** row index and column names; missing combinations propagate as `NaN`.
* **Walkthrough:** Compare `df1` and `df2` row/column overlap before reading `result`-only `B` at `row1`/`row2` lines up.

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

```
DataFrame 1:
      A  B
row1  1  4
row2  2  5
row3  3  6

DataFrame 2:
      B   C
row1  7  10
row2  8  11
row4  9  12

Result of addition:
       A     B   C
row1 NaN  11.0 NaN
row2 NaN  13.0 NaN
row3 NaN   NaN NaN
row4 NaN   NaN NaN
```

Notice how:

* Only column 'B' exists in both DataFrames
* Column 'A' only exists in df1
* Column 'C' only exists in df2
* Row 'row4' only exists in df2

***

### Filling Missing Values

You can specify a fill value for missing data during operations:

**Using `fill_value` before alignment**

* **Purpose:** Treat missing cells as a numeric default (here `0`) before the operation so you do not lose entire rows/columns to `NaN` when only one side is missing.
* **Walkthrough:** `df1.add(df2, fill_value=0)` uses the `df1`/`df2` from the previous block, re-run that cell first in a notebook.

```python
# Add with fill_value
result = df1.add(df2, fill_value=0)
print("Result with fill_value=0:")
print(result)
```

```
Result with fill_value=0:
        A     B     C
row1  1.0  11.0  10.0
row2  2.0  13.0  11.0
row3  3.0   6.0   NaN
row4  NaN   9.0  12.0
```

This is like saying "if a value is missing in one DataFrame, treat it as 0 for the calculation"

## Arithmetic Methods

***

### Available Methods

Pandas provides several arithmetic methods:

* **add()** or **+**: Addition
* **sub()** or **-**: Subtraction
* **mul()** or **\***: Multiplication
* **div()** or **/**: Division

**Side-by-side DataFrames before `mul`**

* **Purpose:** Print `prices` and `quantity` with default `0..n` index so you can see why naive multiplication needs a shared index (next block).
* **Walkthrough:** `Item` is still a column here-`set_index` happens in the following snippet.

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

```
Prices:
     Item  Price
0   Apple    0.5
1  Banana    0.3
2  Orange    0.6

Quantities:
     Item  Quantity
0   Apple        10
1  Orange         8
2   Mango         5
```

***

### Using Arithmetic Methods

Methods give you more control over operations:

**Element-wise multiply with aligned index**

* **Purpose:** After setting `Item` as the index, multiply `Price` × `Quantity` per product so missing products get `0` via `fill_value=0`.
* **Walkthrough:** `prices['Price'].mul(quantity['Quantity'], fill_value=0)` aligns on `Item` and fills missing sides with 0.

```python
# Set Item as index for both DataFrames
prices.set_index('Item', inplace=True)
quantity.set_index('Item', inplace=True)

# Calculate total cost
total = prices['Price'].mul(quantity['Quantity'], fill_value=0)
print("\nTotal cost per item:")
print(total)
```

```

Total cost per item:
Item
Apple     5.0
Banana    0.0
Mango     0.0
Orange    4.8
dtype: float64
```

The `fill_value` parameter helps handle missing data more gracefully than the default NaN values.

## Combining Overlapping Data

***

### Using combine\_first()

`combine_first()` is perfect when you have two datasets and want to:

* Use values from the first dataset where available
* Fill in missing values from the second dataset

**Merge two overlapping tables**

* **Purpose:** Use `combine_first` so `primary_data` wins where non-null, and `secondary_data` fills gaps.
* **Walkthrough:** Compare `primary_data`/`secondary_data` before `combined`-note how `row3` only exists in `primary` and `row4` only in `secondary`.

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

```
Primary data:
        A    B
row1  1.0  NaN
row2  NaN  5.0
row3  3.0  6.0

Secondary data:
       A   B
row1  10  40
row2  20  50
row4  30  60

Combined data:
         A     B
row1   1.0  40.0
row2  20.0   5.0
row3   3.0   6.0
row4  30.0  60.0
```

***

### Real-World Example

Here's a practical example using sales data:

**Prefer primary store figures, backfill from store 2**

* **Purpose:** Model reconciling two stores' sales tables where one is authoritative and the other fills missing products.
* **Walkthrough:** `combine_first` keeps Store 1's `100` and `150`, fills `Orange` from Store 2, and adds `Grape` from Store 2.

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

```
Store 1 Sales:
         Sales
Product
Apple    100.0
Banana   150.0
Orange     NaN

Store 2 Sales:
         Sales
Product
Apple       80
Orange     120
Grape       90

Combined Sales Data:
         Sales
Product
Apple    100.0
Banana   150.0
Grape     90.0
Orange   120.0
```

## Best Practices and Tips

1.  **Always Check Your Data**:

    **Quick NaN audit before arithmetic**

    * **Purpose:** Count missing values per column on `df1`/`df2` so you know whether alignment will explode into `NaN`s.
    * **Walkthrough:** Uses `isna().sum()` on the same `df1`/`df2` from the "Basic DataFrame operations" example.

    ```python
    # Before operations, check for:
    print("Missing values in df1:", df1.isna().sum())
    print("Missing values in df2:", df2.isna().sum())
    ```

```
Missing values in df1: A    0
B    0
dtype: int64
Missing values in df2: B    0
C    0
dtype: int64
```

2.  **Use Appropriate Fill Values**:

    **Additive vs multiplicative fill**

    * **Purpose:** Illustrate that `0` is a natural default for add/subtract, while `1` is often used for multiply/divide so missing factors do not zero out products incorrectly.
    * **Walkthrough:** Same `df1`/`df2` as earlier; `mul(..., fill_value=1)` avoids turning missing cells into 0 before multiplication.

    ```python
    # Choose fill_value based on your data:
    # 0 for additive operations
    df1.add(df2, fill_value=0)

    # 1 for multiplicative operations
    df1.mul(df2, fill_value=1)
    ```

```
        A     B     C
row1  1.0  28.0  10.0
row2  2.0  40.0  11.0
row3  3.0   6.0   NaN
row4  NaN   9.0  12.0
```

3.  **Handle Index Alignment**:

    **Force a common index**

    * **Purpose:** When you intend row-wise operation only after matching labels, reindex one frame to the other's index.
    * **Walkthrough:** Pick `df1.reindex(df2.index)` or the reverse depending on which index is canonical.

    ```python
    # Make sure indexes match when needed
    df1 = df1.reindex(df2.index)
    # or
    df2 = df2.reindex(df1.index)
    ```
4.  **Document Your Choices**:

    **Comment the merge rule**

    * **Purpose:** Show that `combine_first` order encodes business logic (here: recent vs historical); comments should state that rule for readers.
    * **Walkthrough:** `recent_data` / `historical_data` are placeholders, replace with your real DataFrames.

    ```python
    # Add comments explaining your decisions
    # Example: Combining sales data, preferring recent data
    combined = recent_data.combine_first(historical_data)  # Recent data takes precedence
    ```

Remember: Data alignment is automatic in Pandas, but understanding how it works helps you handle missing or mismatched data effectively!

## Common pitfalls

* **Accidental outer joins**: Adding two Series with different indexes creates union alignment; check lengths and **NaN** counts after the op.
* **fill\_value surprises**: **fill\_value** in **add**/**mul** changes what "missing" means; document what you chose.
* **Modifying views**: **SettingWithCopy** warnings often mean you chained indexing; assign with **.loc** on the intended object.

## Next steps

You have completed the core pandas lessons in Module 1. Continue to [SQL fundamentals](../../2-data-wrangling/2.1-sql/) in Module 2, or use the [full curriculum](../../curriculum.md) for the full path.
