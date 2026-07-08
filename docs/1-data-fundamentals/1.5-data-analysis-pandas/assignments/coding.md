# Assignment: Data Analysis with Pandas

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

First, create the following mock e-commerce dataset using this code:

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(2024)

# Create a mock dataset
data = {
    'order_id': range(1, 11),
    'customer_id': np.random.randint(1000, 1020, size=10),
    'product_id': np.random.randint(100, 110, size=10),
    'quantity': np.random.randint(1, 5, size=10),
    'price': np.random.uniform(10.0, 100.0, size=10),
    'order_date': pd.date_range(start='2021-01-01', periods=10, freq='D')
}

df = pd.DataFrame(data)
print(df)
```

```
   order_id  customer_id  product_id  quantity      price order_date
0         1         1008         105         3  29.325423 2021-01-01
1         2         1000         101         4  32.731698 2021-01-02
2         3         1000         107         2  67.374953 2021-01-03
3         4         1004         107         3  53.093335 2021-01-04
4         5         1009         106         4  74.511018 2021-01-05
5         6         1001         102         3  70.628991 2021-01-06
6         7         1003         107         4  48.263789 2021-01-07
7         8         1010         109         2  71.342279 2021-01-08
8         9         1002         106         2  66.276559 2021-01-09
9        10         1000         104         2  70.429624 2021-01-10
```

## Tasks

### 1. Basic Data Exploration

1. Display the data types of each column in the DataFrame
2. Calculate basic statistics (mean, min, max) for the 'quantity' column
3. Check if there are any missing values in the DataFrame

### 2. Data Manipulation & Arithmetic

1. Create a new column 'total_amount' by multiplying 'quantity' and 'price'
2. Calculate the daily revenue (sum of total_amount) and store it in a new Series
3. Add 5% tax to all prices and store in a new column 'price_with_tax'
4. Find orders where the quantity is above the mean quantity

### 3. Sorting & Ranking

1. Sort the DataFrame by total_amount in descending order
2. Rank the orders based on their price (highest price = rank 1)
3. Find the top 3 orders by total_amount
4. Sort the orders by date and quantity

### 4. Function Application

1. Create a function that categorizes total_amount into 'High' (>$200), 'Medium' ($100-$200), and 'Low' (<$100)
2. Apply this function to create a new column 'order_category'
3. Format the price and total_amount columns to display as currency with 2 decimal places
4. Calculate the cumulative sum of total_amount ordered by date

### 5. Index Operations

1. Set the order_date as the index of the DataFrame
2. Select all orders from the first 5 days
3. Reset the index back to default numeric indices
4. Create a new copy of the DataFrame with order_id as the index

## Instructions

1. Complete each task in order
2. Document your code with comments
3. Use appropriate pandas methods and functions
4. Format your output for readability

## Deliverable

Submit a Jupyter notebook containing:

- The setup code
- All task solutions with explanations
- A brief summary of insights found in the data

Your notebook should be well-organized with markdown cells or comments in code explaining your approach for each task.


## Hints

<details>
<summary>Show hints</summary>

## 1. Basic data exploration
- **Where:** [DataFrame](../dataframe.md), inspection methods.
- **Think:** `dtypes` (attribute, no parens), per-column stats with `df['quantity'].describe()` if you want them all at once, and `df.isnull().sum()` for a column-by-column missing count.
- **Starter:**
  ```python
  print(df.dtypes)
  print(df['quantity'].agg(['mean', 'min', 'max']))
  print(df.isnull().sum())
  ```

```
order_id                int64
customer_id             int64
product_id              int64
quantity                int64
price                 float64
order_date     datetime64[us]
dtype: object
mean    2.9
min     2.0
max     4.0
Name: quantity, dtype: float64
order_id       0
customer_id    0
product_id     0
quantity       0
price          0
order_date     0
dtype: int64
```

## 2. Data manipulation & arithmetic
- **Where:** [DataFrame](../dataframe.md), [Function mapping](../function-mapping.md), [Arithmetic alignment](../arithmetic-alignment.md).
- **Think:**
  - **New column:** assign with bracket notation: `df['total_amount'] = df['quantity'] * df['price']`.
  - **Daily revenue:** group by `order_date` and sum `total_amount`. Each order has its own date in this dataset, so the group ≈ one row each.
  - **+5% tax:** scalar broadcast, `df['price'] * 1.05`.
  - **Above mean quantity:** boolean indexing, `df[df['quantity'] > df['quantity'].mean()]`.

## 3. Sorting & ranking
- **Where:** [Sorting & ranking](../sorting-ranking.md).
- **Think:**
  - `sort_values('col', ascending=False)` for sort.
  - `rank(ascending=False)` for "highest = rank 1".
  - `nlargest(3, 'total_amount')` is the idiomatic top-3 (no need to sort and slice).
  - Sort by multiple keys: `sort_values(['order_date', 'quantity'])`.

## 4. Function application
- **Where:** [Function mapping](../function-mapping.md).
- **Think:**
  - Write `categorize_amount(amount)` returning 'High' / 'Medium' / 'Low', mind the **boundaries** (>200 high, 100-200 medium, <100 low; equal to 100 should be Medium under the spec).
  - Apply with `df['order_category'] = df['total_amount'].apply(categorize_amount)`.
  - Currency formatting via `df['price'].map('${:,.2f}'.format)`.
  - Cumulative sum: sort by date first, then `.cumsum()`.
- **Starter:**
  ```python
  def categorize_amount(amount):
      if amount > 200: return 'High'
      if amount >= 100: return 'Medium'
      return 'Low'
  df['order_category'] = df['total_amount'].apply(categorize_amount)
  ```

## 5. Index operations
- **Where:** [Data types & index](../data-types-index.md), [Reindexing & dropping](../reindexing-dropping.md).
- **Think:**
  - `set_index('order_date')`, does not reorder rows by itself.
  - First 5 days: positional slice `dated.iloc[:5]`, or date slice with `.loc[:'2021-01-05']` once date-indexed.
  - `reset_index()` undoes it.
  - `set_index('order_id')` for the order-id-indexed copy (assign to a new variable to keep the original).

## Common pitfalls
- Forgetting `inplace=False`, `df.sort_values(...)` returns a new frame; assign back if you want to keep the change.
- Mutating the source `df` between tasks, copy first if a task expects the original shape.
- Mismatched dtypes after `read_csv` (`order_date` may come back as string), use `pd.to_datetime` if needed.

</details>
