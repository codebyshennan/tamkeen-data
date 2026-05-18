# 1.5 Coding — Hints

For each task we point at the lesson section and give a **way to start** plus a starter pattern. We never give the full solution.

> Open [the assignment](./coding.md) in another tab.

## 1. Basic data exploration
- **Where:** [DataFrame](../dataframe.md) — inspection methods.
- **Think:** `dtypes` (attribute, no parens), per-column stats with `df['quantity'].describe()` if you want them all at once, and `df.isnull().sum()` for a column-by-column missing count.
- **Starter:**
  ```python
  print(df.dtypes)
  print(df['quantity'].agg(['mean', 'min', 'max']))
  print(df.isnull().sum())
  ```

## 2. Data manipulation & arithmetic
- **Where:** [DataFrame](../dataframe.md), [Function mapping](../function-mapping.md), [Arithmetic alignment](../arithmetic-alignment.md).
- **Think:**
  - **New column:** assign with bracket notation: `df['total_amount'] = df['quantity'] * df['price']`.
  - **Daily revenue:** group by `order_date` and sum `total_amount`. Each order has its own date in this dataset, so the group ≈ one row each.
  - **+5% tax:** scalar broadcast — `df['price'] * 1.05`.
  - **Above mean quantity:** boolean indexing — `df[df['quantity'] > df['quantity'].mean()]`.

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
  - Write `categorize_amount(amount)` returning 'High' / 'Medium' / 'Low' — mind the **boundaries** (>200 high, 100–200 medium, <100 low; equal to 100 should be Medium under the spec).
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
  - `set_index('order_date')` — does not reorder rows by itself.
  - First 5 days: positional slice `dated.iloc[:5]`, or date slice with `.loc[:'2021-01-05']` once date-indexed.
  - `reset_index()` undoes it.
  - `set_index('order_id')` for the order-id-indexed copy (assign to a new variable to keep the original).

## Common pitfalls
- Forgetting `inplace=False` — `df.sort_values(...)` returns a new frame; assign back if you want to keep the change.
- Mutating the source `df` between tasks — copy first if a task expects the original shape.
- Mismatched dtypes after `read_csv` (`order_date` may come back as string) — use `pd.to_datetime` if needed.
