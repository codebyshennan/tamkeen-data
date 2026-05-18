# 2.3 EDA — Hints

For each task we point at the lesson section and give a **way to start** plus a starter pattern. We never give the full solution.

> Open [the assignment](./coding.md) in another tab.

## 1. DateTime analysis
- **Where:** [Time series](../time-series.md).
- **Think:**
  - **Ensure datetime:** `flights['date'] = pd.to_datetime(flights['date'])`.
  - **Monthly delays:** `flights.groupby(flights['date'].dt.month)[['departure_delay', 'arrival_delay']].mean()`.
  - **7-day rolling passengers:** sort by date, then `flights['passengers'].rolling('7D').mean()` (needs date index) — or `.rolling(7).mean()` for a row-count window.
  - **Busiest day per airline:** group by airline + day-of-week, sum or count, then `idxmax`.

## 2. Correlation analysis
- **Where:** [Relationships](../relationships.md).
- **Think:**
  - `flights[['departure_delay', 'arrival_delay']].corr()` returns a 2×2 matrix; the off-diagonal is the answer.
  - For per-airline correlation, group then call `.corr()` on each subgroup, or use `groupby(...).apply(lambda g: g['x'].corr(g['y']))`.

## 3. Data reshaping
- **Where:** [Time series](../time-series.md), [Relationships](../relationships.md) — pivot vs melt.
- **Think:**
  - **Pivot of avg ticket price:** `pivot_table(values='ticket_price', index='airline', columns='destination', aggfunc='mean')`.
  - **Wide format daily passengers:** pivot on `index='date'`, `columns='airline'`, sum passengers.
  - **Wide → long:** `melt(id_vars='date', var_name='airline', value_name='passengers')`.

## 4. Hierarchical indexing
- **Where:** [Relationships](../relationships.md) — MultiIndex.
- **Think:**
  - Build via `groupby(['airline', 'destination']).agg({'departure_delay': 'mean', 'ticket_price': 'mean'})`.
  - Two access patterns: `.loc[('SIA', 'Tokyo')]` for a single row, `xs('Tokyo', level='destination')` to slice the inner level.

## 5. Data combination
- **Where:** [Relationships](../relationships.md) — concat / merge.
- **Think:**
  - Split with a date boundary: `first_half = flights[flights['date'].dt.month <= 6]`.
  - Merge them back: try `concat([h1, h2])` (stack), then `merge(h1, h2, how='outer'/'inner')` for comparison.
  - For monthly summaries, group then merge back on month derived from the date.

## 6. Advanced aggregation
- **Where:** [Time series](../time-series.md), [Relationships](../relationships.md).
- **Think:**
  - **Cross-tab:** `pd.crosstab(flights['airline'], flights['destination'])`.
  - **% delayed per airline:** create a boolean `delayed = flights['departure_delay'] > 15`, then `groupby('airline')['delayed'].mean() * 100`.
  - **Multi-stat agg:** `flights.groupby('airline').agg({'departure_delay': ['min', 'max', 'mean', 'count'], 'ticket_price': ['mean']})`.

## Common pitfalls
- Forgetting to **sort by date** before time-series operations (rolling, cumulative).
- Using `mean()` on a boolean column to get a proportion is fine — but multiply by 100 if you want a percentage.
- Pivoting with duplicate index/column pairs requires an `aggfunc`; otherwise pandas raises.
- `corr()` ignores non-numeric columns silently — narrow to numeric first if you want to know what's included.
