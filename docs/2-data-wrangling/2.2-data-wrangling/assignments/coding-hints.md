# 2.2 Data Wrangling — Hints

For each task we point at the lesson section and give a **way to start** plus a starter pattern. We never give the full solution.

> Open [the assignment](./coding.md) in another tab.

## 1. Data cleaning
- **Where:** [Missing values](../missing-values.md), [Data quality](../data-quality.md).
- **Think:**
  - **Dedupe case-insensitive:** can't pass a flag to `drop_duplicates`. Add a helper column `name_lc = data['name'].str.lower()`, drop dupes on it, then drop the helper.
  - **Sentinel −999 → NaN:** `data['age'] = data['age'].replace(-999, np.nan)`. The sentinel value is masquerading as a number; the lesson covers this exact pattern.
  - **Drop rows with missing names:** `dropna(subset=['name'])`.
- **Starter:**
  ```python
  data = data.assign(_name_lc=data['name'].str.lower()) \
             .drop_duplicates(subset='_name_lc') \
             .drop(columns='_name_lc')
  ```

## 2. String manipulation
- **Where:** [Transformations](../transformations.md) — string operations.
- **Think:** Vectorized string methods live on `.str`.
  - **Title case:** `data['name'].str.title()`.
  - **Email domain:** `data['email'].str.split('@').str[-1]`.
  - **`is_gmail`:** compare the extracted domain to `'gmail.com'`, **or** use `.str.endswith('@gmail.com')` directly. Watch for `NaN` emails — the result will also be NaN unless you handle it.

## 3. Categorical data
- **Where:** [Transformations](../transformations.md) — categorical encoding.
- **Think:**
  - Convert with `astype('category')`.
  - Dummies with `pd.get_dummies(data['category'])` — returns a new DataFrame of 0/1 columns.
  - Frequency: `data['category'].value_counts()`.

## 4. Handling outliers
- **Where:** [Outliers](../outliers.md).
- **Think:** "More than 2 SDs from the mean" = absolute Z-score above 2.
  - `z = (s - s.mean()) / s.std(); outliers = data[z.abs() > 2]`.
  - Replace by assignment: `data.loc[z.abs() > 2, 'score'] = data['score'].mean()`.
  - Watch for chicken-and-egg: if you replace **then** describe, do the calculation on the cleaned column.

## 5. Data transformation
- **Where:** [Transformations](../transformations.md) — mapping.
- **Think:**
  - Build the dict `{'A': 1, 'B': 2, 'C': 3}` and call `.map(...)` on the category column.
  - Sort: `sort_values(['category_num', 'score'])`.
- **Starter:**
  ```python
  mapping = {'A': 1, 'B': 2, 'C': 3}
  data['category_num'] = data['category'].map(mapping)
  data = data.sort_values(['category_num', 'score'])
  ```

## Common pitfalls
- Forgetting to **reassign** the DataFrame after a non-in-place method (`drop_duplicates`, `dropna`, `sort_values`).
- Computing the mean/std **before** removing sentinels — they pollute the statistic.
- `get_dummies` returns a new frame; you usually want to `concat` it back to the original or assign it to a new variable rather than overwriting `data`.
- `.str` methods return NaN for NaN inputs unless you `.fillna('')` first.
