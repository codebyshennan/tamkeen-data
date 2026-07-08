# Assignment: Data Wrangling with Pandas

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to create your dataset:

```python
import pandas as pd
import numpy as np

# Create a messy dataset
data = pd.DataFrame({
    'name': ['John Smith', 'Jane Doe', 'Bob Wilson', 'MARY JONES', 'john smith', np.nan, 'Jane doe'],
    'email': ['john@gmail.com', 'jane@yahoo.com', np.nan, 'mary@gmail.com', 'john2@gmail.com', 'unknown@test.com', 'jane@yahoo.com'],
    'age': [25, 30, -999, 35, 25, 40, 30],
    'category': ['A', 'B', 'A', 'C', 'A', 'B', 'B'],
    'score': [85.5, 90.0, 77.5, 995.0, 85.5, 88.0, 90.0]
})
```

## Tasks

1. Data Cleaning:

   - Remove duplicate rows based on the 'name' column (case-insensitive)
   - Replace the age value -999 with NaN
   - Remove any rows with missing names

2. String Manipulation:

   - Convert all names to title case
   - Create a new column 'domain' that extracts the domain part from email addresses (everything after @)
   - Create a boolean column 'is_gmail' that is True if the email is from gmail.com

3. Categorical Data:

   - Convert the 'category' column to categorical type
   - Create dummy variables for the category column
   - Display the frequency count of categories

4. Handling Outliers:

   - Find any scores that are more than 2 standard deviations from the mean
   - Replace these outlier scores with the mean score
   - Calculate and display descriptive statistics for the cleaned score column

5. Data Transformation:
   - Create a dictionary that maps categories to numeric values (A=1, B=2, C=3)
   - Add a new column 'category_num' using this mapping
   - Sort the DataFrame by category_num and score

## Deliverable

Submit your solution as a Python script with:

1. All code clearly commented with
   1. Brief explanations of your approach for complex operations
   2. Any assumptions or additional features you implemented
2. Results for each task printed with appropriate labels

## Hints

<details>
<summary>Show hints</summary>

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
- **Where:** [Transformations](../transformations.md), string operations.
- **Think:** Vectorized string methods live on `.str`.
  - **Title case:** `data['name'].str.title()`.
  - **Email domain:** `data['email'].str.split('@').str[-1]`.
  - **`is_gmail`:** compare the extracted domain to `'gmail.com'`, **or** use `.str.endswith('@gmail.com')` directly. Watch for `NaN` emails, the result will also be NaN unless you handle it.

## 3. Categorical data
- **Where:** [Transformations](../transformations.md), categorical encoding.
- **Think:**
  - Convert with `astype('category')`.
  - Dummies with `pd.get_dummies(data['category'])`, returns a new DataFrame of 0/1 columns.
  - Frequency: `data['category'].value_counts()`.

## 4. Handling outliers
- **Where:** [Outliers](../outliers.md).
- **Think:** "More than 2 SDs from the mean" = absolute Z-score above 2.
  - `z = (s - s.mean()) / s.std(); outliers = data[z.abs() > 2]`.
  - Replace by assignment: `data.loc[z.abs() > 2, 'score'] = data['score'].mean()`.
  - Watch for chicken-and-egg: if you replace **then** describe, do the calculation on the cleaned column.

## 5. Data transformation
- **Where:** [Transformations](../transformations.md), mapping.
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
- Computing the mean/std **before** removing sentinels, they pollute the statistic.
- `get_dummies` returns a new frame; you usually want to `concat` it back to the original or assign it to a new variable rather than overwriting `data`.
- `.str` methods return NaN for NaN inputs unless you `.fillna('')` first.

</details>
