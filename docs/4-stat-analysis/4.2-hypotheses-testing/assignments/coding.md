# Assignment: Hypothesis Testing

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to create your datasets:

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# Dataset 1: Exam scores for two teaching methods
method_a = np.array([72, 68, 75, 80, 74, 70, 78, 65, 82, 76,
                     71, 69, 77, 73, 79, 66, 83, 75, 70, 74])
method_b = np.array([78, 82, 85, 80, 88, 76, 90, 83, 87, 79,
                     84, 81, 86, 77, 91, 85, 80, 88, 82, 84])

# Dataset 2: Customer purchase decisions by ad type (counts)
# Rows: [Ad Type A, Ad Type B]
# Cols: [Purchased, Did Not Purchase]
purchase_table = np.array([
    [45, 55],   # Ad Type A: 45 purchased, 55 did not
    [62, 38],   # Ad Type B: 62 purchased, 38 did not
])

# Dataset 3: Blood pressure before and after a 4-week exercise programme
before = np.array([145, 138, 152, 160, 143, 149, 155, 141, 163, 147])
after  = np.array([138, 130, 145, 151, 137, 142, 148, 136, 155, 140])
```

## Tasks

### Task 1: Formulate hypotheses

Before running any test, write out the null and alternative hypotheses in plain language **and** in mathematical notation for each of the three datasets.

- For Dataset 1 (two independent groups): state H₀ and H₁ for the difference in mean exam scores.
- For Dataset 2 (contingency table): state H₀ and H₁ about whether ad type and purchase decision are independent.
- For Dataset 3 (paired measurements): state H₀ and H₁ for the change in blood pressure.

For each, decide **before** looking at any output whether you will use a one-sided or two-sided test, and justify your choice in a comment.

### Task 2: Run a two-sample t-test (Dataset 1)

- Check whether the variances of `method_a` and `method_b` appear roughly equal using `np.var(..., ddof=1)`. Print both values.
- Run `scipy.stats.ttest_ind(method_a, method_b)` to compare the two teaching methods.
- Print the t-statistic and p-value.
- At α = 0.05, state your conclusion in plain language: is there sufficient evidence that the two methods produce different mean scores?
- Compute the mean of each group and the difference in means.

### Task 3: Run a chi-squared test (Dataset 2)

- Use `scipy.stats.chi2_contingency(purchase_table)` on the 2×2 table.
- Print the chi-squared statistic, p-value, degrees of freedom, and the expected frequency table.
- Verify that all expected frequencies are ≥ 5 (a standard assumption check for the chi-squared test).
- At α = 0.05, conclude whether ad type and purchase decision are independent.

### Task 4: Interpret and compare results

- For Dataset 3 (paired data), run `scipy.stats.ttest_rel(before, after)` and print the result.
- Build a summary table (as a printed DataFrame or formatted print statements) showing for all three tests: test name, test statistic, p-value, and your conclusion at α = 0.05.
- Write a short comment (2-4 sentences) explaining why you used a different test for each dataset, connecting your choice to the type of data and study design.

## Deliverable

Submit your solution as a Python script with:

1. All hypotheses written as comments before the code that tests them.
2. Results for each task printed with clear labels.
3. A plain-language conclusion for each test at α = 0.05.
4. A short comment on test selection (Task 4).

## Hints

<details>
<summary>Show hints</summary>

### Task 1: Formulating hypotheses
- **Where:** [Hypothesis Formulation](../hypothesis-formulation.md), "The Anatomy of a Hypothesis".
- **Think:** The null always represents "no difference" or "no association." The alternative should reflect the specific research question, is there a directional expectation (one-sided) or just "any difference" (two-sided)?
- **One vs two-sided:** [Hypothesis Formulation](../hypothesis-formulation.md), "The Three Pillars of Good Hypotheses".

### Task 2: Two-sample t-test
- **Where:** [Statistical Tests](../statistical-tests.md), "T-Tests: Comparing Means".
- **Think:**
  - `ttest_ind` assumes the two groups are independent, confirm that assumption holds for Dataset 1.
  - The function returns `(statistic, pvalue)`. Unpack both: `t_stat, p_val = stats.ttest_ind(...)`.
  - Compare `p_val` to your chosen α; write the conclusion before checking whether it aligns with intuition.
- **Starter:**
  ```python
  t_stat, p_val = stats.ttest_ind(method_a, method_b)
  print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
  ```

```
t = -6.376, p = 0.0000
```

### Task 3: Chi-squared test
- **Where:** [Statistical Tests](../statistical-tests.md), "Chi-Square Tests: Testing for Independence".
- **Think:**
  - `chi2_contingency` returns `(chi2, p, dof, expected)`. Capture all four values.
  - Print the `expected` array and check that every cell is ≥ 5; if not, Fisher's exact test would be more appropriate.
  - Degrees of freedom = (rows − 1) × (cols − 1) = 1 for a 2×2 table.
- **Starter:**
  ```python
  chi2, p, dof, expected = stats.chi2_contingency(purchase_table)
  print(f"chi2 = {chi2:.3f}, p = {p:.4f}, dof = {dof}")
  print("Expected frequencies:\n", expected)
  ```

```
chi2 = 5.145, p = 0.0233, dof = 1
Expected frequencies:
 [[53.5 46.5]
 [53.5 46.5]]
```

### Task 4: Paired t-test and summary
- **Where:** [Statistical Tests](../statistical-tests.md), "T-Tests: Comparing Means" (paired variant) and [Experimental Design](../experimental-design.md), "The Three Pillars of Experimental Design".
- **Think:**
  - `ttest_rel` is used when the same subjects are measured twice; the test operates on the differences `before - after`.
  - For the summary table, consider using a list of dicts fed into `pd.DataFrame(...)` so the table is tidy.
  - Test selection logic: independent continuous → `ttest_ind`; counts in categories → `chi2_contingency`; paired continuous → `ttest_rel`.

### Common pitfalls
- Writing hypotheses after looking at results ("HARKing", Hypothesizing After Results are Known) inflates Type I error.
- Using `ttest_ind` on the before/after data would ignore the pairing and lose statistical power.
- Forgetting to check the expected-frequency assumption for chi-squared can produce unreliable p-values with small cell counts.
- The two-sided p-value from `ttest_ind` should not be halved to get a one-sided p-value unless you pre-committed to a directional hypothesis before data collection.

</details>
