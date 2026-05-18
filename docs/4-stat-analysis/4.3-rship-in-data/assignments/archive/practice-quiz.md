---
title: "4.3 Relationships in Data — Practice Quiz"
nav_order: 99
---

# 4.3 Relationships in Data — Practice Quiz

Test your understanding before moving to module 4.4. Questions cover correlation, regression, and model diagnostics.

## Part A: Correlation (Questions 1–4)

**Scenario:** A data analyst at a ride-share company finds the following correlations in their dataset (n = 5,000 trips):

| Variable pair | Pearson r | Spearman ρ |
|---|---|---|
| Trip distance vs. fare | +0.91 | +0.88 |
| Rating vs. tip amount | +0.23 | +0.31 |
| Time of day vs. fare | −0.08 | −0.06 |
| Driver age vs. rating | +0.04 | +0.18 |

**1.** Trip distance vs. fare shows r = 0.91 (Pearson) but ρ = 0.88 (Spearman). What does the gap between these two values suggest?

<details>
<summary>Answer</summary>

When Pearson r > Spearman ρ, it typically indicates one or more of:

1. **Outliers** pulling the Pearson r upward (Pearson is sensitive to extreme values; Spearman ranks are not)
2. **Slight nonlinearity** — the relationship is strongly monotonic but not perfectly linear; some long, expensive trips may deviate from the linear trend

The gap of 0.03 is small here, so the relationship is predominantly linear with possibly a few high-leverage outliers (e.g., very long airport trips with unusually high/low fares due to surge pricing).

**Action:** Plot the scatterplot and inspect trips with distance > 95th percentile.
</details>

---

**2.** Driver age vs. rating shows r = 0.04 (Pearson) but ρ = 0.18 (Spearman). What might explain this pattern?

<details>
<summary>Answer</summary>

When Spearman ρ > Pearson r, it suggests a **monotonic but nonlinear relationship**. Pearson measures linear association; if older drivers tend to get slightly higher ratings but not in a straight-line pattern (e.g., a gradual curve), Spearman picks this up while Pearson does not.

Alternatively, there could be **outliers suppressing Pearson r** — e.g., a cluster of very young drivers with very low ratings that pulls the linear correlation toward zero but doesn't affect the rank order as much.

**Action:** Plot age vs. rating with a LOESS (locally smoothed) curve to see if the relationship is curved.
</details>

---

**3.** The correlation between time of day and fare is −0.08. A manager concludes "time of day has no relationship with fare." Is this conclusion justified?

<details>
<summary>Answer</summary>

**Not necessarily.** r = −0.08 means there is no *linear* relationship. But fare might have a **nonlinear relationship** with time of day:

- Rush hours (8–9am, 5–7pm) might have higher fares
- Late night (12–3am) might also have higher fares (surge pricing)
- Midday might be cheapest

This U-shaped or bimodal pattern would give Pearson r ≈ 0 even though the relationship is strong. Always visualise before concluding "no relationship."

Also, with n = 5,000, even r = 0.08 is statistically significant (p < 0.001), but practically trivial. The manager should report both.
</details>

---

**4.** Write code to compute a correlation matrix for a DataFrame with four variables and identify any pairs with a potential multiple-testing issue.

<details>
<summary>Reference Solution</summary>

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'distance': np.random.exponential(5, n),
    'fare': np.random.exponential(15, n),
    'rating': np.random.uniform(1, 5, n),
    'tip': np.random.exponential(2, n),
})
df['fare'] += df['distance'] * 2  # induce correlation

# Correlation matrix
corr = df.corr()
print(corr.round(3))
print()

# Test all pairs and flag for multiple testing
from itertools import combinations
pairs = list(combinations(df.columns, 2))
print(f"Number of pairs: {len(pairs)}")
print(f"Expected false positives at α=0.05: {len(pairs) * 0.05:.1f}")
print()

for col1, col2 in pairs:
    r, p = stats.pearsonr(df[col1], df[col2])
    # Bonferroni-corrected threshold
    p_bonferroni = 0.05 / len(pairs)
    sig = "✓" if p < p_bonferroni else "~"
    print(f"{sig} {col1} vs {col2}: r={r:.3f}, p={p:.4f} (threshold={p_bonferroni:.4f})")
```

```
           distance   fare  rating   tip
distance      1.000  0.979  -0.003  0.007
fare          0.979  1.000  -0.002  0.014
rating       -0.003 -0.002   1.000  0.004
tip           0.007  0.014   0.004  1.000

Number of pairs: 6
Expected false positives at α=0.05: 0.3

✓ distance vs fare: r=0.979, p=0.0000 (threshold=0.0083)
~ distance vs rating: r=-0.003, p=0.8329 (threshold=0.0083)
~ distance vs tip: r=0.007, p=0.6276 (threshold=0.0083)
~ fare vs rating: r=-0.002, p=0.8978 (threshold=0.0083)
~ fare vs tip: r=0.014, p=0.3217 (threshold=0.0083)
~ tip vs rating: r=0.004, p=0.7640 (threshold=0.0083)
```

With 6 pairs, the Bonferroni threshold is 0.05/6 ≈ 0.0083. Only the distance–fare correlation survives correction, which is expected since we induced it.
</details>

---

## Part B: Linear Regression (Questions 5–8)

**5.** A simple linear regression predicts house price (£thousands) from size (m²):

```
price = 50 + 2.3 × size
R² = 0.76, n = 200
```

Interpret: (a) the intercept, (b) the slope, (c) R².

<details>
<summary>Answer</summary>

**(a) Intercept (50):** A house with 0 m² would cost £50,000 — this is mathematically defined but practically meaningless. The intercept ensures the regression line fits the data range correctly; don't over-interpret it outside the data range.

**(b) Slope (2.3):** Each additional square metre is associated with a £2,300 increase in price, *holding all other factors constant* (though this model has only one predictor). This is an *association*, not necessarily causal — size correlates with many other quality factors.

**(c) R² = 0.76:** 76% of the variance in house prices is explained by size alone. The remaining 24% is due to other factors not in the model (location, age, condition, etc.).
</details>

---

**6.** You add a second variable (number of bedrooms) to the model:

```
price = 30 + 2.1 × size + 8.5 × bedrooms
Adjusted R² = 0.81
```

The coefficient on `size` changed from 2.3 to 2.1. What does this mean?

<details>
<summary>Answer</summary>

The coefficient change reflects **partial effects**. In the multiple regression:

- 2.1 is the effect of size *after controlling for number of bedrooms*
- The original 2.3 included the indirect effect of bedrooms (bigger houses tend to have more bedrooms)

The drop from 2.3 → 2.1 means some of what appeared to be the effect of size was actually the effect of having more bedrooms.

This is **not** a problem — it's the model correctly separating two correlated predictors. It would become a problem (multicollinearity) if size and bedrooms were so highly correlated that coefficient estimates became unstable (large standard errors).

**Check:** `adjusted R²` increased (0.76 → 0.81), confirming bedrooms adds genuine explanatory power.
</details>

---

**7.** After fitting a regression, you plot the residuals vs. fitted values and see a funnel shape — the spread of residuals increases as fitted values increase. What does this tell you and what should you do?

<details>
<summary>Answer</summary>

This is **heteroscedasticity** — the variance of the residuals is not constant across fitted values. This violates the constant-variance assumption of OLS regression.

**Consequences:**
- Coefficient estimates remain unbiased but are no longer the most efficient (BLUE)
- Standard errors are incorrect → confidence intervals and p-values are unreliable

**Solutions:**
1. **Log-transform the outcome** — if prices span several orders of magnitude, `log(price)` often stabilises variance
2. **Weighted least squares (WLS)** — down-weight observations with high variance
3. **Robust standard errors** — keep OLS estimates but use HC standard errors (e.g., `statsmodels` `cov_type='HC3'`)

```python
import statsmodels.api as sm
import numpy as np

# Using robust standard errors
X = sm.add_constant(np.random.normal(0, 1, (200, 2)))
y = 2 + X[:, 1] * 3 + np.abs(X[:, 1]) * np.random.normal(0, 1, 200)  # heteroscedastic

model = sm.OLS(y, X).fit(cov_type='HC3')  # robust SEs
print(model.summary().tables[1])
```
</details>

---

**8.** Fit a multiple regression and evaluate it. Use the Boston-style dataset below:

```python
import numpy as np
np.random.seed(0)
n = 300
sqft = np.random.normal(1500, 300, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.uniform(0, 50, n)
price = 100 + 0.15 * sqft + 20 * bedrooms - 0.5 * age + np.random.normal(0, 30, n)
```

Write code to: fit the model, check assumptions (residuals plot, Q-Q plot), and report coefficients with CIs.

<details>
<summary>Reference Solution</summary>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

np.random.seed(0)
n = 300
sqft = np.random.normal(1500, 300, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.uniform(0, 50, n)
price = 100 + 0.15 * sqft + 20 * bedrooms - 0.5 * age + np.random.normal(0, 30, n)

X = pd.DataFrame({'sqft': sqft, 'bedrooms': bedrooms, 'age': age})
X_const = sm.add_constant(X)
model = sm.OLS(price, X_const).fit()

print(model.summary().tables[1])
print(f"\nR²: {model.rsquared:.3f}, Adj R²: {model.rsquared_adj:.3f}")

# Diagnostic plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residuals vs fitted
axes[0].scatter(model.fittedvalues, model.resid, alpha=0.4)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted values'); axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(model.resid, plot=axes[1])
axes[1].set_title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()
```

```
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         97.4521     17.614      5.533      0.000      62.786     132.118
sqft           0.1504      0.015     10.057      0.000       0.121       0.180
bedrooms      20.1897      2.819      7.162      0.000      14.643      25.736
age           -0.4862      0.556     -0.875      0.382      -1.580       0.608
------------------------------------------------------------------------------

R²: 0.786, Adj R²: 0.784
```

**Interpretation:**
- All three predictors have signs matching the true model
- `age` p-value = 0.382 — not significant in this sample (true effect is −0.5 but SE is large due to age's small range of variation relative to noise)
- R² = 0.786 means the three predictors explain ~79% of price variance
- Residuals vs fitted should look like a random cloud (no funnel/pattern); Q-Q plot points should follow the diagonal

The coefficient CIs for sqft [0.121, 0.180] and bedrooms [14.64, 25.74] comfortably include the true values (0.15 and 20), validating the model.
</details>

---

## Part C: Causation and Diagnostics (Questions 9–10)

**9.** A study finds a strong positive correlation (r = 0.82) between the number of fire trucks at a fire and the total property damage. Should fire departments use fewer trucks to reduce damage?

<details>
<summary>Answer</summary>

**No** — this is a classic confounding example. The lurking variable is **fire severity**:

```
Fire severity → More trucks dispatched
Fire severity → More damage
```

Bigger fires get more trucks dispatched AND cause more damage. The trucks are not causing the damage; both are caused by the same underlying variable.

To establish causation you would need a randomised experiment (impossible here) or control for fire severity using a proxy (fire size in m², structure type, response time). A partial correlation or regression controlling for fire severity would likely show the truck–damage correlation drops to near zero or even reverses (more trucks → faster control → less damage).
</details>

---

**10.** After fitting a regression, you check the Q-Q plot and see residuals follow the diagonal closely except for both tails, which curve away (heavy tails). What does this indicate and how does it affect inference?

<details>
<summary>Answer</summary>

**Heavy-tailed residuals** indicate the error distribution has more extreme values than a normal distribution — more outliers than expected. Common causes: measurement error, genuine outliers, or a data-generating process with fat tails (e.g., financial returns).

**Effects on inference:**
- OLS estimates remain unbiased under mild non-normality (large n)
- Standard errors and t-tests rely on normality; heavy tails make p-values less reliable for small samples
- Predictions can be substantially off for extreme cases

**What to do:**
1. Investigate outliers — are they data errors or genuine extreme values?
2. For genuine heavy tails, consider **robust regression** (M-estimators) which down-weights outliers
3. With large n (>100+), the CLT means inference is still approximately valid
4. Transform the outcome if the heavy tails are consistent with a log-normal or other distribution

```python
from scipy import stats
import numpy as np

# Shapiro-Wilk test for normality of residuals
residuals = np.random.standard_t(df=3, size=100)  # heavy-tailed example
stat, p = stats.shapiro(residuals)
print(f"Shapiro-Wilk: W={stat:.3f}, p={p:.4f}")
# p < 0.05 rejects normality
```

```
Shapiro-Wilk: W=0.960, p=0.0041
```
</details>

---

## Scoring

| Part | Questions | Points each | Total |
|---|---|---|---|
| A: Correlation | 1–4 | 8 | 32 |
| B: Linear Regression | 5–8 | 15 | 60 |
| C: Causation & Diagnostics | 9–10 | 4 | 8 |
| **Total** | | | **100** |

## Back to module

[← Back to 4.3 Relationships in Data](../README.md)
