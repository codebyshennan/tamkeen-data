# Assignment: Statistical Modelling

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to create your datasets:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)

np.random.seed(7)

# --- Dataset 1: Binary classification (loan approval) ---
n_cls = 200
income        = np.random.normal(50, 15, n_cls)          # $k
credit_score  = np.random.normal(650, 80, n_cls)
debt_ratio    = np.random.uniform(0.1, 0.6, n_cls)
log_odds      = -8 + 0.06 * income + 0.008 * credit_score - 5 * debt_ratio
prob_approve  = 1 / (1 + np.exp(-log_odds))
approved      = (np.random.uniform(size=n_cls) < prob_approve).astype(int)

df_cls = pd.DataFrame({
    'income':       income,
    'credit_score': credit_score,
    'debt_ratio':   debt_ratio,
    'approved':     approved,
})

# --- Dataset 2: Non-linear relationship (engine RPM → fuel efficiency) ---
n_poly = 60
rpm = np.linspace(1000, 6000, n_poly)
mpg = 18 + 0.012 * rpm - 0.0000018 * rpm**2 + np.random.normal(0, 1.5, n_poly)

# --- Dataset 3: Many predictors for regularisation ---
n_reg = 150
X_reg = np.random.randn(n_reg, 20)    # 20 features, most are noise
# Only features 0, 3, 7 have real signal
y_reg = (2.5 * X_reg[:, 0]
         - 1.8 * X_reg[:, 3]
         + 3.1 * X_reg[:, 7]
         + np.random.normal(0, 2, n_reg))
```

## Tasks

### Task 1: Logistic regression: fitting and interpreting

- Split `df_cls` into features `X` (income, credit_score, debt_ratio) and target `y` (approved). Use `train_test_split` with `test_size=0.25` and `random_state=7`.
- Scale the features with `StandardScaler` before fitting `LogisticRegression`.
- Print the model's coefficients alongside feature names. For each coefficient, state whether the predictor increases or decreases the log-odds of approval, and by how much.
- Convert each coefficient to an **odds ratio** (`np.exp(coef)`) and print both side by side.
- Evaluate the model on the test set: print accuracy, the confusion matrix, and the full classification report.

### Task 2: Polynomial regression: spotting overfitting

- Using the `rpm` and `mpg` arrays from Dataset 2, fit polynomial regression models of degrees 1, 2, 4, and 8 using `Pipeline([('poly', PolynomialFeatures(degree=d)), ('lr', LinearRegression())])`.
- For each degree, compute the **training R²** and the **5-fold cross-validated R²** using `cross_val_score(..., cv=5, scoring='r2')`.
- Print a table showing degree, training R², and mean CV R² for each degree.
- Plot all four fitted curves on a single scatter plot of the raw data. Use a smooth range of x-values (`np.linspace(1000, 6000, 300)`) for the prediction curves.
- In a comment, identify which degree best balances fit quality and generalisability, and describe what happens to CV R² at degree 8.

### Task 3: Ridge vs Lasso regularisation

- Using Dataset 3 (`X_reg`, `y_reg`), scale the features with `StandardScaler`.
- Fit a `Ridge(alpha=1.0)` and a `Lasso(alpha=0.5)` model. For each, print all 20 coefficients alongside their feature index.
- Count and print how many Lasso coefficients are exactly zero. Count how many Ridge coefficients are exactly zero. Compare the two.
- Re-fit both models across a range of alpha values (`[0.01, 0.1, 1.0, 10.0, 100.0]`) and print the 5-fold CV R² for each combination using `cross_val_score`.
- In a comment, explain the qualitative difference: why does Lasso produce exact zeros while Ridge does not? Connect your observation to the lesson's description of L1 vs L2 penalties.

### Task 4: Model selection and comparison

- Using Dataset 3, compare four candidate models with 5-fold cross-validated R²:
  1. Linear regression (no regularisation)
  2. Ridge with `alpha=1.0`
  3. Lasso with `alpha=0.5`
  4. Polynomial degree-2 + Ridge (`alpha=1.0`)
- Print a summary table with model name and mean CV R² (± standard deviation).
- Select the best model and justify the choice in a short comment (2-3 sentences).
- For the best model, print the non-zero coefficients and their feature indices.

## Deliverable

Submit your solution as a Python script with:

1. All print statements clearly labelled.
2. Log-odds and odds ratio interpretations for Task 1 written as comments.
3. Overfitting analysis for Task 2 written as a comment.
4. Ridge vs Lasso comparison for Task 3 written as a comment.
5. Model selection justification for Task 4 written as a comment.

## Hints

<details>
<summary>Show hints</summary>

### Task 1: Logistic regression
- **Where:** [Logistic Regression](../logistic-regression.md), "Interpreting Coefficients" and "Odds Ratios".
- **Think:**
  - Scale before fitting: `scaler = StandardScaler(); X_train_sc = scaler.fit_transform(X_train)`. Call `.transform` (not `.fit_transform`) on the test set.
  - Coefficients live at `model.coef_[0]` (a 1-D array); access names from `df_cls.columns[:3]`.
  - Odds ratio = `np.exp(coef)`. A ratio > 1 means the predictor increases the odds of approval; < 1 means it decreases them.
  - Confusion matrix: rows are actual, columns are predicted. The diagonal is correct predictions.
- **Starter:**
  ```python
  scaler = StandardScaler()
  X_train_sc = scaler.fit_transform(X_train)
  X_test_sc  = scaler.transform(X_test)
  model = LogisticRegression(random_state=7).fit(X_train_sc, y_train)
  for name, coef in zip(['income','credit_score','debt_ratio'], model.coef_[0]):
      print(f"{name}: coef={coef:.3f}, OR={np.exp(coef):.3f}")
  ```

### Task 2: Polynomial regression and overfitting
- **Where:** [Polynomial Regression](../polynomial-regression.md), "Diagnosing Under- and Overfitting" and "Cross-Validation".
- **Think:**
  - Training R² always improves (or stays the same) as degree increases. CV R² will eventually fall, that's the overfitting signature.
  - For the smooth prediction curve, create `x_plot = np.linspace(1000, 6000, 300).reshape(-1, 1)` and pass through each pipeline.
  - A degree-8 polynomial with only 60 data points will almost certainly overfit badly.
- **Starter:**
  ```python
  from sklearn.linear_model import LinearRegression
  rpm_2d = rpm.reshape(-1, 1)
  for d in [1, 2, 4, 8]:
      pipe = Pipeline([('poly', PolynomialFeatures(d)), ('lr', LinearRegression())])
      cv_r2 = cross_val_score(pipe, rpm_2d, mpg, cv=5, scoring='r2')
      pipe.fit(rpm_2d, mpg)
      train_r2 = r2_score(mpg, pipe.predict(rpm_2d))
      print(f"Degree {d}: train R²={train_r2:.3f}, CV R²={cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
  ```

```
Degree 1: train R²=0.040, CV R²=-8.098 ± 5.749
Degree 2: train R²=0.880, CV R²=0.464 ± 0.318
Degree 4: train R²=0.881, CV R²=-0.038 ± 0.589
Degree 8: train R²=0.651, CV R²=-61.628 ± 119.277
```

### Task 3: Ridge vs Lasso
- **Where:** [Regularization](../regularization.md), "Ridge (L2) vs Lasso (L1)" and "Bias-Variance Tradeoff".
- **Think:**
  - Always scale features before Ridge or Lasso, the penalty is not scale-invariant.
  - Check for exact zeros with `np.sum(model.coef_ == 0)`. Lasso sets some to exactly 0; Ridge never does (it shrinks toward 0 but not to it).
  - The lesson explains this geometrically: the L1 constraint diamond has corners on axes; the L2 ball does not.
- **Starter:**
  ```python
  scaler = StandardScaler()
  X_sc = scaler.fit_transform(X_reg)
  ridge = Ridge(alpha=1.0).fit(X_sc, y_reg)
  lasso = Lasso(alpha=0.5).fit(X_sc, y_reg)
  print("Ridge zeros:", np.sum(ridge.coef_ == 0))
  print("Lasso zeros:", np.sum(lasso.coef_ == 0))
  ```

```
Ridge zeros: 0
Lasso zeros: 17
```

### Task 4: Model selection
- **Where:** [Model Selection](../model-selection.md), "Cross-Validation" and "Comparing Candidate Models".
- **Think:**
  - Use the same `X_sc` (scaled) for all models so comparisons are fair.
  - Mean ± std of CV scores gives a sense of both performance and variability; prefer a model with high mean and low std.
  - A Lasso model that selects only the three signal features (indices 0, 3, 7) with non-zero coefficients confirms that regularisation helped recover the true sparse structure.
- **Starter:**
  ```python
  models = {
      'Linear':     LinearRegression(),
      'Ridge':      Ridge(alpha=1.0),
      'Lasso':      Lasso(alpha=0.5),
      'Poly2+Ridge': Pipeline([('poly', PolynomialFeatures(2)), ('ridge', Ridge(alpha=1.0))]),
  }
  for name, m in models.items():
      scores = cross_val_score(m, X_sc, y_reg, cv=5, scoring='r2')
      print(f"{name}: CV R² = {scores.mean():.3f} ± {scores.std():.3f}")
  ```

```
Linear: CV R² = 0.755 ± 0.108
Ridge: CV R² = 0.755 ± 0.107
Lasso: CV R² = 0.753 ± 0.077
Poly2+Ridge: CV R² = -0.241 ± 0.376
```

### Common pitfalls
- Calling `scaler.fit_transform` on the test set leaks information from the test distribution into training; always fit the scaler on train data only and call `.transform` on test.
- Using training R² alone to pick polynomial degree will always favour the highest degree, you need CV R² or a holdout set.
- Passing a 1-D `y` array to sklearn models is usually fine, but double-check if you get a `DataConversionWarning`.
- Lasso with very large alpha will shrink all coefficients to zero; if CV R² is near 0, try a smaller alpha.
- `cross_val_score` with `scoring='r2'` can return negative values for very poor models, that is valid and means the model is worse than a horizontal line at the mean.

</details>
