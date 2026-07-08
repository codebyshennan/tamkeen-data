# Assignment: Relationships in Data

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to create your dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(0)

n = 80

study_hours   = np.random.uniform(1, 10, n)
sleep_hours   = np.random.uniform(4, 9, n)
prior_score   = np.random.uniform(40, 80, n)

# Exam score depends on all three predictors plus noise
exam_score = (
    5.0 * study_hours
    + 2.5 * sleep_hours
    + 0.4 * prior_score
    + np.random.normal(0, 5, n)
)
exam_score = np.clip(exam_score, 0, 100)

df = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'prior_score': prior_score,
    'exam_score':  exam_score,
})
```

## Tasks

### Task 1: Correlation analysis

- Compute the Pearson correlation between each of the three predictor columns (`study_hours`, `sleep_hours`, `prior_score`) and `exam_score`. Print each coefficient and its p-value using `scipy.stats.pearsonr`.
- Compute the full correlation matrix for all four columns using `df.corr()` and print it.
- Create a scatter plot of `study_hours` vs `exam_score`. Label the axes and add a title.
- In a comment, state which predictor has the strongest linear association with `exam_score` and interpret the direction of the relationship.

### Task 2: Simple linear regression

- Fit a simple linear regression model predicting `exam_score` from `study_hours` alone using `sklearn.linear_model.LinearRegression`.
- Print the intercept and coefficient (slope) with clear labels.
- Interpret the slope in plain language: "For each additional hour of study, the predicted exam score changes by …"
- Compute and print R² for the fitted model.
- Add the regression line to the scatter plot from Task 1 and save or display the figure.

### Task 3: Multiple linear regression

- Fit a multiple linear regression model using all three predictors (`study_hours`, `sleep_hours`, `prior_score`) to predict `exam_score`.
- Print the intercept and all three coefficients. For each, write a one-sentence conditional interpretation (e.g., "Holding sleep hours and prior score fixed, …").
- Compute R² and RMSE for the multiple regression model.
- Compare R² from the simple model (Task 2) to the multiple model. Print both values side by side and comment on whether adding predictors improved fit.

### Task 4: Residual diagnostics

- Using the multiple regression model from Task 3, compute the residuals: `residuals = exam_score - model.predict(X)`.
- Create two diagnostic plots:
  1. **Residuals vs Fitted values**: scatter plot of predicted values (x-axis) against residuals (y-axis). Add a horizontal line at y = 0.
  2. **Residual histogram**: plot a histogram of the residuals and overlay a normal distribution curve using `scipy.stats.norm.fit`.
- In a comment, describe what you observe: do residuals appear randomly scattered around zero, or is there a pattern? What would a pattern indicate?

## Deliverable

Submit your solution as a Python script with:

1. All print statements clearly labelled.
2. Interpretations written as code comments.
3. Both Task 1 scatter and Task 2 regression-line plot produced (either saved as PNG or displayed).
4. Residual diagnostic plots from Task 4.

## Hints

<details>
<summary>Show hints</summary>

### Task 1: Correlation analysis
- **Where:** [Correlation Analysis](../correlation-analysis.md), "What is Correlation Analysis?" and Pearson coefficient section.
- **Think:**
  - `stats.pearsonr(x, y)` returns `(r, p_value)`. Print both; a large |r| with small p confirms a real linear relationship.
  - `df.corr()` uses Pearson by default. Check whether any predictor pair has high correlation, that foreshadows multicollinearity in Task 3.
- **Starter:**
  ```python
  for col in ['study_hours', 'sleep_hours', 'prior_score']:
      r, p = stats.pearsonr(df[col], df['exam_score'])
      print(f"{col}: r = {r:.3f}, p = {p:.4f}")
  ```

```
study_hours: r = 0.847, p = 0.0000
sleep_hours: r = 0.041, p = 0.7201
prior_score: r = 0.212, p = 0.0590
```

### Task 2: Simple linear regression
- **Where:** [Simple Linear Regression](../simple-linear-regression.md), "Fitting the model" and "Interpreting coefficients".
- **Think:**
  - sklearn expects a 2-D feature matrix: `X = df[['study_hours']]` (double brackets).
  - `model.coef_[0]` is the slope; `model.intercept_` is the intercept.
  - R² from `model.score(X, y)` ranges from 0 to 1; values closer to 1 indicate better fit.
- **Starter:**
  ```python
  from sklearn.linear_model import LinearRegression
  X = df[['study_hours']]
  y = df['exam_score']
  model = LinearRegression().fit(X, y)
  print(f"Intercept: {model.intercept_:.2f}, Slope: {model.coef_[0]:.2f}")
  print(f"R² = {model.score(X, y):.3f}")
  ```

```
Intercept: 42.54, Slope: 4.45
R² = 0.717
```

### Task 3: Multiple linear regression
- **Where:** [Multiple Linear Regression](../multiple-linear-regression.md), "Interpreting Coefficients" and "Model Evaluation".
- **Think:**
  - The conditional interpretation matters: each coefficient describes the effect of one predictor *holding the other predictors constant*.
  - RMSE = `np.sqrt(mean_squared_error(y, y_pred))`.
  - If R² jumps significantly from simple to multiple regression, the additional predictors are contributing real explanatory power.

### Task 4: Residual diagnostics
- **Where:** [Simple Linear Regression](../simple-linear-regression.md), "Residual Diagnostics" section.
- **Think:**
  - A random scatter of residuals around zero supports the linearity and homoscedasticity assumptions.
  - A funnel shape (variance growing with fitted values) indicates heteroscedasticity.
  - A curved pattern suggests the linear form misses non-linearity in the data.
- **Starter:**
  ```python
  y_pred = model.predict(X_multi)
  residuals = y - y_pred
  plt.scatter(y_pred, residuals, alpha=0.6)
  plt.axhline(0, color='red', linestyle='--')
  plt.xlabel('Fitted values')
  plt.ylabel('Residuals')
  plt.title('Residuals vs Fitted')
  plt.show()
  ```

### Common pitfalls
- Passing a 1-D array as `X` to sklearn causes an error; always reshape with `df[['col']]` (double brackets) or `.reshape(-1, 1)`.
- Confusing R² with correlation: R² = r² only for simple regression. For multiple regression, R² summarises overall fit across all predictors.
- Interpreting multiple regression coefficients as raw correlations ignores the conditioning on other predictors; a coefficient can flip sign relative to the simple correlation when predictors are correlated.
- Computing residuals before calling `.fit()` gives you nothing useful; always fit first, then predict, then subtract.

</details>
