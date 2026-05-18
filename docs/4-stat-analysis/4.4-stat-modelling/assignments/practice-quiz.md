---
title: "4.4 Statistical Modelling — Practice Quiz"
nav_order: 99
---

# 4.4 Statistical Modelling — Practice Quiz

Test your understanding of logistic regression, polynomial regression, model selection, regularization, and model interpretation.

## Part A: Logistic Regression (Questions 1–3)

**1.** A logistic regression model predicts whether a customer churns (1) or stays (0). The model produces:

```
log-odds(churn) = −2.1 + 0.8 × months_inactive + 0.3 × support_calls − 1.2 × premium_member
```

Interpret each coefficient in terms of odds ratios.

<details>
<summary>Answer</summary>

Convert each coefficient to an odds ratio using `exp(coef)`:

| Coefficient | Value | Odds ratio = exp(value) | Interpretation |
|---|---|---|---|
| months_inactive | 0.8 | exp(0.8) ≈ 2.23 | Each additional inactive month multiplies churn odds by 2.23 (×2.23) |
| support_calls | 0.3 | exp(0.3) ≈ 1.35 | Each additional support call multiplies churn odds by 1.35 (+35%) |
| premium_member | −1.2 | exp(−1.2) ≈ 0.30 | Premium members have 70% lower churn odds vs. non-premium |

```python
import numpy as np

coefs = {'months_inactive': 0.8, 'support_calls': 0.3, 'premium_member': -1.2}
for feature, coef in coefs.items():
    or_val = np.exp(coef)
    change = (or_val - 1) * 100
    direction = "increase" if change > 0 else "decrease"
    print(f"{feature}: OR = {or_val:.2f} ({abs(change):.0f}% {direction} in odds)")
```

```
months_inactive: OR = 2.23 (123% increase in odds)
support_calls: OR = 1.35 (35% increase in odds)
premium_member: OR = 0.30 (70% decrease in odds)
```
</details>

---

**2.** You train a logistic regression on a dataset where 95% of records are class 0 (no churn) and 5% are class 1 (churn). The model achieves 95% accuracy. Is this a good model? What metric should you use instead?

<details>
<summary>Answer</summary>

**No** — a model that always predicts "no churn" would also achieve 95% accuracy without learning anything. This is the **class imbalance problem**.

**Better metrics:**

| Metric | Formula | What it measures |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of predicted churners, how many actually churned? |
| **Recall (Sensitivity)** | TP / (TP + FN) | Of actual churners, how many did the model catch? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve | Discriminative ability across all thresholds |
| **PR-AUC** | Area under precision-recall curve | Better than ROC for severe class imbalance |

For a churn use case, **recall** is usually the priority (catching churners before they leave is more valuable than avoiding false alarms), but you need to consider the cost of retention campaigns.

```python
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Simulate: model always predicts 0 (no churn)
y_true = np.array([0]*950 + [1]*50)
y_pred_naive = np.zeros(1000, dtype=int)

print("Naive model (always predicts 0):")
print(classification_report(y_true, y_pred_naive, target_names=['stay', 'churn']))
# Shows: accuracy=0.95, but recall for churn = 0.00
```
</details>

---

**3.** Write code to train a logistic regression classifier, evaluate it with appropriate metrics, and adjust the decision threshold to improve recall.

<details>
<summary>Reference Solution</summary>

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 2000

# Simulate churn dataset (5% churn rate)
months_inactive = np.random.exponential(2, n)
support_calls = np.random.poisson(1.5, n)
premium = np.random.binomial(1, 0.3, n)

log_odds = -2.1 + 0.8 * months_inactive + 0.3 * support_calls - 1.2 * premium
prob_churn = 1 / (1 + np.exp(-log_odds))
y = np.random.binomial(1, prob_churn)

X = np.column_stack([months_inactive, support_calls, premium])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Train with class_weight to handle imbalance
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_sc, y_train)

# Default threshold (0.5)
y_pred = model.predict(X_test_sc)
print("Default threshold (0.5):")
print(classification_report(y_test, y_pred, target_names=['stay', 'churn']))
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test_sc)[:,1]):.3f}")

# Lower threshold to increase recall
probs = model.predict_proba(X_test_sc)[:, 1]
threshold = 0.3
y_pred_adj = (probs >= threshold).astype(int)
print(f"\nAdjusted threshold ({threshold}):")
print(classification_report(y_test, y_pred_adj, target_names=['stay', 'churn']))
```

**Key insight:** Lowering the threshold (e.g., 0.5 → 0.3) increases recall (catches more churners) at the cost of lower precision (more false alarms). The right threshold depends on the business cost of missing a churner vs. the cost of an unnecessary retention offer.
</details>

---

## Part B: Overfitting and Model Selection (Questions 4–6)

**4.** You fit polynomial regression models of degrees 1–10 to a dataset and get these results:

| Degree | Train RMSE | Test RMSE |
|---|---|---|
| 1 | 4.2 | 4.3 |
| 2 | 2.1 | 2.2 |
| 3 | 2.0 | 2.1 |
| 5 | 1.8 | 2.4 |
| 7 | 1.5 | 3.1 |
| 10 | 0.9 | 6.8 |

Which degree should you choose and why? What is happening at degree 10?

<details>
<summary>Answer</summary>

**Choose degree 2 or 3.** They have nearly identical test RMSE (2.2 and 2.1), and degree 2 is simpler.

**At degree 10:** Classic **overfitting**. Training RMSE is excellent (0.9) but test RMSE (6.8) is 3× worse than degree 2. The model has memorised the training data — it fits the noise, not the signal.

Key indicators:
- Train–test gap widens dramatically after degree 3
- Test RMSE is *increasing* despite training RMSE decreasing
- "More complex" ≠ "better" once the gap opens

**The rule:** Choose the simplest model that minimises out-of-sample error. When two models have similar test performance, prefer the simpler one (Occam's razor / parsimony).
</details>

---

**5.** What is data leakage in the context of cross-validation, and how does it occur with preprocessing?

<details>
<summary>Answer</summary>

**Data leakage** occurs when information from the test/validation fold "leaks" into the training process, making CV scores optimistic and unreliable.

**How it happens with preprocessing:**

```python
# WRONG — leaks test data into preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np

X = np.random.randn(100, 5)
y = X[:, 0] * 2 + np.random.randn(100)

# Fitting scaler on ALL data before CV
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← test fold statistics leak into training

from sklearn.model_selection import cross_val_score
scores = cross_val_score(Ridge(), X_scaled, y, cv=5)
print("Leaked scores:", -scores.mean())  # Optimistic
```

```python
# CORRECT — pipeline ensures preprocessing is re-fit per fold only on training data
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), Ridge())
scores = cross_val_score(pipeline, X, y, cv=5)
print("Correct scores:", -scores.mean())
```

The same applies to: `SelectKBest`, `PCA`, imputation, encoding. **Whenever a transformation "learns" from the data, it must be inside the pipeline.**
</details>

---

**6.** You have 10 candidate models. You use 5-fold CV to evaluate all 10, pick the best, and report its CV score as the "expected test performance." Why is this optimistic?

<details>
<summary>Answer</summary>

This is **model selection bias** (also called "the winner's curse"). By choosing the model with the best CV score out of 10 candidates, you are implicitly optimising over the CV scores themselves. The best CV score is likely an overestimate of true performance because:

1. Each CV score has random variance — the "best" model may have won partly due to luck on the particular folds
2. You've effectively used the CV data to make a selection decision, spending some of your "budget" for unbiased estimation

**The fix:** Use a proper **nested cross-validation** structure:
- **Outer loop:** Evaluates expected performance of the model selection procedure
- **Inner loop:** Performs model selection / hyperparameter tuning

```
Outer fold 1: test on fold 1
  Inner CV: select best model using folds 2-5
  Evaluate selected model on fold 1 (outer test)

Outer fold 2: test on fold 2
  Inner CV: select best model using folds 1, 3-5
  ...
```

Each outer fold's score is an unbiased estimate of the model selection procedure's performance, not the specific model.
</details>

---

## Part C: Regularization (Questions 7–8)

**7.** A Ridge regression model (L2) and a Lasso model (L1) are fit with the same data and same λ. The Ridge model retains all 20 features; the Lasso model uses 8. Explain why, and when you would choose each.

<details>
<summary>Answer</summary>

**L1 (Lasso) shrinks some coefficients exactly to zero**, producing sparse models with built-in feature selection. This happens because the L1 penalty creates a diamond-shaped constraint region with corners on the axes — the optimum often lands on a corner, where some coefficients are exactly zero.

**L2 (Ridge) shrinks all coefficients toward zero but rarely to exactly zero**, because the circular L2 constraint rarely intersects an axis. It distributes the shrinkage evenly across correlated features.

| | Lasso (L1) | Ridge (L2) |
|---|---|---|
| Coefficient sparsity | Yes — some → exactly 0 | No — all shrunk but non-zero |
| Feature selection | Built-in | Not built-in |
| With correlated features | Picks one, discards others | Averages them |
| Best when | Many irrelevant features | All features informative, multicollinearity present |

```python
from sklearn.linear_model import Lasso, Ridge
import numpy as np

np.random.seed(42)
X = np.random.randn(100, 20)
y = X[:, :5] @ [2, -1, 3, 0.5, -2] + np.random.randn(100)  # only 5 true features

lasso = Lasso(alpha=0.1).fit(X, y)
ridge = Ridge(alpha=0.1).fit(X, y)

print(f"Lasso non-zero coefs: {(lasso.coef_ != 0).sum()}")    # ~5
print(f"Ridge non-zero coefs: {(ridge.coef_ != 0).sum()}")    # 20
```
</details>

---

**8.** You train a Ridge model with λ = 0.001 and λ = 1000. Describe what you expect to happen in each case.

<details>
<summary>Answer</summary>

**λ = 0.001 (very weak regularization):**
- Coefficients are close to OLS estimates (minimal shrinkage)
- Model may overfit if n is small relative to p
- Variance is high; bias is low

**λ = 1000 (very strong regularization):**
- Coefficients are heavily shrunk toward zero
- Model is very smooth / simple
- Variance is low; bias is high
- With λ → ∞, all coefficients → 0 and predictions approach the mean

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(0)
X = np.random.randn(50, 10)
y = X[:, 0] * 3 + X[:, 1] * -2 + np.random.randn(50)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

for alpha in [0.001, 1, 100, 1000]:
    model = Ridge(alpha=alpha).fit(X_sc, y)
    coef_norm = np.linalg.norm(model.coef_)
    print(f"λ={alpha:6.3f}  |coef| = {coef_norm:.3f}  coef[0] = {model.coef_[0]:.3f}")
```

```
λ= 0.001  |coef| = 3.681  coef[0] = 3.121
λ= 1.000  |coef| = 3.131  coef[0] = 2.817
λ=100.000  |coef| = 1.368  coef[0] = 1.248
λ=1000.000  |coef| = 0.260  coef[0] = 0.240
```

The coefficient norm shrinks monotonically with λ. **Always standardise features before regularization** — otherwise, features with different scales receive different amounts of shrinkage.
</details>

---

## Part D: Model Interpretation (Questions 9–10)

**9.** A model predicts loan default. Feature importances from a tree-based model show "age" as the top predictor. A colleague says: "The model has learned that older people default more." What questions should you ask before accepting this interpretation?

<details>
<summary>Answer</summary>

Feature importance tells you the feature is *used heavily by the model* — not that the relationship is causal, linear, or even in the direction implied.

**Questions to ask:**

1. **What is the direction?** Feature importance doesn't indicate positive/negative effect. A partial dependence plot (PDP) would show: as age increases, does predicted default probability go up, down, or curve non-monotonically?

2. **Is age a proxy for something else?** Age often correlates with income, employment history, credit history length. The "age effect" might actually be a "credit history length" effect.

3. **Is the feature importance permutation-based or impurity-based?** Impurity-based (Gini) importances in tree models are biased toward high-cardinality and continuous features.

4. **Does this match domain knowledge?** If the business expects default risk to peak for younger borrowers (first job, high debt), a model that shows age inversely related to default should be investigated.

5. **Are there fairness concerns?** Using age directly in credit decisions may violate fair lending regulations in some jurisdictions.

```python
# Use SHAP for directional and local interpretation
import shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```
</details>

---

**10.** You have trained three models to predict house prices. Summarise which model to choose and why:

| Model | CV RMSE | Train RMSE | # Features | Interpretable? |
|---|---|---|---|---|
| Linear Regression | 28,500 | 27,800 | 5 | Yes |
| Ridge (tuned) | 26,200 | 25,900 | 5 | Yes |
| Random Forest | 24,100 | 12,300 | 5 | No |

<details>
<summary>Answer</summary>

**Depends on the use case:**

**For prediction accuracy:** Random Forest (CV RMSE 24,100) — but notice the large train–test gap (12,300 vs 24,100), suggesting it may overfit. Worth running a nested CV to confirm.

**For interpretability + accuracy balance:** Ridge Regression (CV RMSE 26,200) — only 8% worse than Random Forest on CV, but fully interpretable. If stakeholders need to understand *why* a price was predicted, or if model decisions need to be explained (e.g., for appraisal disputes), Ridge is defensible.

**When NOT to choose:**
- Linear Regression (CV RMSE 28,500) underperforms Ridge without offering meaningful benefits in this case
- Random Forest if the CV score is unreliable due to the train/test gap — need to investigate

**The key signal:** Random Forest's train RMSE (12,300) vs CV RMSE (24,100) is a 2× gap — classic overfitting, even with 5 features. Consider max_depth, min_samples_leaf tuning before committing to it.

**General rule:** When two models are within ~10% on CV performance, prefer the simpler/more interpretable one.
</details>

---

## Scoring

| Part | Questions | Points each | Total |
|---|---|---|---|
| A: Logistic Regression | 1–3 | 12 | 36 |
| B: Overfitting & Model Selection | 4–6 | 10 | 30 |
| C: Regularization | 7–8 | 10 | 20 |
| D: Model Interpretation | 9–10 | 7 | 14 |
| **Total** | | | **100** |

## Back to module

[← Back to 4.4 Statistical Modelling](../README.md)
