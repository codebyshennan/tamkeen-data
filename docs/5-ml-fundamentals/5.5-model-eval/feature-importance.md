---
reading_minutes: 12
objectives:
  - "Distinguish three flavours: **tree built-in** (impurity / Gini), **permutation importance**, and model-agnostic **SHAP** / linear coefficients."
  - "Compute `feature_importances_` and `permutation_importance` in scikit-learn, then read the ranking with appropriate caution."
  - "Recognise the pitfalls: tree impurity inflates high-cardinality features, correlated features split the importance, and importance is **not** causal."
  - "Use feature importance to drive iteration (drop, engineer, investigate) rather than to make ground-truth claims about the world."
---

# Feature Importance

**After this lesson:** you can explain the core ideas in “Feature Importance” and reproduce the examples here in your own notebook or environment.

## Overview

Tree importances, permutation importance, and caveats—correlation and baseline comparisons matter.


## Introduction

Feature importance is a crucial concept in machine learning that helps us understand which features contribute most to our model's predictions. This understanding is essential for model interpretability, feature selection, and domain knowledge validation.

## What is Feature Importance?

Feature importance measures how much each feature contributes to the model's predictions. It helps us:

1. Identify the most influential features
2. Remove irrelevant features
3. Understand model behavior
4. Validate domain knowledge

{% include model-eval-html-diagram.html diagram="feature-importance" title="Feature importance method comparison diagram" %}

> **Read the diagram:** each method answers a slightly different question. Tree importance asks how the fitted trees split. Permutation importance asks how much score drops when a feature is broken. SHAP asks how features contributed to individual predictions and then summarizes those effects.

## Types of Feature Importance

### 1. Tree-Based Methods

#### Gini-based `feature_importances_`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=10,
                           n_informative=5, n_redundant=2,
                           random_state=42)

# Train random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
bars = plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices], rotation=45)
top_pos = 0
top_feature = indices[0]
top_importance = importances[indices][0]
plt.axhline(0.05, color='gray', linestyle='--', linewidth=1, label='Low-importance guide')
bars[top_pos].set_color('tab:green')
plt.annotate(f'top driver: Feature {top_feature}', xy=(top_pos, top_importance),
             xytext=(top_pos + 1.0, top_importance - 0.03),
             arrowprops=dict(arrowstyle='->', color='green'), color='darkgreen')
plt.legend()
plt.tight_layout()
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and Model</span>
    </div>
    <div class="code-callout__body">
      <p>Generate a 10-feature dataset with only 5 truly informative features; the Random Forest should rank those 5 higher than the redundant and noise features.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Extract and Sort Importances</span>
    </div>
    <div class="code-callout__body">
      <p><code>feature_importances_</code> gives mean impurity-decrease per feature summed to 1; <code>argsort[::-1]</code> ranks them highest to lowest for a sorted bar chart.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-25" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Bar Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Plot bars in sorted order using original feature-index labels; <code>rotation=45</code> prevents label overlap for 10 features.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/feature-importance_fig_2.png" alt="feature-importance" />
<figcaption>Figure 1: Feature Importances</figcaption>
</figure>

> **Read Figure 1:** the bars are normalized impurity importances from the fitted forest. Taller bars mean the model often used that feature for useful splits, but continuous or high-cardinality features can be favored even when their true signal is not stronger.

### 2. Permutation Importance

#### Model-agnostic drop in score

**Purpose:** Fit the same random forest, shuffle each feature repeatedly, and plot how much the score drops.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

X, y = make_classification(n_samples=1000, n_features=10,
                           n_informative=5, n_redundant=2,
                           random_state=42)
# Hold out a test set: permutation importance must be measured on data the
# model did not train on, otherwise it reflects memorisation, not generalisation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Calculate permutation importance on the held-out test set
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

# Plot results
plt.figure(figsize=(10, 6))
plt.title('Permutation Importances')
plt.boxplot(result.importances.T, tick_labels=[f'Feature {i}' for i in range(X.shape[1])])
top_feature = int(np.argmax(result.importances_mean))
top_drop = result.importances_mean[top_feature]
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.annotate(f'largest score drop: Feature {top_feature}', xy=(top_feature + 1, top_drop),
             xytext=(top_feature + 1.4, top_drop + 0.03),
             arrowprops=dict(arrowstyle='->', color='green'), color='darkgreen')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


<figure>
<img src="assets/feature-importance_fig_1.png" alt="feature-importance" />
<figcaption>Figure 2: Permutation Importances</figcaption>
</figure>

> **Read Figure 2:** each box shows the score drop across repeated shuffles of one feature. A large positive drop means the model relies on that feature for held-out performance. A box near zero means shuffling that feature barely changes predictions.

### 3. SHAP Values

#### TreeExplainer + summary plot

**Purpose:** Illustrate the SHAP workflow; this is marked no-output because it requires the optional `shap` package and a fitted model from the previous example.

```python
# no-output
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Plot summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar")
plt.tight_layout()
plt.show()
```

## Best Practices

1. **Use Multiple Methods**
   - Combine different importance measures
   - Cross-validate results
   - Consider domain knowledge

2. **Handle Correlated Features**
   - Group correlated features
   - Use appropriate methods
   - Consider feature interactions

3. **Validate Results**
   - Use cross-validation
   - Check stability
   - Compare with domain knowledge

4. **Visualize Effectively**
   - Use appropriate plots
   - Show confidence intervals
   - Include feature names

## Common Mistakes to Avoid

1. **Ignoring Feature Correlations**
   - Not considering interactions
   - Missing important relationships
   - Overlooking multicollinearity

2. **Overlooking Scale**
   - Not normalizing features
   - Comparing different scales
   - Misinterpreting results

3. **Poor Visualization**
   - Unclear plots
   - Missing context
   - Inappropriate scales

## Practical Example: Credit Risk Prediction

Let's analyze feature importance in a credit risk prediction task:

#### Pipeline importances + SHAP on the forest

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# no-output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Create credit risk dataset
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.exponential(50000, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples),
    'employment_length': np.random.exponential(5, n_samples)
}

X = pd.DataFrame(data)
y = (X['credit_score'] + X['income']/1000 + X['age'] > 800).astype(int)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit pipeline
pipeline.fit(X, y)

# Get feature importances
importances = pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances in Credit Risk Prediction')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.tight_layout()
plt.show()

# Calculate and plot SHAP values.
# Note: StandardScaler is a no-op for tree ensembles (splits are scale-invariant),
# so it adds nothing here. It also creates a subtle mismatch — the forest was
# trained on *scaled* features, so feeding raw `X` to TreeExplainer explains the
# model in a different space than it sees. SHAP should receive data in the same
# space the model was fit on (e.g. pipeline[:-1].transform(X)). For a pure tree
# model, the cleanest fix is to drop the scaler and explain on the raw features.
explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X)
plt.tight_layout()
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-22" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Synthetic Credit Data</span>
    </div>
    <div class="code-callout__body">
      <p>Five financial features with realistic distributions; the binary label is a threshold on credit score, income, and age — meaning those three should dominate importance.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-32" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline Fit</span>
    </div>
    <div class="code-callout__body">
      <p>Scale then classify in a single pipeline; access the fitted forest through <code>pipeline.named_steps['classifier']</code> to extract importances and build the SHAP explainer.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="34-44" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Ranked Bar Chart</span>
    </div>
    <div class="code-callout__body">
      <p>Extract importances from the classifier step, sort descending, and plot with real column names so stakeholders can read which financial factors drive the model.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="46-52" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">SHAP Summary Plot</span>
    </div>
    <div class="code-callout__body">
      <p><code>TreeExplainer</code> is applied to the inner forest (not the scaler); the summary plot shows both magnitude and direction of each feature's impact on the output.</p>
    </div>
  </div>
</aside>
</div>

## Gotchas

- **Tree-based impurity importance is biased toward high-cardinality features** — `feature_importances_` from Random Forest sums mean impurity decrease over splits; features with many unique values (e.g., a continuous numeric column) get more split opportunities and can appear more important than they truly are; use permutation importance or SHAP for a less biased view.
- **Permutation importance computed on training data is misleading** — Shuffling a feature on the training set measures how much the model *relied* on it during training, not how useful it is for new data; always compute permutation importance on a held-out validation or test set to measure true generalisation contribution.
- **Correlated features split importance between them** — If `income` and `wealth_score` are strongly correlated, the model may use one or the other interchangeably; both features will show lower individual importances than either deserves alone, and removing one may not hurt performance; cluster correlated features before interpreting rankings.
- **SHAP values require the model, not just predictions** — `shap.TreeExplainer` needs the fitted estimator object; if you only serialised `predict` output without saving the model, you cannot compute SHAP values retrospectively; always save the full fitted model, not just predictions.
- **Treating feature importance as a ranking for causal inference** — A high-importance feature in a predictive model tells you the model uses that feature, not that it causally drives the outcome; recommending business actions based on feature importances alone (e.g., "increase credit score to get approved") conflates correlation with causation.
- **Negative permutation importance does not mean the feature hurts** — A small negative value (near zero) for permutation importance usually means the feature adds negligible predictive value and the drop in score is within random noise, not that the feature actively harms the model; check confidence intervals before removing features with slightly negative scores.

## Additional Resources

1. [Scikit-learn: feature importance user guide](https://scikit-learn.org/stable/modules/permutation_importance.html)
2. [Scikit-learn: permutation importance API](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html)
3. [Scikit-learn example: impurity vs permutation importance](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)
4. [SHAP documentation](https://shap.readthedocs.io/)
