---
reading_minutes: 14
objectives:
  - "Plot training and validation score against a **single hyperparameter** with `sklearn.model_selection.validation_curve`."
  - "Read the U-shape: validation score peaks at the sweet-spot complexity; left of the peak is underfit, right is overfit."
  - "Use the curve to bracket a search range for grid / randomised search rather than as a substitute for full hyperparameter tuning."
  - "Avoid the gotchas: too few CV folds (jagged curve), tuning only one hyperparameter when interactions matter, and confusing validation with **learning** curves."
---

# Validation Curves

**After this lesson:** you can explain the core ideas in “Validation Curves” and reproduce the examples here in your own notebook or environment.

## Overview

**Validation curves** for a single hyperparameter: where the train/CV gap blows up (overfitting onset).


## Introduction

Validation curves are essential tools in machine learning for understanding how a model's performance changes with different hyperparameter values. They help us find the optimal hyperparameter settings and diagnose issues like overfitting and underfitting.

## What are Validation Curves?

Validation curves plot the model's performance (typically error or accuracy) against different values of a hyperparameter. They show:

1. Training score
2. Validation score
3. The relationship between them

{% include model-eval-html-diagram.html diagram="validation-curves" title="Validation curve diagnosis diagram" %}

> **Read the diagram:** move from left to right as the hyperparameter makes the model more flexible. The best region is not where the training score is highest; it is where the validation score is highest and the train-validation gap is still small.

## Types of Validation Curves

### 1. Model Complexity

#### `validation_curve` for tree depth

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)

# Calculate validation curves
param_range = np.arange(1, 11)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42), X, y,
    param_name="max_depth", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, val_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Validation Curves (Model Complexity)')
plt.legend(loc='best')
plt.grid(True)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-15" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sweep Max Depth</span>
    </div>
    <div class="code-callout__body">
      <p><code>validation_curve</code> fits 5 CV folds at each of 10 depth values; the output matrices (n_depths × n_folds) capture how score varies with complexity and split randomness.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-34" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Aggregate and Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Take mean and std across folds (<code>axis=1</code>), then plot both curves with <code>fill_between</code> bands; a widening gap between training and CV scores signals the onset of overfitting.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/validation-curves_fig_1.png" alt="validation-curves" />
<figcaption>Figure 1: Validation Curves (Model Complexity)</figcaption>
</figure>

> **Read Figure 1:** the x-axis is tree depth. Shallow trees underfit because both curves are lower. As depth increases, the training score keeps rising, but the cross-validation score eventually flattens or drops. Choose a depth near the validation-score peak, before the gap becomes large.

### 2. Regularization Strength

#### Logistic `C` on a log scale

> This example reuses `X, y` (and the imported `np`/`plt`) from the first block above.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.linear_model import LogisticRegression

# Calculate validation curves
param_range = np.logspace(-4, 4, 9)
train_scores, val_scores = validation_curve(
    LogisticRegression(random_state=42), X, y,
    param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, label='Training score')
plt.semilogx(param_range, val_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('C (Inverse Regularization Strength)')
plt.ylabel('Score')
plt.title('Validation Curves (Regularization)')
plt.legend(loc='best')
plt.grid(True)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Log-scale C Sweep</span>
    </div>
    <div class="code-callout__body">
      <p><code>logspace(-4, 4, 9)</code> generates nine values from 0.0001 to 10000; small <code>C</code> applies strong L2 regularization while large <code>C</code> approaches an unregularized fit.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-28" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Semilog Plot</span>
    </div>
    <div class="code-callout__body">
      <p><code>semilogx</code> places the log-spaced <code>C</code> values evenly on the x-axis; the convergence of train and CV scores in the middle shows where regularization stops hurting and starts helping.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/validation-curves_fig_2.png" alt="validation-curves" />
<figcaption>Figure 2: Validation Curves (Regularization)</figcaption>
</figure>

> **Read Figure 2:** `C` is inverse regularization strength. Very small `C` means heavy regularization and can underfit. Very large `C` means weak regularization and can overfit. The useful region is the middle plateau where validation performance is strong and stable.

### 3. Learning Rate

#### Gradient boosting `learning_rate`

> This example reuses `X, y` (and the imported `np`/`plt`) from the first block above.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.ensemble import GradientBoostingClassifier

# Calculate validation curves
param_range = np.logspace(-3, 0, 10)
train_scores, val_scores = validation_curve(
    GradientBoostingClassifier(random_state=42), X, y,
    param_name="learning_rate", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, label='Training score')
plt.semilogx(param_range, val_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Learning Rate')
plt.ylabel('Score')
plt.title('Validation Curves (Learning Rate)')
plt.legend(loc='best')
plt.grid(True)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Learning Rate Range</span>
    </div>
    <div class="code-callout__body">
      <p>Sweep learning rate from 0.001 to 1.0 on a log scale; a very low rate needs more trees to converge while a very high rate can overfit with the default number of estimators.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-28" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Gap Analysis</span>
    </div>
    <div class="code-callout__body">
      <p>The same semilog plot pattern as the regularization example; a large train-CV gap at high learning rates identifies the overfitting regime for gradient boosting.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/validation-curves_fig_3.png" alt="validation-curves" />
<figcaption>Figure 3: Validation Curves (Learning Rate)</figcaption>
</figure>

> **Read Figure 3:** the learning rate controls how aggressively boosting corrects previous mistakes. A tiny value may learn too slowly for the fixed number of estimators. A large value can fit training data too sharply. Prefer the range where the validation curve peaks before the training curve separates.

## Interpreting Validation Curves

### 1. Overfitting

- Training score increases
- Validation score decreases
- Large gap between curves
- Need more regularization

### 2. Underfitting

- Both scores are low
- Small gap between curves
- Need more complexity
- More features might help

### 3. Good Fit

- Both scores are high
- Small gap between curves
- Optimal parameter found
- Model is well-tuned

## Best Practices

1. **Choose Appropriate Range**
   - Wide enough to see trends
   - Fine enough for precision
   - Log scale when needed

2. **Use Cross-Validation**
   - Multiple folds
   - Stratified sampling
   - Appropriate metrics

3. **Plot Confidence Intervals**
   - Show standard deviation
   - Multiple runs
   - Clear visualization

4. **Consider Multiple Parameters**
   - Grid search
   - Random search
   - Bayesian optimization

## Common Mistakes to Avoid

1. **Insufficient Range**
   - Too narrow
   - Missing optimal point
   - Wrong conclusions

2. **Poor Cross-Validation**
   - Not enough folds
   - Data leakage
   - Inappropriate metrics

3. **Misinterpretation**
   - Ignoring variance
   - Overlooking trends
   - Wrong conclusions

## Practical Example: Credit Risk Prediction

Let's analyze validation curves for a credit risk prediction model:

#### Pipeline + `classifier__max_depth` sweep

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

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
    ('classifier', RandomForestClassifier(random_state=42))
])

# Calculate validation curves
param_range = np.arange(1, 21)
train_scores, val_scores = validation_curve(
    pipeline, X, y,
    param_name="classifier__max_depth", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, val_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Validation Curves for Credit Risk Prediction')
plt.legend(loc='best')
plt.grid(True)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-29" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Credit Dataset and Pipeline</span>
    </div>
    <div class="code-callout__body">
      <p>Generate the synthetic credit dataset and wrap a scaler+forest in a <code>Pipeline</code>; the pipeline object is passed directly to <code>validation_curve</code> so preprocessing runs correctly inside each fold.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="31-36" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Nested Parameter Name</span>
    </div>
    <div class="code-callout__body">
      <p>Use <code>classifier__max_depth</code> (double underscore) to reach through the pipeline and set the forest's depth; this pattern works for any nested step parameter in sklearn pipelines.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="38-57" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Plot and Interpret</span>
    </div>
    <div class="code-callout__body">
      <p>Plot mean ± std bands across depths 1–20; the depth where CV score peaks and the train-CV gap starts growing is the recommended operating depth for this credit model.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/validation-curves_fig_4.png" alt="validation-curves" />
<figcaption>Figure 4: Validation Curves for Credit Risk Prediction</figcaption>
</figure>

> **Read Figure 4:** this is the same depth sweep in a business-style credit-risk pipeline. If a deeper forest gives almost no validation gain but increases the training-validation gap, the extra depth is complexity without reliable generalization. In a real credit setting, prefer the simpler depth with nearly the same validation score.

## Gotchas

- **Misreading the y-axis direction** — `validation_curve` returns scores (higher is better), not errors; a rising training curve that diverges from a flat validation curve signals overfitting, but learners who expect an error plot may reverse their interpretation and increase complexity when they should reduce it.
- **Sweeping a linear range for parameters that need a log scale** — Regularization parameters like `C` in logistic regression span orders of magnitude; a linear sweep from 1 to 10 misses the critical low-`C` region where the model underfits; always use `np.logspace` for parameters whose effect is multiplicative.
- **Confusing the validation curve peak with the final model** — The hyperparameter value where the CV score peaks on a validation curve was selected by looking at validation data; that score is optimistic; use a separate test set or nested CV to get an unbiased performance estimate for the chosen setting.
- **Passing the full dataset to `validation_curve` and then also evaluating on a held-out test set drawn from the same data** — `validation_curve` uses internal cross-validation on whatever `X, y` you pass; if `X` already excludes your test split, this is fine, but passing all data and later claiming a separate test set breaks the independence requirement.
- **Drawing conclusions from noisy `fill_between` bands** — Wide standard-deviation bands across folds indicate the validation curve estimate is unreliable (often due to small datasets); increasing `cv` folds or dataset size before reading the curve will give cleaner, more actionable results.
- **Forgetting to use the `classifier__` prefix for pipeline parameters** — When the estimator is wrapped in a `Pipeline`, the `param_name` argument must use the double-underscore notation (e.g., `"classifier__max_depth"`); omitting the step prefix raises a `ValueError` that is easy to misread as a data problem.

## Additional Resources

1. [Scikit-learn: validation curve user guide](https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve)
2. [Scikit-learn: `validation_curve` API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html)
3. [Scikit-learn example: plotting validation curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html)
4. [Scikit-learn: tuning estimator hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html)
