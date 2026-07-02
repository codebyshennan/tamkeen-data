---
reading_minutes: 14
objectives:
  - "Plot training and validation score vs **training-set size** with `sklearn.model_selection.learning_curve`."
  - "Read the curves: both flat and high-error = bias; large gap = variance; both still climbing = more data will help."
  - "Decide the next move from the shape: more data, more model capacity, more regularisation, or stop training."
  - "Avoid the gotchas: noisy curves on tiny datasets, mismatched CV strategy, and confusing learning curves with **validation** curves (capacity-vs-error)."
---

# Learning Curves

**After this lesson:** you can explain the core ideas in “Learning Curves” and reproduce the examples here in your own notebook or environment.

## Overview

**Learning curves**: training vs validation error vs sample size—diagnosing bias, variance, and data needs.


## Introduction

Learning curves are powerful tools for diagnosing model performance and understanding how our model learns from data. They help us identify issues like overfitting and underfitting, and guide us in making better decisions about model complexity and data requirements.

## What are Learning Curves?

Learning curves plot the model's performance (e.g., accuracy or error) against the amount of training data. They show us how the model's performance changes as we add more training examples.

### Why Learning Curves Matter

1. Diagnose model performance issues
2. Determine if more data would help
3. Identify overfitting or underfitting
4. Guide model selection and tuning

## Real-World Analogies

### The Student Learning Analogy

Think of learning curves like a student's progress:

- Training curve: How well the student performs on practice problems
- Validation curve: How well the student performs on new problems
- Gap between curves: How well the student generalizes

### The Sports Training Analogy

Learning curves are like sports training:

- Training curve: Performance in practice
- Validation curve: Performance in games
- Gap between curves: Ability to apply skills in real situations

## Understanding Learning Curves

{% include model-eval-html-diagram.html diagram="learning-curves" title="Learning curve diagnosis diagram" %}

> **Read the diagram:** the x-axis is training-set size, not model complexity. Read the final right-hand side first: if both curves end low, the model is underfitting; if the training curve stays much higher than validation, it is overfitting; if both end high and close together, the model is generalizing.

### 1. Ideal Learning Curve

#### `learning_curve` with logistic regression

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=15, n_redundant=5,
                           random_state=42)

# Calculate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(),
    X, y,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
final_gap = train_mean[-1] - val_mean[-1]
plt.scatter([train_sizes[-1]], [val_mean[-1]], color='green', s=80, zorder=5,
            label='Final CV score')
plt.annotate(f'final gap ≈ {final_gap:.2f}', xy=(train_sizes[-1], val_mean[-1]),
             xytext=(train_sizes[-4], val_mean[-1] - 0.08),
             arrowprops=dict(arrowstyle='->', color='green'), color='darkgreen')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Generate a 1000-sample binary classification problem; <code>learning_curve</code> will subsample this at 10 increasing fractions from 10% to 100%.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-26" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compute Curves</span>
    </div>
    <div class="code-callout__body">
      <p><code>learning_curve</code> returns score arrays shaped (train_size, cv_folds); taking <code>mean(axis=1)</code> and <code>std(axis=1)</code> collapses folds into a single mean and spread per size.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-39" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Plot with Confidence Bands</span>
    </div>
    <div class="code-callout__body">
      <p><code>fill_between</code> adds a ±1 std band around each curve; converging curves with a narrow gap indicate a well-generalizing model.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/learning-curves_fig_1.png" alt="learning-curves" />
<figcaption>Figure 1: Learning Curves</figcaption>
</figure>

> **Read Figure 1:** the training score usually starts high because small training sets are easy to memorize. The validation score should rise as more examples are added. A useful curve ends with the two lines close together at a strong score, which means the model has enough data to generalize.

### 2. Overfitting Learning Curve

#### Larger MLP (typical gap)

> This example reuses `X, y` (and the imported `np`/`plt`) from the first block above.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.neural_network import MLPClassifier

# Calculate learning curves for a complex model
train_sizes, train_scores, val_scores = learning_curve(
    MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    X, y,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot overfitting learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.annotate('large generalization gap', xy=(train_sizes[-1], val_mean[-1]),
             xytext=(train_sizes[-5], train_mean[-1] - 0.06),
             arrowprops=dict(arrowstyle='<->', color='red'), color='darkred')
plt.scatter([train_sizes[-1]], [train_mean[-1]], color='red', s=60, zorder=5)
plt.scatter([train_sizes[-1]], [val_mean[-1]], color='red', s=60, zorder=5)
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Overfitting Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">High-capacity Model</span>
    </div>
    <div class="code-callout__body">
      <p>A two-hidden-layer MLP (100, 50 neurons) is more flexible than logistic regression; its training score typically stays high while validation lags, revealing overfitting.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-16" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compute Mean and Std</span>
    </div>
    <div class="code-callout__body">
      <p>Same aggregation as the ideal-fit example — collapse CV fold scores into per-size mean and standard deviation for plotting.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="18-28" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Overfitting Diagnostic</span>
    </div>
    <div class="code-callout__body">
      <p>A large visible gap between the training and validation bands is the visual signature of overfitting — the model memorizes training patterns rather than generalizing.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/learning-curves_fig_2.png" alt="learning-curves" />
<figcaption>Figure 2: Overfitting Learning Curves</figcaption>
</figure>

> **Read Figure 2:** the training curve remaining high while the validation curve stays lower is the key warning sign. More data may help if the validation curve is still climbing at the right edge, but regularization or a simpler model is usually the faster first response.

### 3. Underfitting Learning Curve

#### Dummy baseline (high bias)

> This example reuses `X, y` (and the imported `np`/`plt`) from the first block above.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.dummy import DummyClassifier

# Calculate learning curves for a simple model
train_sizes, train_scores, val_scores = learning_curve(
    DummyClassifier(),
    X, y,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot underfitting learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plateau = (train_mean[-1] + val_mean[-1]) / 2
plt.axhline(plateau, color='red', linestyle='--', linewidth=2,
            label='Low plateau')
plt.annotate('both curves stuck low', xy=(train_sizes[-1], plateau),
             xytext=(train_sizes[-5], plateau + 0.08),
             arrowprops=dict(arrowstyle='->', color='red'), color='darkred')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Underfitting Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Dummy Baseline</span>
    </div>
    <div class="code-callout__body">
      <p><code>DummyClassifier</code> predicts the majority class regardless of input — a worst-case underfitter whose plateau score equals the class frequency.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-16" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Aggregate Scores</span>
    </div>
    <div class="code-callout__body">
      <p>Mean and std across folds collapse the raw score matrix to per-size statistics, consistent with the previous two examples.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="18-28" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Underfitting Diagnostic</span>
    </div>
    <div class="code-callout__body">
      <p>Both curves plateau at a low, flat score with a small gap — the characteristic shape of underfitting where more data provides no improvement.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/learning-curves_fig_3.png" alt="learning-curves" />
<figcaption>Figure 3: Underfitting Learning Curves</figcaption>
</figure>

> **Read Figure 3:** both curves are close together, but that is not automatically good. Because they plateau at a weak score, the model is too simple to learn the pattern. Adding more rows will not fix the main problem; add better features or use a stronger model.

## Interpreting Learning Curves

### 1. High Bias (Underfitting)

- Both curves plateau at low performance
- Small gap between curves
- More data won't help much

### 2. High Variance (Overfitting)

- Training curve much higher than validation curve
- Large gap between curves
- More data might help

### 3. Good Fit

- Both curves plateau at high performance
- Small gap between curves
- Model generalizes well

## Best Practices

1. **Data Preparation**
   - Use training sizes that cover both small-data and near-full-data regimes; the left side shows how quickly the model learns, while the right side shows whether more data is still likely to help.
   - Put preprocessing inside the cross-validation pipeline so each fold learns scaling, imputation, and encoding from its own training subset only.
   - Investigate extreme outliers before plotting; a few corrupted rows can make the early training-size points look unstable and lead to the wrong diagnosis.

2. **Model Selection**
   - Start with a simple baseline because its curve tells you whether the dataset is learnable before adding model complexity.
   - Increase complexity only when the simple model plateaus at low train and validation scores; if the simple model already has a high validation plateau, extra complexity mostly adds variance risk.
   - Use cross-validation curves rather than one split so the final gap reflects typical behaviour across folds.

3. **Regularization**
   - If the training curve stays high but the validation curve remains much lower, add regularisation or simplify the model; the gap is evidence that the model is memorising patterns that do not transfer.
   - Tune regularisation with validation curves after reading the learning curve; the learning curve tells you whether regularisation is the right lever.
   - Watch the validation curve, not only training score: stronger regularisation is useful only if validation performance improves or becomes more stable.

4. **Monitoring**
   - Track train and validation scores together because each score alone is ambiguous: low validation score could mean underfitting, overfitting, or a noisy split.
   - Replot the curve after major feature-engineering or modelling changes so the diagnosis stays current.
   - Use early stopping when validation performance stops improving while training performance continues to rise; this is the point where additional training begins to buy memorisation rather than generalisation.

## Common Mistakes to Avoid

1. **Overfitting**
   - Using too complex models
   - Not using validation sets
   - Ignoring regularization

2. **Underfitting**
   - Using too simple models
   - Not considering feature engineering
   - Insufficient training time

## Gotchas

- **Confusing learning curves with validation curves** — Learning curves vary *training set size* on the x-axis; validation curves vary a *hyperparameter* on the x-axis; mixing them up leads to wrong diagnoses (e.g., concluding "more data won't help" when you're actually looking at a validation curve showing overfitting onset).
- **Interpreting a converging gap as always meaning "good fit"** — Train and validation curves that converge at a *low* value both indicate underfitting, not a good model; a converging gap only confirms a good fit when the convergence point is also *high* (close to your target performance).
- **Using too few or too many training size points** — With `train_sizes=np.linspace(0.1, 1.0, 5)` you get only 5 data points and miss the curve's shape; with 50 points the computation time multiplies; 8–15 points (the default 5 in sklearn is often too few) balances resolution and cost.
- **Not shuffling before calling `learning_curve`** — If data is sorted by class or time, small training subsets may contain only one class, causing artificially low scores at the left end of the curve; pass `shuffle=True` (or use a pre-shuffled dataset) to get representative subsamples at each size.
- **Assuming more data always closes an overfitting gap** — For a high-capacity model like an unregularised deep tree, adding data eventually helps, but the convergence may require far more samples than you have; if the gap is still wide at 100% of your data, regularisation or a simpler model is the right lever, not more data collection.
- **Drawing the curve with training set size in samples vs fractions** — `learning_curve` returns raw sample counts in `train_sizes`; plotting fractions (0 to 1) without dividing by `n_samples` compresses the x-axis and makes it hard to know whether you need 500 or 5000 additional examples to close the gap.

## Additional Resources

1. [Scikit-learn: learning curve user guide](https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve)
2. [Scikit-learn: `learning_curve` API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)
3. [Scikit-learn example: plotting learning curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
4. [Scikit-learn: underfitting vs overfitting example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
