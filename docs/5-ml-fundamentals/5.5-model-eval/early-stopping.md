---
reading_minutes: 15
objectives:
  - "Explain **early stopping** as monitoring validation loss during training and halting when it stops improving for `patience` steps."
  - "Wire it up in scikit-learn (`MLPClassifier`, `HistGradientBoosting`, `SGDClassifier` with `early_stopping=True`), gradient-boosting libraries, and Keras callbacks."
  - "Tune `patience` and the validation-set fraction so noisy curves don't trigger termination, and a clear plateau still does."
  - "Pair early stopping with `restore_best_weights` (or equivalent) so the kept model is the one with lowest validation loss, not the one at the stop step."
---

# Early Stopping

**After this lesson:** you can explain the core ideas in “Early Stopping” and reproduce the examples here in your own notebook or environment.

## Overview

**Early stopping** as regularization: monitoring validation loss/score and halting before overfitting.


## Introduction

Early stopping is a regularization technique that helps prevent overfitting by monitoring the model's performance on a validation set and stopping training when performance starts to degrade.

## What is Early Stopping?

Early stopping works by monitoring the model's performance on a validation set during training. When the performance stops improving or starts to degrade, training is stopped to prevent overfitting.

{% include model-eval-html-diagram.html diagram="early-stopping" title="Early stopping loop diagram" %}

*`patience` controls how many epochs of no-improvement you tolerate before stopping. Typical values: 5–20 for neural networks, 10–50 for gradient boosting.*

### Why Early Stopping Matters

1. Prevents overfitting
2. Saves computational resources
3. Automates model training
4. Improves model generalization

## Real-World Analogies

### The Student Study Analogy

Think of early stopping like studying for an exam:

- Training: Studying the material
- Validation: Taking practice tests
- Early stopping: Stopping when practice test scores start to decline

### The Sports Training Analogy

Early stopping is like sports training:

- Training: Practicing skills
- Validation: Performance in practice games
- Early stopping: Stopping when performance plateaus

## Implementation

### 1. Basic Early Stopping

#### Manual `partial_fit` loop with patience

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize model (warm_start required for repeated partial_fit)
model = MLPClassifier(
    hidden_layer_sizes=(100, 50), max_iter=1, warm_start=True, random_state=42
)

# Train with early stopping
best_val_score = 0
patience = 5
no_improvement = 0

for epoch in range(1000):
    model.partial_fit(X_train, y_train, classes=np.unique(y))
    val_score = model.score(X_val, y_val)

    if val_score > best_val_score:
        best_val_score = val_score
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-15" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Setup and Model Init</span>
    </div>
    <div class="code-callout__body">
      <p>Create a three-way split (train/val/test); <code>warm_start=True</code> and <code>max_iter=1</code> let <code>partial_fit</code> advance training one epoch at a time without resetting weights.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-35" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Patience Loop</span>
    </div>
    <div class="code-callout__body">
      <p>Each iteration calls <code>partial_fit</code> then scores the validation set; the counter resets on any improvement and triggers a break when it reaches the <code>patience</code> limit — the core early-stopping mechanic.</p>
    </div>
  </div>
</aside>
</div>

```
Early stopping at epoch 11
```

### 2. Using Scikit-learn's Early Stopping

#### `SGDClassifier(early_stopping=True)`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline with early stopping
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SGDClassifier(early_stopping=True, validation_fraction=0.2))
])

# Fit and evaluate
pipeline.fit(X_train, y_train)
print(f"Early Stopping Score: {pipeline.score(X_test, y_test):.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and Split</span>
    </div>
    <div class="code-callout__body">
      <p>Generate a 1000-sample classification dataset and split 80/20; the 80% training set is then further split internally by <code>SGDClassifier</code> using <code>validation_fraction</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Built-in Early Stopping</span>
    </div>
    <div class="code-callout__body">
      <p><code>SGDClassifier(early_stopping=True, validation_fraction=0.2)</code> automatically reserves 20% of train data for validation and halts when score stops improving — no manual loop needed.</p>
    </div>
  </div>
</aside>
</div>

```
Early Stopping Score: 0.815
```

### 3. Custom Early Stopping Class

#### Callable tracker object

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(100, 50), max_iter=1, warm_start=True, random_state=42
)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.should_stop

# Use custom early stopping
early_stopping = EarlyStopping(patience=5)
for epoch in range(1000):
    model.partial_fit(X_train, y_train, classes=np.unique(y))
    val_score = model.score(X_val, y_val)

    if early_stopping(val_score):
        print(f"Early stopping at epoch {epoch}")
        break
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-14" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and Model Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Same three-way split and <code>warm_start</code> MLP as the manual loop example; the difference is that the stopping logic is now encapsulated in a callable class rather than inline.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-36" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">EarlyStopping Class</span>
    </div>
    <div class="code-callout__body">
      <p>Define <code>EarlyStopping</code> with <code>patience</code> and <code>min_delta</code>; <code>__call__</code> updates the best score and counter, returning <code>True</code> only when the patience limit is reached.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="38-47" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Training Loop</span>
    </div>
    <div class="code-callout__body">
      <p>Call the instance like a function each epoch; when it returns <code>True</code> the loop breaks — the class can be reused across different models and frameworks with no changes.</p>
    </div>
  </div>
</aside>
</div>

```
Early stopping at epoch 28
```

## Best Practices

1. **Choose Appropriate Metrics**
   - Use validation metrics
   - Consider business objectives
   - Monitor multiple metrics

2. **Set Proper Parameters**
   - Choose appropriate patience
   - Set minimum improvement threshold
   - Consider computational resources

3. **Monitor Training**
   - Track training progress
   - Visualize learning curves
   - Save best model

4. **Validate Results**
   - Test on holdout set
   - Compare with baseline
   - Check for overfitting

## Common Mistakes to Avoid

1. **Too Short Patience**
   - Premature stopping
   - Underfitting
   - Missed improvements

2. **Too Long Patience**
   - Wasted computation
   - Overfitting
   - Poor generalization

3. **Wrong Metrics**
   - Misleading early stopping
   - Poor model selection
   - Inappropriate validation

## Practical Example: Credit Risk Prediction

Let's see how early stopping helps in a credit risk prediction task:

#### Growing `n_estimators` with patience

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Create credit risk dataset
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.normal(35, 10, n_samples)
income = np.random.exponential(50000, n_samples)
credit_score = np.random.normal(700, 100, n_samples)

X = np.column_stack([age, income, credit_score])
y = (credit_score + income/1000 + age > 800).astype(int)  # Binary target

# Create pipeline with early stopping
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train with early stopping
best_score = 0
best_model = None
patience = 5
no_improvement = 0

for n_estimators in range(10, 100, 10):
    pipeline.set_params(classifier__n_estimators=n_estimators)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    if score > best_score:
        best_score = score
        best_model = pipeline
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= patience:
        print(f"Early stopping at {n_estimators} trees")
        break

print(f"Best model score: {best_score:.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-24" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Credit Dataset and Pipeline</span>
    </div>
    <div class="code-callout__body">
      <p>Generate synthetic credit features, derive a binary label, and wrap a RandomForest in a scaler pipeline; the forest's <code>n_estimators</code> will be updated each iteration to simulate an epoch-by-epoch training process.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="30-52" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Tree-count Patience Loop</span>
    </div>
    <div class="code-callout__body">
      <p><code>set_params(classifier__n_estimators=...)</code> grows the forest incrementally; the patience counter triggers early stopping when test score stops improving — pedagogical note: use a validation fold rather than the test set in production.</p>
    </div>
  </div>
</aside>
</div>

```
Early stopping at 90 trees
Best model score: 0.985
```

## Gotchas

- **Monitoring training loss instead of validation loss** — Early stopping only prevents overfitting when triggered by *validation* performance; stopping on training loss can halt before the model has converged because training loss can plateau due to learning rate schedules, not because generalisation has peaked.
- **Setting patience too low and stopping in a temporary dip** — Validation loss often fluctuates between epochs; a patience of 1 or 2 will stop training prematurely during a normal valley that would have recovered; set patience to at least 5–10 epochs and restore the best model weights at the end.
- **Forgetting to restore best weights after stopping** — The model's weights at the stopping epoch are not the best weights — they are the weights from `patience` epochs *after* the best; always save the best checkpoint (e.g., `best_model = pipeline`) and use that for inference, not the final state.
- **Using `SGDClassifier(early_stopping=True)` without understanding `validation_fraction`** — This flag causes sklearn to carve out `validation_fraction` (default 0.1) from the training set internally; if your dataset is small, this hidden split can meaningfully reduce effective training size without any warning.
- **Treating early stopping epoch count as a stable hyperparameter** — The number of epochs at which stopping triggers depends on the train/validation split, random seed, and data order; reporting "we stopped at epoch 47" across different splits is not reproducible; the stopping point will differ on every run unless you also fix all random seeds.
- **Applying early stopping to random forests** — Random forests do not train iteratively in the same sense as gradient-based models; "stopping early" by limiting `n_estimators` is valid but is better handled via OOB error or a held-out validation set with a standard grid search, not a patience-based loop that evaluates the test set each iteration.

## Additional Resources

1. Scikit-learn documentation
2. Research papers on early stopping
3. Online tutorials on model training
