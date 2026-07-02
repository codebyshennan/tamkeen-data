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

> **Read the diagram:** each loop represents one more epoch or boosting iteration. Training continues only while validation performance is improving often enough. The saved checkpoint is the best validation checkpoint, not necessarily the final epoch before stopping.

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

> **Read the output:** the model saw five consecutive epochs without validation improvement by epoch 11, so the patience rule stopped training early. This is a warning to keep the best validation checkpoint; the final epoch is only the point where patience ran out.

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
    ('classifier', SGDClassifier(early_stopping=True, validation_fraction=0.2, random_state=42))
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
Early Stopping Score: 0.775
```

> **Read the output:** this is the held-out test score after `SGDClassifier` internally reserved part of the training set for early stopping. If the score is weaker than expected, check whether `validation_fraction` left too little data for fitting.

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

> **Read the output:** the custom tracker allowed more training than the first manual loop because the validation-score pattern differed. The important part is not the exact epoch number; it is that the same patience logic can be reused and audited.

## Best Practices

1. **Choose Appropriate Metrics**
   - Stop on validation metrics, not training loss, because the goal is to stop when generalisation stops improving.
   - Choose a metric that matches the business cost. In fraud or credit risk, recall or precision may matter more than accuracy, so stopping on accuracy can freeze the wrong model.
   - Monitor a secondary metric to catch trade-offs; for example, validation AUC may improve while minority-class recall gets worse.

2. **Set Proper Parameters**
   - Set patience long enough to ignore normal validation noise; a patience of 1 can stop before the model recovers from a temporary dip.
   - Set a minimum improvement threshold so tiny random metric changes do not reset the patience counter indefinitely.
   - Tie the maximum number of iterations to the compute budget so training has a hard stop even if the metric keeps fluctuating.

3. **Monitor Training**
   - Track train and validation curves together; if training keeps improving while validation flattens, the stopping point is doing useful regularisation.
   - Visualise the best epoch or estimator count directly on the curve so the selected checkpoint can be audited later.
   - Save the best validation checkpoint, not the last checkpoint, because the final iteration may already be past the best generalisation point.

4. **Validate Results**
   - Evaluate the stopped model once on a holdout test set after tuning patience and thresholds.
   - Compare against a no-early-stopping baseline; early stopping is useful only if it preserves or improves validation/test performance while reducing overfit or compute.
   - Check whether the selected stop point is stable across seeds. If it jumps widely, the validation signal may be too noisy for a confident stopping rule.

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

# Split data: train for fitting, validation for stopping, test for final reporting
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# Monitor the validation set during the tree-count sweep
best_score = 0
best_n_estimators = None
patience = 5
no_improvement = 0

for n_estimators in range(10, 100, 10):
    pipeline.set_params(classifier__n_estimators=n_estimators)
    pipeline.fit(X_train, y_train)
    val_score = pipeline.score(X_val, y_val)

    if val_score > best_score:
        best_score = val_score
        best_n_estimators = n_estimators
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= patience:
        print(f"Early stopping at {n_estimators} trees")
        break

# Refit the selected model on all non-test data, then test once
best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=best_n_estimators, max_depth=10, random_state=42
    ))
])
best_model.fit(X_train_full, y_train_full)
test_score = best_model.score(X_test, y_test)

print(f"Best validation score: {best_score:.3f}")
print(f"Selected n_estimators: {best_n_estimators}")
print(f"Final test score: {test_score:.3f}")
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
  <div class="code-callout" data-lines="30-69" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Tree-count Patience Loop</span>
    </div>
    <div class="code-callout__body">
      <p><code>set_params(classifier__n_estimators=...)</code> grows the forest incrementally; the patience counter watches validation score only, then the selected tree count is refit on all non-test data before one final test evaluation.</p>
    </div>
  </div>
</aside>
</div>

```
Early stopping at 80 trees
Best validation score: 0.985
Selected n_estimators: 30
Final test score: 0.985
```

> **Read the output:** validation score first peaked at 30 trees, then failed to improve for five checked tree counts, so the loop stopped at 80 trees. The final test score is reported only after refitting the selected 30-tree model on all non-test data.

## Gotchas

- **Monitoring training loss instead of validation loss** — Early stopping only prevents overfitting when triggered by *validation* performance; stopping on training loss can halt before the model has converged because training loss can plateau due to learning rate schedules, not because generalisation has peaked.
- **Setting patience too low and stopping in a temporary dip** — Validation loss often fluctuates between epochs; a patience of 1 or 2 will stop training prematurely during a normal valley that would have recovered; set patience to at least 5–10 epochs and restore the best model weights at the end.
- **Forgetting to restore best weights after stopping** — The model's weights at the stopping epoch are not the best weights — they are the weights from `patience` epochs *after* the best; always save the best checkpoint (e.g., `best_model = pipeline`) and use that for inference, not the final state.
- **Using `SGDClassifier(early_stopping=True)` without understanding `validation_fraction`** — This flag causes sklearn to carve out `validation_fraction` (default 0.1) from the training set internally; if your dataset is small, this hidden split can meaningfully reduce effective training size without any warning.
- **Treating early stopping epoch count as a stable hyperparameter** — The number of epochs at which stopping triggers depends on the train/validation split, random seed, and data order; reporting "we stopped at epoch 47" across different splits is not reproducible; the stopping point will differ on every run unless you also fix all random seeds.
- **Applying early stopping to random forests** — Random forests do not train iteratively in the same sense as gradient-based models; "stopping early" by limiting `n_estimators` is valid but is better handled via OOB error or a held-out validation set with a standard grid search, not a patience-based loop that evaluates the test set each iteration.

## Additional Resources

1. [Scikit-learn: early stopping in stochastic gradient descent](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_early_stopping.html)
2. [Scikit-learn: `MLPClassifier` early-stopping parameters](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
3. [Scikit-learn: gradient boosting early stopping example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html)
