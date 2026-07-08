---
reading_minutes: 15
objectives:
  - >-
    Explain **early stopping** as monitoring validation loss during training and
    halting when it stops improving for `patience` steps.
  - >-
    Wire it up in scikit-learn (`MLPClassifier`, `HistGradientBoosting`,
    `SGDClassifier` with `early_stopping=True`), gradient-boosting libraries,
    and Keras callbacks.
  - >-
    Tune `patience` and the validation-set fraction so noisy curves don't
    trigger termination, and a clear plateau still does.
  - >-
    Pair early stopping with `restore_best_weights` (or equivalent) so the kept
    model is the one with lowest validation loss, not the one at the stop step.
---

# Early Stopping

**After this lesson:** you can explain Early Stopping and try the examples in your own notebook.

## Overview

**Early stopping** as regularization: monitoring validation loss/score and halting before overfitting.

## Introduction

Early stopping is a regularization technique that helps prevent overfitting by monitoring the model's performance on a validation set and stopping training when performance starts to degrade.

> **Key idea:** early stopping uses validation performance as a brake. Training can still improve while generalisation gets worse.

## What is Early Stopping?

Early stopping works by monitoring the model's performance on a validation set during training. When the performance stops improving or starts to degrade, training is stopped to prevent overfitting.

> **Highlight:** `patience` controls how many epochs of no-improvement you tolerate before stopping. Typical values: **5-20** for neural networks, **10-50** for gradient boosting.

> **Read the diagram:** each loop represents one more epoch or boosting iteration. Training continues only while validation performance is improving often enough. The saved checkpoint is the best validation checkpoint, not necessarily the final epoch before stopping.

### Why Early Stopping Matters

1. Prevents **overfitting**
2. Saves **computational resources**
3. Automates model training
4. Improves **model generalization**

## Real-World Analogies

### The Student Study Analogy

Think of early stopping like studying for an exam:

* Training: Studying the material
* Validation: Taking practice tests
* Early stopping: Stopping when practice test scores start to decline

### The Sports Training Analogy

Early stopping is like sports training:

* Training: Practicing skills
* Validation: Performance in practice games
* Early stopping: Stopping when performance plateaus

## Implementation

### 1. Basic Early Stopping

#### Manual `partial_fit` loop with patience

Setup and Model Init

Create a three-way split (train/val/test); `warm_start=True` and `max_iter=1` let `partial_fit` advance training one epoch at a time without resetting weights.

Patience Loop

Each iteration calls `partial_fit` then scores the validation set; the counter resets on any improvement and triggers a break when it reaches the `patience` limit, the core early-stopping mechanic.

```
Early stopping at epoch 11
```

> **Read the output:** the model saw five consecutive epochs without validation improvement by epoch 11, so the patience rule stopped training early. This is a warning to keep the best validation checkpoint; the final epoch is only the point where patience ran out.

### 2. Using Scikit-learn's Early Stopping

#### `SGDClassifier(early_stopping=True)`

Data and Split

Generate a 1000-sample classification dataset and split 80/20; the 80% training set is then further split internally by `SGDClassifier` using `validation_fraction`.

Built-in Early Stopping

`SGDClassifier(early_stopping=True, validation_fraction=0.2)` automatically reserves 20% of train data for validation and halts when score stops improving, no manual loop needed.

```
Early Stopping Score: 0.775
```

> **Read the output:** this is the held-out test score after `SGDClassifier` internally reserved part of the training set for early stopping. If the score is weaker than expected, check whether `validation_fraction` left too little data for fitting.

### 3. Custom Early Stopping Class

#### Callable tracker object

Data and Model Setup

Same three-way split and `warm_start` MLP as the manual loop example; the difference is that the stopping logic is now encapsulated in a callable class rather than inline.

EarlyStopping Class

Define `EarlyStopping` with `patience` and `min_delta`; `__call__` updates the best score and counter, returning `True` only when the patience limit is reached.

Training Loop

Call the instance like a function each epoch; when it returns `True` the loop breaks, the class can be reused across different models and frameworks with no changes.

```
Early stopping at epoch 28
```

> **Read the output:** the custom tracker allowed more training than the first manual loop because the validation-score pattern differed. The important part is not the exact epoch number; it is that the same patience logic can be reused and audited.

## Best Practices

1. **Choose Appropriate Metrics**
   * Stop on validation metrics, not training loss, because the goal is to stop when generalisation stops improving.
   * Choose a metric that matches the business cost. In fraud or credit risk, recall or precision may matter more than accuracy, so stopping on accuracy can freeze the wrong model.
   * Monitor a secondary metric to catch trade-offs; for example, validation AUC may improve while minority-class recall gets worse.
2. **Set Proper Parameters**
   * Set patience long enough to ignore normal validation noise; a patience of 1 can stop before the model recovers from a temporary dip.
   * Set a minimum improvement threshold so tiny random metric changes do not reset the patience counter indefinitely.
   * Tie the maximum number of iterations to the compute budget so training has a hard stop even if the metric keeps fluctuating.
3. **Monitor Training**
   * Track train and validation curves together; if training keeps improving while validation flattens, the stopping point is doing useful regularisation.
   * Visualise the best epoch or estimator count directly on the curve so the selected checkpoint can be audited later.
   * Save the best validation checkpoint, not the last checkpoint, because the final iteration may already be past the best generalisation point.
4. **Validate Results**
   * Evaluate the stopped model once on a holdout test set after tuning patience and thresholds.
   * Compare against a no-early-stopping baseline; early stopping is useful only if it preserves or improves validation/test performance while reducing overfit or compute.
   * Check whether the selected stop point is stable across seeds. If it jumps widely, the validation signal may be too noisy for a confident stopping rule.

## Common Mistakes to Avoid

1. **Too Short Patience**
   * Premature stopping
   * Underfitting
   * Missed improvements
2. **Too Long Patience**
   * Wasted computation
   * Overfitting
   * Poor generalization
3. **Wrong Metrics**
   * Misleading early stopping
   * Poor model selection
   * Inappropriate validation

## Practical Example: Credit Risk Prediction

Look at how early stopping helps in a credit risk prediction task:

#### Growing `n_estimators` with patience

Credit Dataset and Pipeline

Generate synthetic credit features, derive a binary label, and wrap a RandomForest in a scaler pipeline; the forest's `n_estimators` will be updated each iteration to simulate an epoch-by-epoch training process.

Tree-count Patience Loop

`set_params(classifier__n_estimators=...)` grows the forest incrementally; the patience counter watches validation score only, then the selected tree count is refit on all non-test data before one final test evaluation.

```
Early stopping at 80 trees
Best validation score: 0.985
Selected n_estimators: 30
Final test score: 0.985
```

> **Read the output:** validation score first peaked at 30 trees, then failed to improve for five checked tree counts, so the loop stopped at 80 trees. The final test score is reported only after refitting the selected 30-tree model on all non-test data.

## Gotchas

* **Monitoring training loss instead of validation loss**: Early stopping only prevents overfitting when triggered by _validation_ performance; stopping on training loss can halt before the model has converged because training loss can plateau due to learning rate schedules, not because generalisation has peaked.
* **Setting patience too low and stopping in a temporary dip**: Validation loss often fluctuates between epochs; a patience of 1 or 2 will stop training prematurely during a normal valley that would have recovered; set patience to at least 5-10 epochs and restore the best model weights at the end.
* **Forgetting to restore best weights after stopping**: The model's weights at the stopping epoch are not the best weights, they are the weights from `patience` epochs _after_ the best; always save the best checkpoint (e.g., `best_model = pipeline`) and use that for inference, not the final state.
* **Using `SGDClassifier(early_stopping=True)` without understanding `validation_fraction`**: This flag causes sklearn to carve out `validation_fraction` (default 0.1) from the training set internally; if your dataset is small, this hidden split can meaningfully reduce effective training size without any warning.
* **Treating early stopping epoch count as a stable hyperparameter**: The number of epochs at which stopping triggers depends on the train/validation split, random seed, and data order; reporting "we stopped at epoch 47" across different splits is not reproducible; the stopping point will differ on every run unless you also fix all random seeds.
* **Applying early stopping to random forests**: Random forests do not train iteratively in the same sense as gradient-based models; "stopping early" by limiting `n_estimators` is valid but is better handled via OOB error or a held-out validation set with a standard grid search, not a patience-based loop that evaluates the test set each iteration.

## Additional Resources

1. [Scikit-learn: early stopping in stochastic gradient descent](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_early_stopping.html)
2. [Scikit-learn: `MLPClassifier` early-stopping parameters](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
3. [Scikit-learn: gradient boosting early stopping example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html)
