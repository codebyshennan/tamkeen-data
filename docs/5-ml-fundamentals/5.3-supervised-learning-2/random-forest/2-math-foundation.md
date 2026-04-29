---
reading_minutes: 20
objectives:
  - "Walk through bootstrap aggregating (bagging) with replacement and explain how it reduces variance compared with a single tree."
  - "Compute Gini impurity / entropy at a split, and understand why limiting `max_features` decorrelates trees."
  - "Read out-of-bag (OOB) error as a free validation estimate, and rank features with Mean Decrease in Impurity vs Permutation Importance."
---

# Mathematical Foundation of Random Forest

**After this lesson:** you can explain the core ideas in “Mathematical Foundation of Random Forest” and reproduce the examples here in your own notebook or environment.

## Overview

Explains **bagging**, random subspaces at each split, and why variance drops when trees are decorrelated.

[Introduction](1-introduction.md); compare with [decision trees in 5.2](../../5.2-supervised-learning-1/decision-trees/1-introduction.md).

## Helpful video

Crash Course AI: supervised learning framing (~15 min).

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qVRBYAdLAo" title="Supervised Learning: Crash Course AI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Bootstrap Aggregating (Bagging)

### What is Bagging?

Imagine you're trying to understand how people feel about a new movie. Instead of asking just one person, you:

1. Randomly select people from the audience
2. Some people might be asked multiple times
3. Each group gives you a different perspective

This is exactly how bagging works in Random Forest!

### Mathematical Definition

For a dataset of size n, we create m new datasets by randomly sampling with replacement. Each data point has about a 63.2% chance of being selected in each sample.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np

def bootstrap_sample(X, y):
    """
    Create a bootstrap sample from the dataset.

    Parameters:
    X: Features
    y: Target variable

    Returns:
    X_sample: Sampled features
    y_sample: Sampled target
    """
    n_samples = X.shape[0]
    # Randomly select indices with replacement
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]
{% endhighlight %}

<figure>
<img src="assets/2-math-foundation_fig_1.png" alt="2-math-foundation" />
<figcaption>Figure 1: Generated visualization</figcaption>
</figure>


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Function Signature</span>
    </div>
    <div class="code-callout__body">
      <p>Takes feature matrix <code>X</code> and target <code>y</code>; the docstring documents inputs and outputs — critical since ~36.8% of samples will be left out (OOB) for each tree.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sample with Replacement</span>
    </div>
    <div class="code-callout__body">
      <p><code>np.random.choice(..., replace=True)</code> selects n indices from 0…n-1 allowing duplicates — this is bootstrapping, the core mechanism that makes each tree in the forest see different training data.</p>
    </div>
  </div>
</aside>
</div>

### Out-of-Bag (OOB) Estimation

Think of this as a built-in validation set. For each tree, some data points weren't used in training - we can use these to estimate how well the model will perform on new data.

## Random Feature Selection

### What is Feature Selection?

Imagine each expert in our committee only looks at certain aspects of a car:

- One expert might focus on safety features
- Another might look at fuel efficiency
- A third might consider price and maintenance costs

This is how Random Forest selects features - each tree only considers a random subset of features when making decisions.

![Feature Importance](assets/feature_importance.png)
*Figure 1: Feature importance shows which features contribute most to the model's predictions.*

### Feature Sampling

At each split in a tree, we only consider a random subset of features:

- For classification: typically $\sqrt{p}$ features
- For regression: typically $p/3$ features
where $p$ is the total number of features.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def get_random_features(n_features, n_select):
    """
    Select a random subset of features.

    Parameters:
    n_features: Total number of features
    n_select: Number of features to select

    Returns:
    selected_features: Indices of selected features
    """
    return np.random.choice(
        n_features,
        size=n_select,
        replace=False
    )
{% endhighlight %}

<figure>
<img src="assets/2-math-foundation_fig_1.png" alt="2-math-foundation" />
<figcaption>Figure 1: Generated visualization</figcaption>
</figure>


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Parameters</span>
    </div>
    <div class="code-callout__body">
      <p>Takes total feature count and how many to keep; the docstring clarifies that the return is feature <em>indices</em>, not values — these indices are then used to slice columns of <code>X</code> at each tree split.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Random Subset</span>
    </div>
    <div class="code-callout__body">
      <p><code>replace=False</code> ensures no feature is picked twice per split; in sklearn this corresponds to <code>max_features='sqrt'</code> (classification) or <code>max_features='log2'</code> by convention.</p>
    </div>
  </div>
</aside>
</div>

## Ensemble Prediction

### Classification

For classification problems, it's like taking a vote among all the experts. The most common prediction wins!

### Regression

For regression problems, it's like taking the average of all expert opinions. This helps balance out individual biases.

![Ensemble Prediction](assets/ensemble_prediction.png)
*Figure 2: How individual tree predictions combine to form the final ensemble prediction.*

## Feature Importance

### What is Feature Importance?

Think of this as understanding which factors matter most in making a decision. For example, in predicting house prices:

- Location might be very important
- Number of bedrooms might be somewhat important
- Color of the walls might not matter much

### Gini Importance

The Gini importance measures how much each feature contributes to reducing uncertainty in the predictions.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def gini_impurity(y):
    """
    Calculate Gini impurity - a measure of how mixed the classes are.

    Parameters:
    y: Target variable

    Returns:
    impurity: Gini impurity score
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)
{% endhighlight %}

<figure>
<img src="assets/2-math-foundation_fig_1.png" alt="2-math-foundation" />
<figcaption>Figure 1: Generated visualization</figcaption>
</figure>


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Function and Docstring</span>
    </div>
    <div class="code-callout__body">
      <p>Takes a 1D label array <code>y</code> and returns a scalar impurity score between 0 (pure node) and 0.5 (maximally mixed for two classes).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Gini Calculation</span>
    </div>
    <div class="code-callout__body">
      <p><code>np.unique</code> with <code>return_counts=True</code> tallies class frequencies; dividing by <code>len(y)</code> gives class probabilities; <code>1 - sum(p²)</code> implements the Gini formula.</p>
    </div>
  </div>
</aside>
</div>

## Error Analysis

### Bias-Variance Tradeoff

Think of this as the balance between:

- **Bias**: How far off our predictions are on average
- **Variance**: How much our predictions vary from one tree to another

Random Forests help reduce variance while maintaining bias, making the model more stable.

![Bias-Variance Tradeoff](assets/bias_variance.png)
*Figure 3: The bias-variance tradeoff in Random Forests - how model complexity affects predictions.*

## Convergence Properties

### Law of Large Numbers

As we add more trees to our forest, the predictions become more stable and reliable. This is like how a larger sample size gives us more confidence in our results.

## Optimization Criteria

### Split Quality

When deciding how to split the data at each node, we look for splits that:

1. Create more homogeneous groups
2. Reduce uncertainty in our predictions

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def information_gain(parent, left, right):
    """
    Calculate how much information we gain from a split.

    Parameters:
    parent: Parent node data
    left: Left child node data
    right: Right child node data

    Returns:
    gain: Information gain from the split
    """
    n = len(parent)
    n_l, n_r = len(left), len(right)

    gain = gini_impurity(parent) - (
        n_l/n * gini_impurity(left) +
        n_r/n * gini_impurity(right)
    )
    return gain
{% endhighlight %}

<figure>
<img src="assets/2-math-foundation_fig_1.png" alt="2-math-foundation" />
<figcaption>Figure 1: Generated visualization</figcaption>
</figure>


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Signature and Docstring</span>
    </div>
    <div class="code-callout__body">
      <p>Takes three label arrays (parent node, left child, right child); information gain is the reduction in Gini impurity from the parent to the weighted-average child impurity.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Weighted Impurity Reduction</span>
    </div>
    <div class="code-callout__body">
      <p>Weights each child's Gini impurity by its fraction of the total samples (<code>n_l/n</code>, <code>n_r/n</code>); a larger gain means this split creates purer children — the tree picks the feature and threshold that maximizes this value.</p>
    </div>
  </div>
</aside>
</div>

## Hyperparameter Effects

### Number of Trees

- More trees = more stable predictions
- But diminishing returns after a certain point
- Think of it like adding more experts to a committee - after a while, adding more doesn't help much

### Max Features

- Fewer features = more diverse trees
- More features = better individual trees
- It's like deciding how many aspects each expert should consider

### Tree Depth

- Deeper trees = more detailed decisions
- Shallower trees = more general decisions
- It's like deciding how many questions each expert can ask

![Decision Tree vs Random Forest](assets/decision_tree_boundary.png)
*Figure 4: A single decision tree (left) makes simple, piecewise linear decisions, while a Random Forest (right) combines multiple trees to create more complex decision boundaries.*

## Gotchas

- **The 63.2% rule only holds for large datasets** — each bootstrap sample contains ~63.2% unique observations when n is large; for small datasets (n < 100) this fraction varies significantly, so OOB estimates become unreliable and a proper validation split is still needed.
- **Gini importance double-counts correlated features** — when two features carry the same information, each will steal splits from the other across trees, making both appear less important than they truly are; this is why permutation importance on a held-out set is preferred.
- **`max_features='sqrt'` is the default for classification but `max_features=1.0` is the default for regression** — forgetting that defaults differ between `RandomForestClassifier` and `RandomForestRegressor` leads to silently different variance-reduction behaviour.
- **Information gain in this implementation uses Gini, not entropy** — sklearn's `RandomForestClassifier` uses Gini impurity by default; switching to `criterion='entropy'` changes split decisions and can produce slightly different trees, but neither is universally better.
- **Adding trees beyond convergence wastes memory without reducing bias** — the Law of Large Numbers guarantees variance converges as tree count grows, but bias is fixed by individual tree depth; no amount of additional trees can fix underfitting caused by shallow `max_depth`.
- **Feature importance scores sum to 1.0 but are not probabilities** — the normalised sum-to-one property is an artefact of the calculation, not a probabilistic statement; a feature with importance 0.4 is not "40% responsible" for predictions.

## Next Steps

Now that you understand the mathematics behind Random Forests, let's move on to [Implementation](3-implementation.md) to see how to put these concepts into practice!
