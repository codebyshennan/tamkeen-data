---
reading_minutes: 35
objectives:
  - "Apply cost-complexity pruning (`ccp_alpha`) after fitting, choosing alpha from a validation curve."
  - "Use a tree's `feature_importances_` to drop low-signal columns before retraining."
  - "Compare a single tree against `RandomForestClassifier` and `GradientBoostingClassifier` baselines to motivate ensembles."
  - "Handle class imbalance with `class_weight='balanced'` and validate with stratified k-fold cross-validation."
---
# Advanced Decision Tree Techniques

**After this lesson:** you can explain the core ideas in “Advanced Decision Tree Techniques” and reproduce the examples here in your own notebook or environment.

## Overview

**Pruning**, cost-complexity ideas, and ways to curb overfitting when a single tree is still the right interpretable model.

See [implementation](3-implementation.md) for baseline code paths.

## Helpful video

Crash Course AI: supervised learning for classical algorithms.

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qVRBYAdLAo" title="Supervised Learning: Crash Course AI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Understanding Tree Pruning

Think of pruning like trimming a tree in your garden. You remove unnecessary branches to keep the tree healthy and manageable.

### Why Prune Trees?

1. **Prevent Overfitting**
   - Like removing unnecessary details from a story
   - Keeps the model from memorizing the training data
   - Makes the model more generalizable

2. **Improve Performance**
   - Faster predictions
   - Less memory usage
   - Clearer decision rules

### Types of Pruning

#### 1. Pre-pruning (Early Stopping)

This is like setting rules before the tree starts growing:

##### Pre-pruning hyperparameters on Iris

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a tree with strict growth rules
tree = DecisionTreeClassifier(
    max_depth=3,              # Don't grow too deep
    min_samples_split=10,     # Need enough samples to split
    min_samples_leaf=5,       # Each leaf needs enough samples
    max_features='sqrt',      # Consider subset of features
    min_impurity_decrease=0.01  # Only split if it helps enough
)

# Fit the tree to our data
tree.fit(X, y)

# Evaluate performance with cross-validation
scores = cross_val_score(tree, X, y, cv=5)
print(f"Average accuracy: {scores.mean():.3f}")

# Visualize the tree
plt.figure(figsize=(15, 10))
plot_tree(
    tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.title('Pre-pruned Decision Tree')
plt.show()
{% endhighlight %}

<figure>
<img src="assets/4-advanced_fig_1.png" alt="4-advanced" />
<figcaption>Figure 1: Pre-pruned Decision Tree</figcaption>
</figure>

```
Average accuracy: 0.940
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports and Data</span>
    </div>
    <div class="code-callout__body">
      <p>Iris is a clean 150-sample dataset; it gives reproducible results to demonstrate the effect of each pre-pruning hyperparameter.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Five Pruning Controls</span>
    </div>
    <div class="code-callout__body">
      <p>All five constraints fire together: depth cap, minimum split size, minimum leaf size, feature subsampling, and minimum impurity gain per split.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-26" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit and Evaluate</span>
    </div>
    <div class="code-callout__body">
      <p>5-fold CV on the full dataset gives an honest accuracy estimate without a separate test split for this small demo.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-38" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Visualize Tree</span>
    </div>
    <div class="code-callout__body">
      <p><code>plot_tree</code> renders the pruned result so you can visually confirm the tree is compact compared to an unconstrained version.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/4-advanced_fig_1.png" alt="4-advanced" />
<figcaption>Figure 1: Pre-pruned Decision Tree</figcaption>
</figure>

```
Average accuracy: 0.940
```

Pre-pruning is a preventative approach where we set limits before training the tree. This prevents the tree from growing too complex in the first place. The parameters used above control different aspects of tree complexity:

- <code>max_depth</code>: Limits how deep the tree can grow
- <code>min_samples_split</code>: Requires a minimum number of samples to split a node
- <code>min_samples_leaf</code>: Ensures each leaf node has enough samples
- <code>max_features</code>: Limits how many features to consider at each split
- <code>min_impurity_decrease</code>: Only allows splits that improve purity by a certain amount

#### 2. Post-pruning (Cost-Complexity Pruning)

This is like trimming the tree after it's grown:

##### Cost-complexity pruning path (`ccp_alpha`)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load a dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create initial tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Get pruning path
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Remove the last element (it's usually too high)
ccp_alphas = ccp_alphas[:-1]

# Try different pruning levels
trees = []
train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    # Create a tree with this pruning parameter
    tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    tree.fit(X_train, y_train)

    # Save the tree and scores
    trees.append(tree)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# Plot accuracy vs alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='Train')
plt.plot(ccp_alphas, test_scores, marker='o', label='Test')
plt.xlabel('Pruning Strength (alpha)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Pruning Strength')

# Plot tree size vs alpha
plt.figure(figsize=(10, 6))
node_counts = [tree.tree_.node_count for tree in trees]
plt.plot(ccp_alphas, node_counts, marker='o')
plt.xlabel('Pruning Strength (alpha)')
plt.ylabel('Number of Nodes')
plt.title('Tree Size vs Pruning Strength')
plt.show()

# Find the best alpha
best_alpha_idx = np.argmax(test_scores)
best_alpha = ccp_alphas[best_alpha_idx]
best_tree = trees[best_alpha_idx]

print(f"Best pruning parameter: {best_alpha:.6f}")
print(f"Training accuracy: {train_scores[best_alpha_idx]:.3f}")
print(f"Testing accuracy: {test_scores[best_alpha_idx]:.3f}")
print(f"Tree size: {node_counts[best_alpha_idx]} nodes")
{% endhighlight %}

<figure>
<img src="assets/4-advanced_fig_2.png" alt="4-advanced" />
<figcaption>Figure 2: Accuracy vs Pruning Strength</figcaption>
</figure>


<figure>
<img src="assets/4-advanced_fig_3.png" alt="4-advanced" />
<figcaption>Figure 3: Tree Size vs Pruning Strength</figcaption>
</figure>

```
Best pruning parameter: 0.004915
Training accuracy: 0.990
Testing accuracy: 0.965
Tree size: 19 nodes
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-24" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Get Pruning Path</span>
    </div>
    <div class="code-callout__body">
      <p><code>cost_complexity_pruning_path</code> returns the sequence of <code>ccp_alphas</code> at which subtrees are removed; removing the last avoids a trivially pruned root.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="26-40" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sweep Alpha Values</span>
    </div>
    <div class="code-callout__body">
      <p>A new tree is refit at each alpha; higher alpha prunes more aggressively, trading training accuracy for smaller tree size and (usually) better generalization.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="42-56" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Dual Plots</span>
    </div>
    <div class="code-callout__body">
      <p>Accuracy vs alpha shows the sweet spot; node count vs alpha shows how aggressively the tree shrinks as pruning increases.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="58-66" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Best Alpha Summary</span>
    </div>
    <div class="code-callout__body">
      <p><code>argmax(test_scores)</code> picks the pruning level with best held-out accuracy; the resulting node count shows how compact the optimal tree is.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/4-advanced_fig_2.png" alt="4-advanced" />
<figcaption>Figure 2: Accuracy vs Pruning Strength</figcaption>
</figure>


<figure>
<img src="assets/4-advanced_fig_3.png" alt="4-advanced" />
<figcaption>Figure 3: Tree Size vs Pruning Strength</figcaption>
</figure>

```
Best pruning parameter: 0.004915
Training accuracy: 0.990
Testing accuracy: 0.965
Tree size: 19 nodes
```

Post-pruning is a corrective approach where we first grow a full tree and then trim it back. The `ccp_alpha` parameter controls the strength of pruning:
- Higher values lead to more pruning (smaller trees)
- Lower values lead to less pruning (larger trees)

The optimal pruning strength balances underfitting and overfitting, maximizing performance on unseen data.

## Advanced Tree Growing Techniques

### Custom Impurity Measures

This example shows how to implement and use a custom impurity function:

##### Toy “cubic” impurity vs `gini` / `entropy` trees

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Create a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Custom function to calculate impurity
def calculate_custom_impurity(y_classes):
    """Calculate a custom impurity measure (cubic instead of quadratic)"""
    # Get class probabilities
    _, counts = np.unique(y_classes, return_counts=True)
    probabilities = counts / len(y_classes)

    # Custom impurity (1 - sum of cubed probabilities)
    # Standard Gini would use squared probabilities
    return 1 - np.sum(probabilities ** 3)

# Let's manually calculate impurity for some examples
sample1 = np.array([0, 0, 0, 0, 1])  # 80% class 0, 20% class 1
sample2 = np.array([0, 0, 1, 1, 1])  # 40% class 0, 60% class 1
sample3 = np.array([0, 1, 0, 1, 0])  # 60% class 0, 40% class 1

print(f"Sample 1 impurity: {calculate_custom_impurity(sample1):.3f}")
print(f"Sample 2 impurity: {calculate_custom_impurity(sample2):.3f}")
print(f"Sample 3 impurity: {calculate_custom_impurity(sample3):.3f}")

# While we can't directly use this in sklearn's DecisionTreeClassifier,
# we can compare it with the built-in criteria
for criterion in ['gini', 'entropy']:
    tree = DecisionTreeClassifier(criterion=criterion, random_state=42)
    tree.fit(X, y)
    accuracy = tree.score(X, y)
    nodes = tree.tree_.node_count
    print(f"{criterion.capitalize()} criterion - Accuracy: {accuracy:.3f}, Nodes: {nodes}")
{% endhighlight %}
```
Sample 1 impurity: 0.480
Sample 2 impurity: 0.720
Sample 3 impurity: 0.720
Gini criterion - Accuracy: 1.000, Nodes: 127
Entropy criterion - Accuracy: 1.000, Nodes: 117
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Synthetic Dataset</span>
    </div>
    <div class="code-callout__body">
      <p>1,000 samples with 10 features (5 informative, 2 redundant) give a realistic classification task to compare criterion choices.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Cubic Impurity</span>
    </div>
    <div class="code-callout__body">
      <p>Replaces the squared probabilities of Gini with cubed ones; this de-emphasizes moderate imbalance and focuses on near-pure vs very mixed nodes.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-42" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Criterion Comparison</span>
    </div>
    <div class="code-callout__body">
      <p>Gini and entropy trees are fit identically and compared on in-sample accuracy and node count to show how the criterion affects tree structure.</p>
    </div>
  </div>
</aside>
</div>

```
Sample 1 impurity: 0.480
Sample 2 impurity: 0.720
Sample 3 impurity: 0.720
Gini criterion - Accuracy: 1.000, Nodes: 127
Entropy criterion - Accuracy: 1.000, Nodes: 117
```

While scikit-learn doesn't allow us to directly use custom impurity functions in its implementation, we can understand how different impurity measures affect tree performance. The built-in options are:

- <code>gini</code>: Measures how "mixed" the classes are (based on squared probabilities)
- <code>entropy</code>: Measures how "uncertain" the classes are (based on logarithms)

Different impurity measures can lead to different tree structures and decisions.

## Feature Selection with Decision Trees

Decision trees can help us identify which features are most important:

##### Wine: importances and a reduced feature subset

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load a dataset with many features
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a decision tree
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# Get feature importance
importances = tree.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.bar(range(X.shape[1]), sorted_importances)
plt.xticks(range(X.shape[1]), sorted_features, rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Let's use only the top 5 features
top_features = indices[:5]
X_train_top = X_train[:, top_features]
X_test_top = X_test[:, top_features]

# Train a new tree with only top features
tree_top = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_top.fit(X_train_top, y_train)

# Compare performance
full_accuracy = tree.score(X_test, y_test)
top_accuracy = tree_top.score(X_test_top, y_test)

print(f"Accuracy with all features: {full_accuracy:.3f}")
print(f"Accuracy with top 5 features: {top_accuracy:.3f}")
print(f"Top 5 features: {', '.join([feature_names[i] for i in top_features])}")
{% endhighlight %}

<figure>
<img src="assets/4-advanced_fig_4.png" alt="4-advanced" />
<figcaption>Figure 4: Feature Importance</figcaption>
</figure>

```
Accuracy with all features: 0.963
Accuracy with top 5 features: 0.963
Top 5 features: flavanoids, color_intensity, proline, ash, alcohol
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-20" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train Full Tree</span>
    </div>
    <div class="code-callout__body">
      <p>A depth-4 tree is fit on all 13 Wine features to generate importances from impurity-reduction accumulated across all splits.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-35" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Rank and Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Features are sorted descending by importance; the bar chart shows the relative contribution of each column at a glance.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="37-53" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Reduced Feature Retrain</span>
    </div>
    <div class="code-callout__body">
      <p>The top-5 feature subset is used to refit the same tree; comparing accuracy before and after reveals whether the 8 dropped features added value.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/4-advanced_fig_4.png" alt="4-advanced" />
<figcaption>Figure 4: Feature Importance</figcaption>
</figure>

```
Accuracy with all features: 0.963
Accuracy with top 5 features: 0.963
Top 5 features: flavanoids, color_intensity, proline, ash, alcohol
```

This technique shows how we can:
1. Identify which features are most important in our decision tree
2. Use this information to create simpler models with fewer features
3. Often maintain similar performance with a much simpler model

Feature importance in decision trees is calculated based on how much each feature improves the purity at each split across the entire tree.

## Introduction to Tree Ensembles

Think of ensembles like a team of experts working together. Each expert (tree) might make mistakes, but together they're more accurate.

### 1. Random Forest Preview

##### Single tree vs `RandomForestClassifier` (CV boxplot)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

# Load a dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Compare single tree vs forest
# Single tree
tree = DecisionTreeClassifier(max_depth=3)
tree_scores = cross_val_score(tree, X, y, cv=5)

# Random forest
forest = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=3,       # Depth of each tree
    random_state=42
)
forest_scores = cross_val_score(forest, X, y, cv=5)

# Plot comparison
plt.figure(figsize=(8, 6))
plt.boxplot([tree_scores, forest_scores],
            labels=['Single Tree', 'Random Forest'])
plt.title('Which is Better: One Expert or Many?')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
plt.grid(axis='y')
plt.show()

# Print average scores
print(f"Single Tree Average: {tree_scores.mean():.3f}")
print(f"Random Forest Average: {forest_scores.mean():.3f}")
{% endhighlight %}

<figure>
<img src="assets/4-advanced_fig_5.png" alt="4-advanced" />
<figcaption>Figure 5: Which is Better: One Expert or Many?</figcaption>
</figure>

```
Single Tree Average: 0.924
Random Forest Average: 0.954
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load Breast Cancer</span>
    </div>
    <div class="code-callout__body">
      <p>569 samples with 30 features; binary classification makes the accuracy gap between tree and forest easy to see.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-24" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Single Tree vs Forest</span>
    </div>
    <div class="code-callout__body">
      <p>Both models use <code>max_depth=3</code> per tree so the only difference is bagging and feature subsampling inside the forest.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="26-38" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Boxplot Comparison</span>
    </div>
    <div class="code-callout__body">
      <p>A boxplot of 5-fold scores shows median accuracy and variance for each model; the forest typically has a higher median and narrower spread.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/4-advanced_fig_5.png" alt="4-advanced" />
<figcaption>Figure 5: Which is Better: One Expert or Many?</figcaption>
</figure>

```
Single Tree Average: 0.924
Random Forest Average: 0.954
```

Random Forest creates many diverse decision trees by:
1. Training each tree on a random subset of the data (bootstrapping)
2. Considering only a random subset of features at each split
3. Combining their predictions through voting (for classification) or averaging (for regression)

This diversity helps the ensemble overcome individual tree weaknesses and produce more robust predictions.

### 2. Gradient Boosting Preview

##### `make_circles`: sequential boosting with refits per `n_estimators`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

# Create a challenging dataset
X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize a gradient boosting model
boosting = GradientBoostingClassifier(
    n_estimators=100,      # Maximum number of trees
    learning_rate=0.1,     # How much to learn from each mistake
    max_depth=3,           # Depth of each tree
    random_state=42
)

# Train and track performance as we add more trees
train_scores = []
test_scores = []
n_estimators_range = range(1, 101, 5)  # Check every 5 trees

for i in n_estimators_range:
    # Set number of trees
    boosting.n_estimators = i
    # Train model
    boosting.fit(X_train, y_train)
    # Record scores
    train_scores.append(boosting.score(X_train, y_train))
    test_scores.append(boosting.score(X_test, y_test))

# Plot learning progress
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'b-', label='Training')
plt.plot(n_estimators_range, test_scores, 'r-', label='Testing')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Learning from Mistakes Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Find optimal number of trees
best_n_estimators = n_estimators_range[np.argmax(test_scores)]
print(f"Optimal number of trees: {best_n_estimators}")
print(f"Best accuracy: {max(test_scores):.3f}")
{% endhighlight %}

<figure>
<img src="assets/4-advanced_fig_6.png" alt="4-advanced" />
<figcaption>Figure 6: Learning from Mistakes Over Time</figcaption>
</figure>

```
Optimal number of trees: 6
Best accuracy: 0.827
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Circles Dataset</span>
    </div>
    <div class="code-callout__body">
      <p><code>make_circles</code> creates a non-linear two-class problem; boosting handles it by stacking many shallow trees that each correct residual errors.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">GBM Setup</span>
    </div>
    <div class="code-callout__body">
      <p><code>learning_rate=0.1</code> shrinks each tree's contribution; lower values require more trees but generalize better.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-35" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Stage-wise Scoring</span>
    </div>
    <div class="code-callout__body">
      <p>Resetting <code>n_estimators</code> and refitting at each step tracks how accuracy evolves as more trees are added to the ensemble.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="37-49" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Learning Curve and Optimum</span>
    </div>
    <div class="code-callout__body">
      <p>The plot shows test accuracy stabilizing or declining after the optimal stage; <code>argmax</code> identifies the best tree count from the sweep.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/4-advanced_fig_6.png" alt="4-advanced" />
<figcaption>Figure 6: Learning from Mistakes Over Time</figcaption>
</figure>

```
Optimal number of trees: 6
Best accuracy: 0.827
```

Gradient Boosting works by:
1. Starting with a simple model
2. Identifying where this model makes mistakes
3. Adding a new tree specifically focused on correcting those mistakes
4. Repeating this process, with each new tree focusing on the remaining errors

This sequential learning process allows the model to focus on the difficult cases and gradually improve its predictions.

## Advanced Visualization Techniques

### Decision Path Highlighter

##### `decision_path` and printing split rules for one sample

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create and train a tree
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)

# Select a sample to trace
sample_idx = 42
sample = X[sample_idx:sample_idx+1]
true_class = iris.target_names[y[sample_idx]]

# Get decision path
path = tree_clf.decision_path(sample)

# Print sample information
print(f"Sample features: {sample[0]}")
print(f"True class: {true_class}")
print(f"Predicted class: {iris.target_names[tree_clf.predict(sample)[0]]}")

# Create visualization with highlighted path
plt.figure(figsize=(20, 10))
plot_tree(
    tree_clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

# Extract node IDs in the path
path_indices = path.indices

# Print the decision rules for this sample
print("\nDecision path:")
for i, node_id in enumerate(path_indices):
    if node_id != tree_clf.tree_.node_count - 1:  # Skip leaf nodes
        feature = tree_clf.tree_.feature[node_id]
        threshold = tree_clf.tree_.threshold[node_id]
        feature_name = iris.feature_names[feature]

        if sample[0, feature] <= threshold:
            direction = "<="
        else:
            direction = ">"

        print(f"Step {i+1}: Is {feature_name} {direction} {threshold:.2f}? {'Yes' if sample[0, feature] <= threshold else 'No'}")

plt.title('Decision Tree with Highlighted Path')
plt.show()
{% endhighlight %}

<figure>
<img src="assets/4-advanced_fig_7.png" alt="4-advanced" />
<figcaption>Figure 7: Decision Tree with Highlighted Path</figcaption>
</figure>

```
Sample features: [4.4 3.2 1.3 0.2]
True class: setosa
Predicted class: setosa

Decision path:
Step 1: Is petal length (cm) <= 2.45? Yes
Step 2: Is petal length (cm) > -2.00? No
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-18" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Setup and Select Sample</span>
    </div>
    <div class="code-callout__body">
      <p>A depth-3 Iris tree is fit; sample index 42 is sliced as a single-row matrix (required by <code>decision_path</code>) and its true class is stored for comparison.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-26" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Get and Print Path</span>
    </div>
    <div class="code-callout__body">
      <p><code>decision_path</code> returns a sparse matrix of node visits; <code>.indices</code> gives the ordered list of node IDs from root to leaf.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-36" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Visualize Full Tree</span>
    </div>
    <div class="code-callout__body">
      <p><code>plot_tree</code> shows the complete structure; in practice you would color the path nodes—here the printed trace explains each split rule verbally.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="38-55" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Trace Split Rules</span>
    </div>
    <div class="code-callout__body">
      <p>For each visited node (excluding the leaf), the feature name, threshold, and direction are printed so you can follow the logic step by step.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/4-advanced_fig_7.png" alt="4-advanced" />
<figcaption>Figure 7: Decision Tree with Highlighted Path</figcaption>
</figure>

```
Sample features: [4.4 3.2 1.3 0.2]
True class: setosa
Predicted class: setosa

Decision path:
Step 1: Is petal length (cm) <= 2.45? Yes
Step 2: Is petal length (cm) > -2.00? No
```

This visualization helps us understand exactly how a decision tree makes a specific prediction by:
1. Tracing the path from the root to the leaf for a specific sample
2. Showing each decision point along the way
3. Revealing the decision rules that led to the final prediction

This transparency is one of the major advantages of decision trees over black-box models.

## Common Advanced Techniques

### 1. Handling Imbalanced Data

##### `class_weight='balanced'` vs default on synthetic imbalance

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Create an imbalanced dataset (90% class 0, 10% class 1)
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.9, 0.1],  # Class distribution
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train a regular tree
regular_tree = DecisionTreeClassifier(random_state=42)
regular_tree.fit(X_train, y_train)

# Train a tree with class weights
weighted_tree = DecisionTreeClassifier(
    class_weight='balanced',  # Give more weight to minority class
    random_state=42
)
weighted_tree.fit(X_train, y_train)

# Evaluate both models
print("Regular Tree:")
y_pred_regular = regular_tree.predict(X_test)
print(confusion_matrix(y_test, y_pred_regular))
print(classification_report(y_test, y_pred_regular))

print("\nWeighted Tree:")
y_pred_weighted = weighted_tree.predict(X_test)
print(confusion_matrix(y_test, y_pred_weighted))
print(classification_report(y_test, y_pred_weighted))
{% endhighlight %}
```
Regular Tree:
[[256  13]
 [ 13  18]]
              precision    recall  f1-score   support

           0       0.95      0.95      0.95       269
           1       0.58      0.58      0.58        31

    accuracy                           0.91       300
   macro avg       0.77      0.77      0.77       300
weighted avg       0.91      0.91      0.91       300


Weighted Tree:
[[260   9]
 [ 14  17]]
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       269
           1       0.65      0.55      0.60        31

    accuracy                           0.92       300
   macro avg       0.80      0.76      0.78       300
weighted avg       0.92      0.92      0.92       300
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imbalanced Dataset</span>
    </div>
    <div class="code-callout__body">
      <p>The 90/10 class split simulates a realistic imbalance scenario (fraud, rare disease); <code>stratify=y</code> preserves this ratio in the test split.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-30" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Two Trees</span>
    </div>
    <div class="code-callout__body">
      <p>The regular tree ignores class frequency; <code>class_weight='balanced'</code> upweights minority samples so the tree doesn't just predict the majority class.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="32-41" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compare Reports</span>
    </div>
    <div class="code-callout__body">
      <p>The confusion matrix and classification report expose precision, recall, and F1 per class so you can see how weighting shifts the minority-class trade-off.</p>
    </div>
  </div>
</aside>
</div>

```
Regular Tree:
[[256  13]
 [ 13  18]]
              precision    recall  f1-score   support

           0       0.95      0.95      0.95       269
           1       0.58      0.58      0.58        31

    accuracy                           0.91       300
   macro avg       0.77      0.77      0.77       300
weighted avg       0.91      0.91      0.91       300


Weighted Tree:
[[260   9]
 [ 14  17]]
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       269
           1       0.65      0.55      0.60        31

    accuracy                           0.92       300
   macro avg       0.80      0.76      0.78       300
weighted avg       0.92      0.92      0.92       300
```

When dealing with imbalanced data (where some classes are much more common than others), we can:
1. Use <code>class_weight='balanced'</code> to automatically adjust weights inversely proportional to class frequencies
2. Manually specify weights for each class using a dictionary, e.g., <code>class_weight={0: 1, 1: 9}</code>
3. Evaluate models using metrics beyond accuracy, such as precision, recall, and F1-score

These techniques help ensure the model pays attention to minority classes instead of just predicting the majority class.

### 2. Cross-Validation

##### `KFold` vs `StratifiedKFold` on breast cancer

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.datasets import load_breast_cancer

# Load a dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Regular K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
scores_kf = cross_val_score(tree, X, y, cv=kf)

# Stratified K-Fold (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(tree, X, y, cv=skf)

print("Regular K-Fold CV scores:", scores_kf)
print(f"Average: {scores_kf.mean():.3f}, Std Dev: {scores_kf.std():.3f}")

print("\nStratified K-Fold CV scores:", scores_skf)
print(f"Average: {scores_skf.mean():.3f}, Std Dev: {scores_skf.std():.3f}")
{% endhighlight %}
```
Regular K-Fold CV scores: [0.94736842 0.95614035 0.9122807  0.92105263 0.9380531 ]
Average: 0.935, Std Dev: 0.016

Stratified K-Fold CV scores: [0.92105263 0.88596491 0.94736842 0.92982456 0.9380531 ]
Average: 0.924, Std Dev: 0.021
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load and Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Breast cancer is a binary dataset with ~63/37 class split — a realistic scenario where stratification matters for consistent fold class ratios.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-18" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Two CV Strategies</span>
    </div>
    <div class="code-callout__body">
      <p><code>KFold</code> shuffles randomly; <code>StratifiedKFold</code> ensures each fold mirrors the overall class distribution, reducing variance in fold-to-fold scores.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-25" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Mean and Std</span>
    </div>
    <div class="code-callout__body">
      <p>Printing per-fold scores plus mean and std lets you compare stability; stratified CV often shows a tighter std on imbalanced data.</p>
    </div>
  </div>
</aside>
</div>

```
Regular K-Fold CV scores: [0.94736842 0.95614035 0.9122807  0.92105263 0.9380531 ]
Average: 0.935, Std Dev: 0.016

Stratified K-Fold CV scores: [0.92105263 0.88596491 0.94736842 0.92982456 0.9380531 ]
Average: 0.924, Std Dev: 0.021
```

Cross-validation helps us get a more reliable estimate of model performance by:
1. Splitting the data into multiple folds
2. Training and evaluating the model multiple times on different splits
3. Averaging the results to get a more stable performance metric

Stratified cross-validation specifically ensures that each fold maintains the same class distribution as the original dataset, which is especially important for imbalanced data.

## Gotchas

- **Choosing `ccp_alpha` from the pruning path without cross-validation** — the post-pruning example prints the optimal alpha based on a single 70/30 split; with small datasets, the "best" alpha can vary substantially across splits; use `cross_val_score` at several candidate alpha values before committing.
- **Assuming `class_weight='balanced'` always improves results** — balanced weighting forces the model to pay equal attention to all classes regardless of their true prevalence; if the majority class is genuinely the correct answer most of the time, balanced weights can hurt both precision and overall accuracy by over-correcting.
- **Comparing accuracy of Gini vs entropy trees on training data only** — both built-in criteria produce perfect 1.000 in-sample accuracy on the synthetic 1,000-sample dataset; the meaningful comparison is cross-validated test accuracy and tree compactness (node count), not in-sample score.
- **Mutating `boosting.n_estimators` in a loop and calling `fit` each iteration** — this works but refits the full model from scratch each time, which is O(n × T) instead of O(T); the correct approach is to fit once with the maximum `n_estimators` and use `staged_predict` to extract scores at intermediate stages without re-fitting.
- **Using the decision path trace as a full explanation of confidence** — `decision_path` shows which nodes fired for a sample, but a leaf with 1 sample from training has 100% confidence for the majority class even though no generalisation evidence supports that certainty; high `predict_proba` at a small-sample leaf is unreliable.
- **Selecting features by tree importance and then using cross-validation on the reduced feature set without re-running the selection inside each fold** — fitting the feature selector on all data and then cross-validating inflates performance because the CV folds have already "seen" the importance ranking; the feature selection step must be inside the CV pipeline to avoid leakage.

## Practice Exercise

Try these advanced techniques on your own:

1. Compare pre-pruning and post-pruning on a dataset of your choice
2. Implement a custom impurity measure and compare it to Gini and entropy
3. Visualize feature importances and decision paths for a specific prediction
4. Apply class weights to handle an imbalanced dataset

## Next Steps

Ready to apply these techniques? Check out:

1. [Real-world applications](5-applications.md) of decision trees
2. How to deploy decision trees in production
3. Advanced ensemble methods (Random Forests, Gradient Boosting)
4. Hyperparameter tuning techniques
