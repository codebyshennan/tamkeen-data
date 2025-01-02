# Mathematical Foundation of Random Forest 📐

Let's dive into the mathematical concepts that make Random Forests work! Understanding these foundations will help you make better decisions when implementing and tuning your models.

## Bootstrap Aggregating (Bagging) 🎲

### Mathematical Definition
For a dataset $D$ of size $n$, bagging creates $m$ new datasets $D_i$ of size $n$ by sampling from $D$ uniformly and with replacement. Each data point has probability $1 - (1-\frac{1}{n})^n \approx 0.632$ of being selected.

```python
import numpy as np

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]
```

### Out-of-Bag (OOB) Estimation
The OOB error is an unbiased estimate of the test error:

$$E_{oob} = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{f}_{oob}(x_i))$$

where $\hat{f}_{oob}(x_i)$ is the average prediction of all trees where $i$ was not used in training.

## Random Feature Selection 🎯

### Feature Sampling
At each split, only a random subset of features is considered:
- For classification: typically $\sqrt{p}$ features
- For regression: typically $p/3$ features
where $p$ is the total number of features.

```python
def get_random_features(n_features, n_select):
    """Select random subset of features"""
    return np.random.choice(
        n_features, 
        size=n_select, 
        replace=False
    )
```

## Ensemble Prediction 🤝

### Classification
For a classification problem with $K$ classes, the final prediction is:

$$\hat{y} = \text{argmax}_k \sum_{t=1}^T I(h_t(x) = k)$$

where:
- $h_t(x)$ is the prediction of the $t$-th tree
- $T$ is the total number of trees
- $I()$ is the indicator function

### Regression
For regression, the final prediction is the average:

$$\hat{y} = \frac{1}{T}\sum_{t=1}^T h_t(x)$$

## Feature Importance 📊

### Gini Importance
For a feature $j$, the importance is:

$$\text{Imp}_j = \sum_{t=1}^T \sum_{n \in N_t} w_n \Delta i(s_t^n, j)$$

where:
- $N_t$ is the set of nodes in tree $t$
- $w_n$ is the weighted number of samples reaching node $n$
- $\Delta i(s_t^n, j)$ is the decrease in impurity

```python
def gini_impurity(y):
    """Calculate Gini impurity"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)
```

## Error Analysis 📉

### Bias-Variance Decomposition
Random Forests reduce variance while maintaining bias:

$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

The variance reduction comes from averaging multiple trees:

$$\text{Var}(\text{average}) = \frac{\rho \sigma^2}{T} + \frac{1-\rho}{T}\sigma^2$$

where:
- $\rho$ is the correlation between trees
- $\sigma^2$ is the variance of individual trees
- $T$ is the number of trees

## Convergence Properties 🎯

### Law of Large Numbers
As the number of trees increases, the forest converges to:

$$\lim_{T \to \infty} \frac{1}{T}\sum_{t=1}^T h_t(x) = E_{D_t}[h(x|D_t)]$$

This explains why Random Forests don't overfit as more trees are added.

## Optimization Criteria 🎛️

### Split Quality
For a split $s$ on feature $j$ at node $n$:

$$\Delta i(s, j) = i(n) - \frac{n_L}{N}i(n_L) - \frac{n_R}{N}i(n_R)$$

where:
- $i(n)$ is the impurity at node $n$
- $n_L, n_R$ are the number of samples in left/right children
- $N$ is the total number of samples at the node

```python
def information_gain(parent, left, right):
    """Calculate information gain for a split"""
    n = len(parent)
    n_l, n_r = len(left), len(right)
    
    gain = gini_impurity(parent) - (
        n_l/n * gini_impurity(left) +
        n_r/n * gini_impurity(right)
    )
    return gain
```

## Hyperparameter Effects 🔧

### Number of Trees
- Error rate converges as $T \to \infty$
- More trees = better stability
- Diminishing returns after certain point

### Max Features
- Lower = more randomness = higher diversity
- Higher = better individual trees
- Optimal value depends on problem

### Tree Depth
- Controls bias-variance tradeoff
- Deeper trees = lower bias, higher variance
- Shallower trees = higher bias, lower variance

## Next Steps 🚀

Now that you understand the mathematics behind Random Forests, let's move on to [Implementation](3-implementation.md) to see how to put these concepts into practice!
