# Mathematical Foundation of Gradient Boosting 📐

Let's dive into the mathematical concepts that make Gradient Boosting work! Understanding these foundations will help you make better decisions when implementing and tuning your models.

## The Boosting Framework 🎯

### Additive Model
Gradient Boosting builds an additive model:

$$F_M(x) = \sum_{m=1}^M \gamma_m h_m(x)$$

where:
- $F_M(x)$ is the final model
- $h_m(x)$ are the weak learners
- $\gamma_m$ are the weights
- $M$ is the number of iterations

## Gradient Descent in Function Space 📉

### Loss Function
The model minimizes a loss function $L(y, F(x))$:

$$F_m(x) = F_{m-1}(x) - \gamma_m \nabla_F L(y, F_{m-1}(x))$$

Common loss functions:
- **Regression**: MSE, MAE, Huber
- **Classification**: Log loss, Exponential loss

```python
def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    return np.mean((y_true - y_pred) ** 2)

def log_loss(y_true, y_pred):
    """Binary Cross Entropy loss"""
    return -np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * np.log(1 - y_pred)
    )
```

## Residual Learning 🎯

### Computing Residuals
Each new model fits the residuals of previous models:

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

```python
def compute_residuals(y_true, y_pred, loss='mse'):
    """Compute residuals based on loss function"""
    if loss == 'mse':
        return y_true - y_pred
    elif loss == 'log':
        return y_true - 1 / (1 + np.exp(-y_pred))
```

## Learning Rate and Shrinkage 🐢

### Shrinkage Parameter
The learning rate $\nu$ controls the contribution of each tree:

$$F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)$$

where $0 < \nu \leq 1$ is the learning rate.

```python
def update_predictions(y_pred, tree_pred, learning_rate=0.1):
    """Update predictions with learning rate"""
    return y_pred + learning_rate * tree_pred
```

## Tree Building Process 🌳

### Split Finding
For regression trees, find split $s$ that maximizes gain:

$$\text{Gain}(s) = \frac{1}{2} \left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right]$$

where:
- $G$ is the sum of gradients
- $H$ is the sum of hessians
- $\lambda$ is the regularization parameter

```python
def find_best_split(gradients, hessians, feature_values):
    """Find best split point using gradients and hessians"""
    best_gain = 0
    best_split = None
    
    for value in feature_values:
        left_mask = feature_values <= value
        right_mask = ~left_mask
        
        G_L = gradients[left_mask].sum()
        G_R = gradients[right_mask].sum()
        H_L = hessians[left_mask].sum()
        H_R = hessians[right_mask].sum()
        
        gain = calculate_split_gain(G_L, G_R, H_L, H_R)
        
        if gain > best_gain:
            best_gain = gain
            best_split = value
    
    return best_split, best_gain
```

## Regularization Terms 🎛️

### Objective Function
The regularized objective:

$$\text{Obj} = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

where $\Omega(f)$ is the regularization term:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

- $T$ is the number of leaves
- $w_j$ are the leaf weights
- $\gamma$ and $\lambda$ are regularization parameters

## Early Stopping 🛑

### Validation-based Stopping
Stop training when validation error increases:

$$\text{Stop if } L_{\text{val}}^{(t)} > L_{\text{val}}^{(t-k)} \text{ for } k \text{ consecutive rounds}$$

```python
class EarlyStopping:
    """Early stopping implementation"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

## Feature Importance 📊

### Gain-based Importance
Feature importance based on gain:

$$\text{Importance}(f) = \sum_{t=1}^T \sum_{j \in \{splits on f\}} \text{Gain}(j)$$

```python
def calculate_feature_importance(trees, feature_names):
    """Calculate feature importance across all trees"""
    importance = defaultdict(float)
    
    for tree in trees:
        for feature, gain in tree.feature_gains.items():
            importance[feature_names[feature]] += gain
    
    # Normalize
    total = sum(importance.values())
    return {f: v/total for f, v in importance.items()}
```

## Convergence Properties 🎯

### Exponential Loss Bound
For classification with exponential loss:

$$L(F_M) \leq \exp(-2\sum_{m=1}^M \gamma_m^2)$$

This shows that the training error decreases exponentially with the number of iterations.

## Next Steps 🚀

Now that you understand the mathematics behind Gradient Boosting, let's move on to [Implementation](3-implementation.md) to see how to put these concepts into practice!
