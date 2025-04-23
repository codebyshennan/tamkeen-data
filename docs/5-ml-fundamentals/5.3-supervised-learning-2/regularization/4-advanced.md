# Advanced Regularization Techniques 🚀

Think of advanced regularization techniques like learning advanced driving techniques - they build upon the basics but help you handle more complex situations. Let's explore these sophisticated methods in a way that's easy to understand!

## Adaptive Regularization 🎯

Adaptive regularization is like having a smart teacher who adjusts their teaching style based on each student's needs. Instead of treating all features the same, it gives more attention to the important ones.

### 1. Adaptive Lasso

```python
class AdaptiveLasso:
    """Adaptive Lasso implementation"""
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha  # Regularization strength
        self.gamma = gamma  # Weight adjustment power
        self.weights = None  # Feature weights
        self.lasso = None   # Lasso model
    
    def fit(self, X, y):
        # Initial OLS fit
        # This is like getting a first impression of each feature's importance
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression()
        ols.fit(X, y)
        
        # Calculate adaptive weights
        # This is like adjusting the difficulty level for each subject
        self.weights = 1 / (np.abs(ols.coef_) ** self.gamma)
        
        # Scale features by weights
        # This is like giving more attention to important subjects
        X_weighted = X * self.weights
        
        # Fit Lasso with weighted features
        # This is like having a strict teacher who focuses on important subjects
        self.lasso = Lasso(alpha=self.alpha)
        self.lasso.fit(X_weighted, y)
        
        return self
    
    def predict(self, X):
        X_weighted = X * self.weights
        return self.lasso.predict(X_weighted)
```

### 2. Group Lasso

Group Lasso is like having a team coach who manages groups of players together, rather than individual players. It's useful when you have related features that should be selected or dropped as a group.

```python
def group_lasso_penalty(coef_groups):
    """Calculate group lasso penalty"""
    return sum(
        np.sqrt(len(group)) * np.linalg.norm(group, 2)
        for group in coef_groups
    )

class GroupLasso:
    """Group Lasso implementation"""
    def __init__(self, groups, alpha=1.0):
        self.groups = groups  # Feature groups
        self.alpha = alpha    # Regularization strength
    
    def fit(self, X, y):
        # Implementation using proximal gradient descent
        # This is like taking small steps while respecting group boundaries
        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)
        
        for _ in range(1000):  # Max iterations
            # Gradient step
            # This is like moving in the direction that improves performance
            grad = -X.T @ (y - X @ self.coef_)
            self.coef_ -= 0.01 * grad  # Learning rate
            
            # Proximal operator for group lasso
            # This is like enforcing group-level sparsity
            for group in self.groups:
                group_norm = np.linalg.norm(
                    self.coef_[group], 2
                )
                if group_norm > 0:
                    shrinkage = max(
                        0, 
                        1 - self.alpha / group_norm
                    )
                    self.coef_[group] *= shrinkage
        
        return self
```

## Advanced Optimization Techniques 🔧

### 1. Coordinate Descent

Coordinate descent is like solving a puzzle one piece at a time. Instead of trying to solve everything at once, it focuses on one feature at a time.

```python
def coordinate_descent_lasso(X, y, alpha, max_iter=1000):
    """Coordinate descent for Lasso"""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    
    for _ in range(max_iter):
        coef_old = coef.copy()
        
        # Update each coordinate
        # This is like adjusting one piece of the puzzle at a time
        for j in range(n_features):
            # Remove current feature contribution
            r = y - X @ coef + X[:, j] * coef[j]
            
            # Calculate update
            rho = X[:, j] @ r
            if abs(rho) <= alpha:
                coef[j] = 0  # Feature is not important enough
            else:
                coef[j] = (
                    np.sign(rho) * (abs(rho) - alpha) /
                    (X[:, j] @ X[:, j])
                )
        
        # Check convergence
        # This is like checking if the puzzle is complete
        if np.allclose(coef, coef_old):
            break
    
    return coef
```

### 2. ADMM Implementation

ADMM (Alternating Direction Method of Multipliers) is like having two people work together to solve a problem, each focusing on their part while coordinating with the other.

```python
def admm_lasso(X, y, alpha, rho=1.0, max_iter=1000):
    """ADMM algorithm for Lasso"""
    n_samples, n_features = X.shape
    
    # Initialize variables
    beta = np.zeros(n_features)  # Main variable
    z = np.zeros(n_features)     # Auxiliary variable
    u = np.zeros(n_features)     # Dual variable
    
    # Pre-compute matrix inverse
    # This is like preparing tools before starting work
    XtX = X.T @ X
    L = np.linalg.cholesky(XtX + rho * np.eye(n_features))
    
    for _ in range(max_iter):
        # Update beta
        # This is like solving the main problem
        q = X.T @ y + rho * (z - u)
        beta = np.linalg.solve(L.T, np.linalg.solve(L, q))
        
        # Update z
        # This is like solving the auxiliary problem
        z_old = z
        beta_hat = beta + u
        z = np.sign(beta_hat) * np.maximum(
            np.abs(beta_hat) - alpha/rho, 0
        )
        
        # Update u
        # This is like coordinating between the two solutions
        u = u + beta - z
        
        # Check convergence
        # This is like checking if both people agree
        if np.allclose(z, z_old):
            break
    
    return beta
```

## Advanced Cross-Validation 📊

### 1. Stability Selection

Stability selection is like taking multiple tests to ensure you really understand the material, not just memorizing the answers. It helps identify features that are consistently important.

```python
class StabilitySelection:
    """Stability selection for feature selection"""
    def __init__(self, estimator, n_bootstrap=100, 
                 threshold=0.5):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap  # Number of tests
        self.threshold = threshold      # Minimum selection probability
        self.selection_probabilities_ = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        feature_counts = np.zeros(n_features)
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            # This is like taking a different test each time
            indices = np.random.choice(
                n_samples, size=n_samples//2, replace=False
            )
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit model
            # This is like solving the test
            self.estimator.fit(X_boot, y_boot)
            
            # Count selected features
            # This is like counting how often each topic appears
            feature_counts += (
                self.estimator.coef_ != 0
            ).astype(int)
        
        # Calculate selection probabilities
        # This is like calculating how often each topic is important
        self.selection_probabilities_ = (
            feature_counts / self.n_bootstrap
        )
        
        return self
```

### 2. Randomized Lasso

Randomized Lasso is like having multiple teachers evaluate a student, each with slightly different criteria. This helps identify features that are robustly important.

```python
class RandomizedLasso:
    """Randomized Lasso implementation"""
    def __init__(self, alpha=1.0, scaling=0.5, 
                 n_resampling=100):
        self.alpha = alpha
        self.scaling = scaling
        self.n_resampling = n_resampling
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        scores = np.zeros(n_features)
        
        for _ in range(self.n_resampling):
            # Random scaling of features
            # This is like having different teachers with different priorities
            scalings = np.random.uniform(
                self.scaling, 1.0, size=n_features
            )
            X_scaled = X * scalings
            
            # Fit Lasso
            # This is like having each teacher evaluate the student
            lasso = Lasso(alpha=self.alpha)
            lasso.fit(X_scaled, y)
            
            # Update scores
            # This is like combining all teachers' evaluations
            scores += (lasso.coef_ != 0).astype(int)
        
        self.scores_ = scores / self.n_resampling
        return self
```

## Regularization for Neural Networks 🧠

### 1. Weight Decay Implementation

Weight decay in neural networks is like having rules that prevent the network from becoming too complex, similar to how regularization works in linear models.

```python
import tensorflow as tf

def create_regularized_model(input_shape, l2_lambda=0.01):
    """Create neural network with L2 regularization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            128, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
            input_shape=input_shape
        ),
        tf.keras.layers.Dropout(0.3),  # Like randomly forgetting some information
        tf.keras.layers.Dense(
            64, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    return model
```

## Common Mistakes to Avoid ⚠️

1. Using too complex regularization when simpler methods would work
2. Not understanding the assumptions behind each method
3. Ignoring feature scaling in advanced methods
4. Not validating the stability of selected features
5. Overlooking the computational cost of advanced methods

## Next Steps 🚀

Now that you understand advanced regularization techniques, let's move on to [Applications](5-applications.md) to see how these methods are used in real-world scenarios!

## Additional Resources 📚

- [Advanced Regularization Techniques](https://towardsdatascience.com/advanced-regularization-techniques-1c4e6b5c5343)
- [Stability Selection in Practice](https://www.stat.berkeley.edu/~bickel/papers/2010_StabilitySelection.pdf)
- [ADMM for Machine Learning](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
