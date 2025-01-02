# Advanced Regularization Techniques 🚀

Let's explore advanced concepts and techniques to take your regularization skills to the next level!

## Adaptive Regularization 🎯

### 1. Adaptive Lasso
```python
class AdaptiveLasso:
    """Adaptive Lasso implementation"""
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.weights = None
        self.lasso = None
    
    def fit(self, X, y):
        # Initial OLS fit
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression()
        ols.fit(X, y)
        
        # Calculate adaptive weights
        self.weights = 1 / (np.abs(ols.coef_) ** self.gamma)
        
        # Scale features by weights
        X_weighted = X * self.weights
        
        # Fit Lasso with weighted features
        self.lasso = Lasso(alpha=self.alpha)
        self.lasso.fit(X_weighted, y)
        
        return self
    
    def predict(self, X):
        X_weighted = X * self.weights
        return self.lasso.predict(X_weighted)
```

### 2. Group Lasso
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
        self.groups = groups
        self.alpha = alpha
    
    def fit(self, X, y):
        # Implementation using proximal gradient descent
        # (simplified version)
        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)
        
        for _ in range(1000):  # Max iterations
            # Gradient step
            grad = -X.T @ (y - X @ self.coef_)
            self.coef_ -= 0.01 * grad  # Learning rate
            
            # Proximal operator for group lasso
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
```python
def coordinate_descent_lasso(X, y, alpha, max_iter=1000):
    """Coordinate descent for Lasso"""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    
    for _ in range(max_iter):
        coef_old = coef.copy()
        
        # Update each coordinate
        for j in range(n_features):
            # Remove current feature contribution
            r = y - X @ coef + X[:, j] * coef[j]
            
            # Calculate update
            rho = X[:, j] @ r
            if abs(rho) <= alpha:
                coef[j] = 0
            else:
                coef[j] = (
                    np.sign(rho) * (abs(rho) - alpha) /
                    (X[:, j] @ X[:, j])
                )
        
        # Check convergence
        if np.allclose(coef, coef_old):
            break
    
    return coef
```

### 2. ADMM Implementation
```python
def admm_lasso(X, y, alpha, rho=1.0, max_iter=1000):
    """ADMM algorithm for Lasso"""
    n_samples, n_features = X.shape
    
    # Initialize variables
    beta = np.zeros(n_features)
    z = np.zeros(n_features)
    u = np.zeros(n_features)
    
    # Pre-compute matrix inverse
    XtX = X.T @ X
    L = np.linalg.cholesky(XtX + rho * np.eye(n_features))
    
    for _ in range(max_iter):
        # Update beta
        q = X.T @ y + rho * (z - u)
        beta = np.linalg.solve(L.T, np.linalg.solve(L, q))
        
        # Update z
        z_old = z
        beta_hat = beta + u
        z = np.sign(beta_hat) * np.maximum(
            np.abs(beta_hat) - alpha/rho, 0
        )
        
        # Update u
        u = u + beta - z
        
        # Check convergence
        if np.allclose(z, z_old):
            break
    
    return beta
```

## Advanced Cross-Validation 📊

### 1. Stability Selection
```python
class StabilitySelection:
    """Stability selection for feature selection"""
    def __init__(self, estimator, n_bootstrap=100, 
                 threshold=0.5):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.selection_probabilities_ = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        feature_counts = np.zeros(n_features)
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(
                n_samples, size=n_samples//2, replace=False
            )
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit model
            self.estimator.fit(X_boot, y_boot)
            
            # Count selected features
            feature_counts += (
                self.estimator.coef_ != 0
            ).astype(int)
        
        # Calculate selection probabilities
        self.selection_probabilities_ = (
            feature_counts / self.n_bootstrap
        )
        
        return self
```

### 2. Randomized Lasso
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
            scalings = np.random.uniform(
                self.scaling, 1.0, size=n_features
            )
            X_scaled = X * scalings
            
            # Fit Lasso
            lasso = Lasso(alpha=self.alpha)
            lasso.fit(X_scaled, y)
            
            # Update scores
            scores += (lasso.coef_ != 0).astype(int)
        
        self.scores_ = scores / self.n_resampling
        return self
```

## Regularization for Neural Networks 🧠

### 1. Weight Decay Implementation
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
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            64, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    return model
```

### 2. Custom Regularizers
```python
class ElasticNetRegularizer(tf.keras.regularizers.Regularizer):
    """Custom Elastic Net regularizer for neural networks"""
    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2
    
    def __call__(self, weights):
        return (
            self.l1 * tf.reduce_sum(tf.abs(weights)) +
            self.l2 * tf.reduce_sum(tf.square(weights))
        )
    
    def get_config(self):
        return {'l1': self.l1, 'l2': self.l2}
```

## Production Deployment 🚀

### 1. Model Versioning
```python
class VersionedRegularizedModel:
    """Regularized model with versioning"""
    def __init__(self, model_type='ridge', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.version = 1
        self.history = {}
        
        # Create model
        if model_type == 'ridge':
            self.model = Ridge(**kwargs)
        elif model_type == 'lasso':
            self.model = Lasso(**kwargs)
        else:
            self.model = ElasticNet(**kwargs)
    
    def fit(self, X, y):
        # Train model
        self.model.fit(X, y)
        
        # Save version info
        self.history[self.version] = {
            'timestamp': pd.Timestamp.now(),
            'params': self.kwargs,
            'feature_importance': dict(zip(
                X.columns,
                self.model.coef_
            ))
        }
        
        self.version += 1
        return self
```

### 2. Online Learning
```python
class OnlineRegularizedModel:
    """Online learning with regularization"""
    def __init__(self, alpha=1.0, batch_size=32):
        self.alpha = alpha
        self.batch_size = batch_size
        self.model = SGDRegressor(
            alpha=alpha,
            learning_rate='optimal',
            penalty='elasticnet'
        )
    
    def partial_fit(self, X, y):
        """Update model with new data"""
        return self.model.partial_fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
```

## Next Steps 🎯

Ready to see regularization in action? Continue to [Applications](5-applications.md) to explore real-world use cases!
