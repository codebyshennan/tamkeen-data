# Hyperparameter Tuning

Imagine finding the perfect recipe - you need to adjust ingredients and cooking time to get it just right. Hyperparameter tuning is similar: we adjust model settings to find the optimal combination! Let's explore different tuning strategies. 🎛️

## Understanding Hyperparameter Tuning 🎯

Hyperparameter tuning helps us:
1. Optimize model performance
2. Prevent overfitting/underfitting
3. Find best model configuration
4. Save computational resources

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Grid Search 📊

Systematically work through multiple combinations of parameter tunes:

```python
from sklearn.model_selection import GridSearchCV

def visualize_grid_search():
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.1, 1, 10]
    }
    
    # Create model
    svm = SVC(kernel='rbf')
    
    # Perform grid search
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Create visualization
    C_values = param_grid['C']
    gamma_values = param_grid['gamma']
    scores = grid_search.cv_results_['mean_test_score']
    scores = scores.reshape(len(C_values), len(gamma_values))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(scores, interpolation='nearest', cmap='viridis')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_values)), gamma_values)
    plt.yticks(np.arange(len(C_values)), C_values)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.title('Grid Search Scores')
    
    # Add score annotations
    for i in range(len(C_values)):
        for j in range(len(gamma_values)):
            plt.text(j, i, f"{scores[i, j]:.3f}",
                    ha="center", va="center")
    
    plt.show()
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

visualize_grid_search()
```

## Random Search 🎲

Randomly sample from parameter space:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def visualize_random_search():
    # Define parameter distributions
    param_dist = {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.1, 10)
    }
    
    # Create model
    svm = SVC(kernel='rbf')
    
    # Perform random search
    random_search = RandomizedSearchCV(
        svm, param_dist, n_iter=20, cv=5, random_state=42
    )
    random_search.fit(X_train, y_train)
    
    # Plot results
    results = random_search.cv_results_
    C_values = results['param_C']
    gamma_values = results['param_gamma']
    scores = results['mean_test_score']
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(C_values, gamma_values, c=scores,
                         cmap='viridis', s=100)
    plt.colorbar(scatter)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('C')
    plt.ylabel('gamma')
    plt.title('Random Search Results')
    plt.show()
    
    print("Best parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)

visualize_random_search()
```

## Bayesian Optimization 🧠

Use probabilistic model to guide search:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def bayesian_optimization_demo():
    # Create simple objective function
    def objective(x):
        return -(x - 2) ** 2 + 10
    
    # Generate initial data
    X = np.linspace(0, 4, 5).reshape(-1, 1)
    y = objective(X)
    
    # Fit GP
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gpr.fit(X, y)
    
    # Plot results
    x_plot = np.linspace(0, 4, 100).reshape(-1, 1)
    y_pred, sigma = gpr.predict(x_plot, return_std=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, objective(x_plot), 'r:', label='True')
    plt.plot(X, y, 'r.', markersize=10, label='Observations')
    plt.plot(x_plot, y_pred, 'b-', label='Prediction')
    plt.fill_between(x_plot.ravel(), 
                    y_pred - 1.96 * sigma,
                    y_pred + 1.96 * sigma,
                    alpha=0.2)
    plt.legend()
    plt.title('Bayesian Optimization with Gaussian Process')
    plt.show()

bayesian_optimization_demo()
```

## Real-World Example: XGBoost Tuning 🌳

```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Create regression dataset
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3, 5]
}

# Create model
xgb_model = xgb.XGBRegressor()

# Perform grid search with early stopping
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1
)

grid_search.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

# Print results
print("Best parameters:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)  # Convert back to MSE
```

## Advanced Techniques 🚀

### 1. Learning Curves
```python
def plot_learning_curves(estimator, X, y):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=5
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-',
             label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-',
             label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

### 2. Parameter Importance
```python
def analyze_parameter_importance(grid_results):
    results = pd.DataFrame(grid_results.cv_results_)
    
    # Calculate parameter importance
    param_cols = [col for col in results.columns 
                 if col.startswith('param_')]
    
    importance = {}
    for param in param_cols:
        param_values = results[param].unique()
        scores = []
        for value in param_values:
            score = results[results[param] == value]['mean_test_score'].mean()
            scores.append(score)
        importance[param] = np.std(scores)
    
    # Plot importance
    plt.figure(figsize=(10, 5))
    plt.bar(importance.keys(), importance.values())
    plt.xticks(rotation=45)
    plt.title('Parameter Importance')
    plt.tight_layout()
    plt.show()
```

### 3. Custom Scoring
```python
from sklearn.metrics import make_scorer

def custom_score(y_true, y_pred):
    # Example: Weighted combination of metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return -0.7 * mse - 0.3 * mae

custom_scorer = make_scorer(custom_score)

# Use in grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5
)
```

## Best Practices 🌟

### 1. Parameter Space Design
```python
# Log-scale for certain parameters
param_grid = {
    'C': np.logspace(-3, 3, 7),
    'gamma': np.logspace(-3, 3, 7)
}

# Categorical parameters
param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'degree': [2, 3, 4] if 'poly' in param_grid['kernel'] else [1]
}
```

### 2. Resource Management
```python
def efficient_search(X, y, param_grid, n_jobs=-1):
    # Use parallel processing
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=n_jobs,
        pre_dispatch='2*n_jobs'
    )
    
    # Monitor memory usage
    with joblib.parallel_backend('multiprocessing'):
        search.fit(X, y)
    
    return search
```

### 3. Validation Strategy
```python
from sklearn.model_selection import RepeatedStratifiedKFold

# More robust validation
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy'
)
```

## Common Pitfalls and Solutions 🚧

1. **Overfitting to Validation Set**
   - Use nested cross-validation
   - Hold out final test set
   - Monitor learning curves

2. **Computational Cost**
   - Start with broad search
   - Use random search first
   - Implement early stopping

3. **Parameter Interactions**
   - Consider joint distributions
   - Use Bayesian optimization
   - Analyze parameter importance

## Next Steps

Now that you understand hyperparameter tuning, let's explore [Scikit-learn Pipelines](./sklearn-pipelines.md) to streamline your ML workflow!
