# Implementing Regularization 💻

Let's put theory into practice! We'll explore how to implement regularization using scikit-learn, from basic usage to advanced techniques.

## Basic Implementation 🚀

### Simple Example with Ridge Regression
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Create sample data
np.random.seed(42)
X = np.random.randn(100, 20)
y = np.random.randn(100)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train ridge model
ridge = Ridge(alpha=0.1)  # alpha is the regularization strength
ridge.fit(X_train_scaled, y_train)

# Make predictions
y_pred = ridge.predict(X_test_scaled)
```

## Real-World Example: House Price Prediction 🏠

```python
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create sample dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'size': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'location_score': np.random.uniform(0, 10, n_samples),
    'noise_feature1': np.random.normal(0, 1, n_samples),
    'noise_feature2': np.random.normal(0, 1, n_samples)
})

# Create target with noise
data['price'] = (
    200 * data['size'] 
    + 50000 * data['bedrooms']
    - 1000 * data['age']
    + 20000 * data['location_score']
    + np.random.normal(0, 50000, n_samples)
)

# Prepare data
X = data.drop('price', axis=1)
y = data['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Compare different regularization methods
from sklearn.linear_model import Lasso, ElasticNet

models = {
    'Lasso': Lasso(alpha=0.1),
    'Ridge': Ridge(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Store results
    results[name] = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Coefficients': model.coef_
    }

# Plot coefficients comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(X.columns))
width = 0.25

for i, (name, result) in enumerate(results.items()):
    plt.bar(x + i*width, result['Coefficients'], 
            width, label=name)

plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Regularization Methods')
plt.xticks(x + width, X.columns, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print performance metrics
for name, result in results.items():
    print(f"\n{name}:")
    print(f"R² Score: {result['R2']:.3f}")
    print(f"RMSE: ${result['RMSE']:,.2f}")
```

## Hyperparameter Tuning 🎛️

```python
from sklearn.model_selection import GridSearchCV

# Setup parameter grid
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Perform grid search
grid_search = GridSearchCV(
    Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

print("Best alpha:", grid_search.best_params_['alpha'])
```

## Feature Selection with Lasso 🎯

```python
def select_features_lasso(X, y, alpha=0.1):
    """Select features using Lasso regularization"""
    # Train Lasso model
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[lasso.coef_ != 0]
    
    # Print results
    print("Selected features:", len(selected_features))
    for feature, coef in zip(X.columns, lasso.coef_):
        if coef != 0:
            print(f"{feature}: {coef:.4f}")
    
    return selected_features

# Use function
selected = select_features_lasso(
    pd.DataFrame(X_train, columns=X.columns), 
    y_train
)
```

## Cross-Validation Implementation 📊

```python
from sklearn.model_selection import cross_val_score

def compare_alphas(X, y, alphas=[0.001, 0.01, 0.1, 1, 10]):
    """Compare different regularization strengths"""
    results = []
    
    for alpha in alphas:
        # Create and evaluate models
        ridge_scores = cross_val_score(
            Ridge(alpha=alpha), X, y, 
            cv=5, scoring='neg_mean_squared_error'
        )
        lasso_scores = cross_val_score(
            Lasso(alpha=alpha), X, y,
            cv=5, scoring='neg_mean_squared_error'
        )
        elastic_scores = cross_val_score(
            ElasticNet(alpha=alpha, l1_ratio=0.5), X, y,
            cv=5, scoring='neg_mean_squared_error'
        )
        
        # Store results
        results.append({
            'alpha': alpha,
            'ridge_rmse': np.sqrt(-ridge_scores.mean()),
            'lasso_rmse': np.sqrt(-lasso_scores.mean()),
            'elastic_rmse': np.sqrt(-elastic_scores.mean())
        })
    
    return pd.DataFrame(results)

# Compare alphas
results_df = compare_alphas(X_scaled, y)
print(results_df)
```

## Model Evaluation Functions 📈

```python
def evaluate_regularized_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive evaluation of regularized model"""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'n_nonzero_coef': np.sum(model.coef_ != 0)
    }
    
    # Print results
    print("Training R²:", results['train_r2'])
    print("Testing R²:", results['test_r2'])
    print("Training RMSE:", results['train_rmse'])
    print("Testing RMSE:", results['test_rmse'])
    print("Non-zero coefficients:", results['n_nonzero_coef'])
    
    return results
```

## Best Practices 🌟

### 1. Feature Scaling
```python
# Always scale features before regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Model Selection
```python
def select_best_model(X, y):
    """Select best regularization method"""
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet()
    }
    
    best_score = float('-inf')
    best_model = None
    
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=5, scoring='r2'
        )
        avg_score = scores.mean()
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = name
    
    return best_model, best_score
```

### 3. Pipeline Creation
```python
from sklearn.pipeline import Pipeline

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# Use pipeline in cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)
```

## Next Steps 🚀

Ready to explore advanced techniques? Continue to [Advanced Topics](4-advanced.md)!
