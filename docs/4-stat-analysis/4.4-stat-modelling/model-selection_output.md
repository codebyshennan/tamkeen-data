# Code Execution Output: model-selection.md

This file contains the output from running the code blocks in `model-selection.md`.


### Code Block 1
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def plot_bias_variance_tradeoff():
    """Visualize the bias-variance tradeoff"""
    np.random.seed(42)
    
    # Generate x-axis values for plotting theoretical curves
    complexity = np.linspace(0, 10, 100)
    
    # Simulate error components - these follow typical patterns in ML
    bias = 1 / (complexity + 0.5)  # Decreases with complexity
    variance = complexity / 10  # Increases with complexity
    total_error = bias + variance  # Total error is the sum
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(complexity, bias, 'b-', label='Bias', linewidth=2)
    plt.plot(complexity, variance, 'r-', label='Variance', linewidth=2)
    plt.plot(complexity, total_error, 'g-', label='Total Error', linewidth=2.5)
    
    # Mark optimal complexity
    optimal = complexity[np.argmin(total_error)]
    plt.axvline(x=optimal, color='k', linestyle='--', 
                label=f'Optimal Complexity ({optimal:.1f})')
    
    # Add annotations
    plt.annotate('Underfitting Region', xy=(1, 1.2), xytext=(1, 1.4),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Overfitting Region', xy=(8, 1.2), xytext=(8, 1.4),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('Model Complexity')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.savefig('bias_variance_tradeoff.png')
    plt.show()

# Execute the function
plot_bias_variance_tradeoff()

```



### Code Block 2
```python
def demonstrate_overfitting_underfitting():
    """Show examples of overfitting and underfitting with a visualization"""
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = np.sin(x) * x  # True underlying pattern
    y = y_true + np.random.normal(0, 2, 100)  # Add noise
    
    X = x.reshape(-1, 1)  # Reshape for scikit-learn
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit different models
    degrees = [1, 3, 15]  # Linear, good fit, overfitted
    plt.figure(figsize=(15, 5))
    
    for i, degree in enumerate(degrees, 1):
        # Create and fit model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Plot
        plt.subplot(1, 3, i)
        
        # Plot training and test data
        plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
        plt.scatter(X_test, y_test, alpha=0.5, label='Test data', color='red')
        
        # Plot true function and model predictions
        x_plot = np.linspace(0, 10, 1000).reshape(-1, 1)
        y_plot_true = np.sin(x_plot) * x_plot
        y_plot_pred = model.predict(x_plot)
        
        plt.plot(x_plot, y_plot_true, 'g-', lw=2, label='True function')
        plt.plot(x_plot, y_plot_pred, 'b--', lw=2, label=f'Degree {degree} model')
        
        if degree == 1:
            plt.title(f'Underfitting (Too Simple)\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')
        elif degree == 3:
            plt.title(f'Good Fit\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')
        else:
            plt.title(f'Overfitting (Too Complex)\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')
        
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('overfitting_underfitting.png')
    plt.show()

# Execute the function
demonstrate_overfitting_underfitting()

```



### Code Block 3
```python
def train_test_split_example(X, y, test_size=0.2):
    """Demonstrate train-test split with a visualization"""
    np.random.seed(42)
    
    # Generate synthetic data if not provided
    if X is None or y is None:
        X = np.random.normal(0, 1, (100, 1))
        y = 3*X.squeeze() + np.random.normal(0, 1, 100)
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Fit a model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Visualize the split and predictions
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
    plt.scatter(X_test, y_test, alpha=0.5, label='Test data', color='red')
    
    # Plot prediction line
    x_range = np.linspace(min(X.min(), X_test.min()), max(X.max(), X_test.max()), 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, 'g-', lw=2, label='Model predictions')
    
    plt.title('Train-Test Split Visualization')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_test_split.png')
    plt.show()
    
    return X_train, X_test, y_train, y_test, model

# Generate some sample data
np.random.seed(42)
X_sample = np.random.normal(0, 1, (100, 1))
y_sample = 3*X_sample.squeeze() + np.random.normal(0, 1, 100)

# Execute the function
X_train, X_test, y_train, y_test, model = train_test_split_example(X_sample, y_sample)

```

Output:
```
Training MSE: 0.8864
Test MSE: 0.8728

```



### Code Block 4
```python
Training MSE: 0.9425
Test MSE: 1.0126

```

Output:
```
Error: invalid syntax (<string>, line 1)

```



### Code Block 5
```python
def cross_validation_example(X, y, k=5):
    """Demonstrate k-fold cross-validation with a visualization"""
    from sklearn.model_selection import KFold
    
    # Generate synthetic data if not provided
    if X is None or y is None:
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))
        y = 3*X.squeeze() + np.random.normal(0, 1, 100)
    
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_indices = list(kf.split(X))
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Placeholder for scores
    scores = []
    
    # Loop through each fold
    for i, (train_index, test_index) in enumerate(fold_indices):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        scores.append(score)
        
        # Plot this fold
        plt.subplot(2, 3, i+1)
        plt.scatter(X, y, alpha=0.2, color='gray', label='All data')
        plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
        plt.scatter(X_test, y_test, alpha=0.5, label='Validation data', color='red')
        
        # Plot prediction line
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_range = model.predict(x_range)
        plt.plot(x_range, y_range, 'g-', lw=2, label='Model')
        
        plt.title(f'Fold {i+1}: MSE = {score:.4f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.grid(True)
        if i == 0:  # Only show legend for first plot
            plt.legend(loc='best')
    
    # Final subplot with average results
    plt.subplot(2, 3, k+1)
    plt.bar(range(1, k+1), scores, alpha=0.7)
    plt.axhline(y=np.mean(scores), color='r', linestyle='--', 
               label=f'Mean MSE: {np.mean(scores):.4f}')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.title('Cross-Validation Results')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cross_validation.png')
    plt.show()
    
    return np.mean(scores), np.std(scores)

# Execute the function
cv_mean, cv_std = cross_validation_example(X_sample, y_sample)

```



### Code Block 6
```python
Mean MSE: 0.9836
Standard Deviation: 0.0423

```

Output:
```
Error: invalid syntax (<string>, line 1)

```



### Code Block 7
```python
def compare_models_aic_bic(X, y, max_degree=5):
    """Compare models using AIC and BIC"""
    # Generate synthetic data if not provided
    if X is None or y is None:
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.sin(X.ravel()) + np.random.normal(0, 0.2, 100)
    
    n = len(y)  # Sample size
    results = []
    
    # Try polynomial models of different degrees
    for degree in range(1, max_degree + 1):
        # Transform features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # Calculate metrics
        k = X_poly.shape[1]  # Number of parameters
        mse = mean_squared_error(y, y_pred)
        
        # Calculate AIC and BIC
        # Note: These are approximations. For exact formulas, we need the log-likelihood
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)
        
        results.append({
            'degree': degree,
            'num_params': k,
            'MSE': mse,
            'AIC': aic,
            'BIC': bic
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['degree'], df['AIC'], 'b-o', label='AIC')
    plt.plot(df['degree'], df['BIC'], 'r-s', label='BIC')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Information Criterion')
    plt.title('Model Selection using Information Criteria')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df['degree'], df['MSE'], 'g-o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Error vs. Model Complexity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('information_criteria.png')
    plt.show()
    
    # Find best models
    best_aic = df.loc[df['AIC'].idxmin()]
    best_bic = df.loc[df['BIC'].idxmin()]
    
    print(f"Best model according to AIC: Degree {best_aic['degree']} polynomial")
    print(f"Best model according to BIC: Degree {best_bic['degree']} polynomial")
    
    return df

# Generate some non-linear data
np.random.seed(42)
X_nonlinear = np.linspace(0, 10, 100).reshape(-1, 1)
y_nonlinear = np.sin(X_nonlinear.ravel()) + np.random.normal(0, 0.2, 100)

# Execute the function
model_comparison = compare_models_aic_bic(X_nonlinear, y_nonlinear)

```

Output:
```
Best model according to AIC: Degree 5.0 polynomial
Best model according to BIC: Degree 5.0 polynomial

```



### Code Block 8
```python
Best model according to AIC: Degree 3 polynomial
Best model according to BIC: Degree 2 polynomial

```

Output:
```
Error: invalid syntax (<string>, line 1)

```



### Code Block 9
```python
def forward_selection(X, y, max_features=None):
    """Implement forward feature selection with visualization"""
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic multivariate data if not provided
    if X is None or y is None:
        np.random.seed(42)
        n_samples, n_features = 100, 10
        X = np.random.normal(0, 1, (n_samples, n_features))
        # Only first 3 features are truly relevant
        y = 5*X[:, 0] + 2*X[:, 1] - 3*X[:, 2] + np.random.normal(0, 1, n_samples)
    
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    selected_features = []
    best_scores_train = []
    best_scores_test = []
    all_features = list(range(n_features))
    
    # If feature names are not provided, create generic names
    if isinstance(X, np.ndarray):
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    else:
        feature_names = X.columns.tolist()
    
    # Perform forward selection
    while len(selected_features) < max_features:
        best_score = float('inf')
        best_feature = None
        
        # Try adding each remaining feature
        for feature in all_features:
            if feature not in selected_features:
                features = selected_features + [feature]
                X_train_subset = X_train[:, features]
                X_test_subset = X_test[:, features]
                
                # Train model
                model = LinearRegression()
                model.fit(X_train_subset, y_train)
                
                # Evaluate
                score = mean_squared_error(y_train, model.predict(X_train_subset))
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
        
        # Add the best feature
        selected_features.append(best_feature)
        
        # Train final model for this iteration
        model = LinearRegression()
        model.fit(X_train[:, selected_features], y_train)
        
        # Calculate scores
        train_score = mean_squared_error(y_train, model.predict(X_train[:, selected_features]))
        test_score = mean_squared_error(y_test, model.predict(X_test[:, selected_features]))
        
        best_scores_train.append(train_score)
        best_scores_test.append(test_score)
        
        print(f"Step {len(selected_features)}: Added {feature_names[best_feature]}, "
              f"Train MSE: {train_score:.4f}, Test MSE: {test_score:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(selected_features) + 1), best_scores_train, 'b-o', label='Training Error')
    plt.plot(range(1, len(selected_features) + 1), best_scores_test, 'r-o', label='Test Error')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Squared Error')
    plt.title('Error vs Number of Features')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(selected_features)), 
            [feature_names[i] for i in selected_features[::-1]], 
            align='center')
    plt.yticks(range(len(selected_features)), 
              [f'Step {len(selected_features)-i}' for i in range(len(selected_features))])
    plt.xlabel('Selected Feature')
    plt.title('Feature Selection Order')
    
    plt.tight_layout()
    plt.savefig('forward_selection.png')
    plt.show()
    
    # Determine optimal number of features based on test error
    optimal_num_features = np.argmin(best_scores_test) + 1
    optimal_features = selected_features[:optimal_num_features]
    
    print(f"\nOptimal number of features: {optimal_num_features}")
    print(f"Optimal features: {[feature_names[i] for i in optimal_features]}")
    
    return selected_features, best_scores_train, best_scores_test

# Generate multivariate data
np.random.seed(42)
n_samples, n_features = 100, 8
X_multi = np.random.normal(0, 1, (n_samples, n_features))
# Only first 3 features are truly relevant
y_multi = 5*X_multi[:, 0] + 2*X_multi[:, 1] - 3*X_multi[:, 2] + np.random.normal(0, 1, n_samples)

# Execute the function
selected, train_errors, test_errors = forward_selection(X_multi, y_multi)

```

Output:
```
Step 1: Added Feature 1, Train MSE: 9.2499, Test MSE: 23.3469
Step 2: Added Feature 3, Train MSE: 3.2018, Test MSE: 9.5926
Step 3: Added Feature 2, Train MSE: 0.7323, Test MSE: 1.1629
Step 4: Added Feature 4, Train MSE: 0.6916, Test MSE: 1.3233
Step 5: Added Feature 5, Train MSE: 0.6876, Test MSE: 1.3434
Step 6: Added Feature 7, Train MSE: 0.6839, Test MSE: 1.3159
Step 7: Added Feature 6, Train MSE: 0.6808, Test MSE: 1.3277
Step 8: Added Feature 8, Train MSE: 0.6803, Test MSE: 1.3185

Optimal number of features: 3
Optimal features: ['Feature 1', 'Feature 3', 'Feature 2']

```



### Code Block 10
```python
Step 1: Added Feature 1, Train MSE: 1.1254, Test MSE: 1.3421
Step 2: Added Feature 3, Train MSE: 0.7856, Test MSE: 0.9124
Step 3: Added Feature 2, Train MSE: 0.6723, Test MSE: 0.8976
...
Optimal number of features: 3
Optimal features: ['Feature 1', 'Feature 3', 'Feature 2']

```

Output:
```
Error: invalid syntax (<string>, line 1)

```



### Code Block 11
```python
def backward_elimination(X, y, min_features=1):
    """Implement backward feature elimination with visualization"""
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic multivariate data if not provided
    if X is None or y is None:
        np.random.seed(42)
        n_samples, n_features = 100, 10
        X = np.random.normal(0, 1, (n_samples, n_features))
        # Only first 3 features are truly relevant
        y = 5*X[:, 0] + 2*X[:, 1] - 3*X[:, 2] + np.random.normal(0, 1, n_samples)
    
    n_features = X.shape[1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # If feature names are not provided, create generic names
    if isinstance(X, np.ndarray):
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    else:
        feature_names = X.columns.tolist()
    
    # Start with all features
    selected_features = list(range(n_features))
    train_scores = []
    test_scores = []
    
    # Train initial model with all features
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate initial scores
    train_score = mean_squared_error(y_train, model.predict(X_train))
    test_score = mean_squared_error(y_test, model.predict(X_test))
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    removed_features = []
    
    # Perform backward elimination until we reach min_features
    while len(selected_features) > min_features:
        best_score = float('inf')
        worst_feature = None
        
        # Try removing each feature
        for i, feature in enumerate(selected_features):
            # Create a feature list without this feature
            features = selected_features.copy()
            features.pop(i)
            
            # Train model without this feature
            model = LinearRegression()
            model.fit(X_train[:, features], y_train)
            
            # Evaluate
            score = mean_squared_error(y_train, model.predict(X_train[:, features]))
            
            if score < best_score:
                best_score = score
                worst_feature = i
        
        # Remove the worst feature
        worst_feature_index = selected_features.pop(worst_feature)
        removed_features.append(worst_feature_index)
        
        # Train final model for this iteration
        model = LinearRegression()
        model.fit(X_train[:, selected_features], y_train)
        
        # Calculate scores
        train_score = mean_squared_error(y_train, model.predict(X_train[:, selected_features]))
        test_score = mean_squared_error(y_test, model.predict(X_test[:, selected_features]))
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"Step {len(removed_features)}: Removed {feature_names[worst_feature_index]}, "
              f"Train MSE: {train_score:.4f}, Test MSE: {test_score:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    x_axis = list(range(n_features, min_features - 1, -1))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, train_scores, 'b-o', label='Training Error')
    plt.plot(x_axis, test_scores, 'r-o', label='Test Error')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Squared Error')
    plt.title('Error vs Number of Features')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(removed_features)), 
            [feature_names[i] for i in removed_features], 
            align='center')
    plt.yticks(range(len(removed_features)), 
              [f'Step {i+1}' for i in range(len(removed_features))])
    plt.xlabel('Removed Feature')
    plt.title('Feature Elimination Order')
    
    plt.tight_layout()
    plt.savefig('backward_elimination.png')
    plt.show()
    
    # Determine optimal number of features based on test error
    optimal_num_features = n_features - np.argmin(test_scores)
    optimal_features = list(range(n_features))
    for i in removed_features[:n_features - optimal_num_features]:
        optimal_features.remove(i)
    
    print(f"\nOptimal number of features: {optimal_num_features}")
    print(f"Optimal features: {[feature_names[i] for i in optimal_features]}")
    
    return selected_features, train_scores, test_scores

# Execute the function with our multivariate data
selected_backward, train_errors_backward, test_errors_backward = backward_elimination(X_multi, y_multi)
## Practical Tips

When selecting models in practice, follow these steps:

1. **Start Simple**
   - Begin with a basic model
   - Understand your baseline performance
   - Add complexity only if needed

2. **Use Multiple Methods**
   - Combine different selection techniques
   - Look for consensus among methods
   - Consider both statistical metrics and practical utility

3. **Validate Thoroughly**
   - Always perform cross-validation
   - Check performance on multiple metrics
   - Test on different data splits or time periods

4. **Consider Tradeoffs**
   - Balance accuracy vs. interpretability
   - Consider training time vs. prediction time
   - Weigh data collection cost vs. model benefit

5. **Document Your Process**
   - Record all models tried
   - Note why certain choices were made
   - Make your selection process reproducible

### Decision Framework

Here's a practical framework for model selection:

1. **Define your goals**:
   - Is prediction accuracy the main goal?
   - Is interpretability important?
   - Are there computational constraints?

2. **Consider your data**:
   - How much data is available?
   - What's the quality of the data?
   - Are there patterns in the data that require specific model types?

3. **Start with simple models**:
   - Linear/logistic regression
   - Decision trees
   - K-nearest neighbors

4. **Gradually increase complexity**:
   - Try polynomial terms
   - Add regularization
   - Consider ensemble methods or more complex algorithms

5. **Compare systematically**:
   - Use cross-validation
   - Evaluate on appropriate metrics
   - Consider computational costs

6. **Select the final model**:
   - Choose the simplest model that meets performance requirements
   - Consider the business or research context
   - Ensure the model is practical to deploy and maintain

## Practice Exercise

Try building a model to predict student performance based on various features. Consider:

1. **Dataset**:
   - Features: study time, previous grades, attendance, etc.
   - Target: final exam score

2. **Questions to address**:
   - Which features are most important?
   - How complex should your model be?
   - How will you validate your model?
   - What metrics will you use to evaluate performance?

### Example Implementation


```

Output:
```
Error: unterminated string literal (detected at line 153) (<string>, line 153)

```

