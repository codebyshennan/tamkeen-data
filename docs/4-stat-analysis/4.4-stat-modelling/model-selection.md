# Model Selection

## Introduction

Model selection is the process of choosing the best statistical model from a set of candidate models. It's a crucial step in the data analysis pipeline that helps us find the right balance between model complexity and predictive performance.

### Why Model Selection Matters

Imagine you're trying to predict house prices. You could use:

1. A simple linear model (one feature)
2. A multiple regression model (several features)
3. A complex polynomial model (many features and interactions)

How do you choose the best one? That's where model selection comes in!

### Real-world Examples

1. **Medical Diagnosis**
   - Simple model: Using only age to predict disease risk
   - Complex model: Using age, weight, blood pressure, family history, etc.
   - Need to balance accuracy with interpretability

2. **Marketing Campaigns**
   - Basic model: Customer demographics only
   - Advanced model: Demographics + purchase history + browsing behavior
   - Must consider cost of data collection vs. predictive power

3. **Financial Forecasting**
   - Simple model: Historical averages
   - Complex model: Multiple economic indicators
   - Need to avoid overfitting to past trends

## Understanding Model Complexity

### The Bias-Variance Tradeoff

Let's visualize the tradeoff between model complexity and error:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_bias_variance_tradeoff():
    """Visualize the bias-variance tradeoff"""
    complexity = np.linspace(0, 10, 100)
    
    # Simulate error components
    bias = 1 / (complexity + 1)
    variance = complexity / 10
    total_error = bias + variance
    
    plt.figure(figsize=(10, 6))
    plt.plot(complexity, bias, 'b-', label='Bias')
    plt.plot(complexity, variance, 'r-', label='Variance')
    plt.plot(complexity, total_error, 'g-', label='Total Error')
    
    # Mark optimal complexity
    optimal = complexity[np.argmin(total_error)]
    plt.axvline(x=optimal, color='k', linestyle='--', 
                label=f'Optimal Complexity ({optimal:.1f})')
    
    plt.xlabel('Model Complexity')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.savefig('bias_variance_tradeoff.png')
    plt.close()
```

### Overfitting vs Underfitting

Let's see what happens when we use models that are too simple or too complex:

```python
def demonstrate_overfitting_underfitting():
    """Show examples of overfitting and underfitting"""
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, 100)
    
    # Fit different models
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    
    degrees = [1, 3, 10]
    plt.figure(figsize=(15, 5))
    
    for i, degree in enumerate(degrees, 1):
        plt.subplot(1, 3, i)
        
        # Fit model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x.reshape(-1, 1), y)
        
        # Plot
        x_plot = np.linspace(0, 10, 1000)
        y_plot = model.predict(x_plot.reshape(-1, 1))
        
        plt.scatter(x, y, alpha=0.5, label='Data')
        plt.plot(x_plot, y_plot, 'r-', label=f'Degree {degree}')
        
        if degree == 1:
            plt.title('Underfitting (Too Simple)')
        elif degree == 3:
            plt.title('Good Fit')
        else:
            plt.title('Overfitting (Too Complex)')
        
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('overfitting_underfitting.png')
    plt.close()
```

## Model Selection Techniques

### 1. Train-Test Split

The simplest way to evaluate a model is to split your data into training and testing sets:

```python
def train_test_split_example(X, y, test_size=0.2):
    """Demonstrate train-test split"""
    from sklearn.model_selection import train_test_split
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
```

### 2. Cross-Validation

A more robust approach is k-fold cross-validation:

```python
def cross_validation_example(X, y, k=5):
    """Demonstrate k-fold cross-validation"""
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

### 3. Information Criteria

For model comparison, we can use information criteria like AIC and BIC:

```python
def compare_models_aic_bic(X, y, models):
    """Compare models using AIC and BIC"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    results = []
    
    for name, model in models.items():
        # Fit model
        model.fit(X, y)
        
        # Calculate metrics
        n = len(y)
        k = X.shape[1] + 1  # +1 for intercept
        mse = mean_squared_error(y, model.predict(X))
        
        # Calculate AIC and BIC
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)
        
        results.append({
            'model': name,
            'AIC': aic,
            'BIC': bic,
            'MSE': mse
        })
    
    return pd.DataFrame(results)
```

## Feature Selection Methods

### 1. Forward Selection

Start with no features and add them one by one:

```python
def forward_selection(X, y, max_features=None):
    """Implement forward feature selection"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    
    selected_features = []
    best_scores = []
    
    while len(selected_features) < max_features:
        best_score = float('inf')
        best_feature = None
        
        # Try adding each remaining feature
        for feature in range(n_features):
            if feature not in selected_features:
                features = selected_features + [feature]
                model = LinearRegression()
                model.fit(X[:, features], y)
                score = mean_squared_error(y, model.predict(X[:, features]))
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
        
        selected_features.append(best_feature)
        best_scores.append(best_score)
    
    return selected_features, best_scores
```

### 2. Backward Elimination

Start with all features and remove them one by one:

```python
def backward_elimination(X, y, min_features=1):
    """Implement backward feature elimination"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    n_features = X.shape[1]
    selected_features = list(range(n_features))
    best_scores = []
    
    while len(selected_features) > min_features:
        best_score = float('inf')
        worst_feature = None
        
        # Try removing each feature
        for feature in selected_features:
            features = [f for f in selected_features if f != feature]
            model = LinearRegression()
            model.fit(X[:, features], y)
            score = mean_squared_error(y, model.predict(X[:, features]))
            
            if score < best_score:
                best_score = score
                worst_feature = feature
        
        selected_features.remove(worst_feature)
        best_scores.append(best_score)
    
    return selected_features, best_scores
```

## Practical Tips

1. **Start Simple**
   - Begin with a basic model
   - Add complexity only if needed
   - Document your model selection process

2. **Use Multiple Methods**
   - Combine different selection techniques
   - Look for consensus among methods
   - Consider both statistical and practical significance

3. **Validate Thoroughly**
   - Use cross-validation
   - Check on multiple metrics
   - Test on different data splits

4. **Consider Tradeoffs**
   - Accuracy vs. interpretability
   - Training time vs. prediction time
   - Data collection cost vs. model performance

## Practice Exercise

Try building a model to predict student performance. Consider:

1. Which features are most important?
2. How complex should your model be?
3. How will you validate your model?
4. What metrics will you use to evaluate performance?

## Additional Resources

- [Scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [Introduction to Statistical Learning](https://www.statlearning.com/) (Chapter 6)
- [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) (Chapter 7)

Remember: The best model is not always the most complex one. Focus on finding the right balance between simplicity and performance!
