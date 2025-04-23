# Model Diagnostics: Validating Your Regression Models

Welcome to the world of model diagnostics! This guide will help you ensure your regression models are reliable and their assumptions are met. Good model diagnostics are crucial for making valid inferences and predictions.

## Why Model Diagnostics Matter

Model diagnostics help you:

1. Validate model assumptions
2. Identify potential problems
3. Ensure reliable predictions
4. Make valid statistical inferences

## Key Assumptions to Check

### 1. Linearity

- The relationship between predictors and outcome should be linear
- Check using residual plots
- Look for non-random patterns

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def check_linearity(model, X, y):
    # Get predictions
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()
```

### 2. Independence of Errors

- Residuals should be independent
- No patterns over time or space
- Check using Durbin-Watson test

```python
from statsmodels.stats.stattools import durbin_watson

def check_independence(residuals):
    # Durbin-Watson test
    dw_statistic = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_statistic:.2f}")
    print("Values close to:")
    print("2.0 suggest no autocorrelation")
    print("<1.0 suggest positive autocorrelation")
    print(">3.0 suggest negative autocorrelation")
```

### 3. Homoscedasticity

- Constant variance of residuals
- Check using scale-location plots
- Look for fan or funnel patterns

```python
def check_homoscedasticity(model, X, y):
    # Get predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, np.abs(residuals))
    plt.xlabel('Fitted values')
    plt.ylabel('|Residuals|')
    plt.title('Scale-Location Plot')
    plt.show()
```

### 4. Normality of Residuals

- Residuals should follow normal distribution
- Check using Q-Q plots and statistical tests
- Consider transformations if needed

```python
def check_normality(residuals):
    # Create Q-Q plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(residuals, bins=30, density=True, alpha=0.7)
    ax1.set_title('Histogram of Residuals')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Shapiro-Wilk test
    stat, p_value = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test p-value: {p_value:.4f}")
    print("If p-value < 0.05, residuals may not be normally distributed")
```

## Influence Measures

### 1. Cook's Distance

- Identifies influential observations
- Measures impact of removing each point
- Values > 4/n are potentially influential

```python
def calculate_cooks_distance(model, X, y):
    # Get predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Calculate leverage
    hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diagonal(hat_matrix)
    
    # Calculate Cook's distance
    n = len(y)
    p = X.shape[1]
    mse = np.sum(residuals**2) / (n - p)
    cooks_d = (residuals**2 * leverage) / (p * mse * (1 - leverage)**2)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(cooks_d)), cooks_d, markerfmt='ro')
    plt.axhline(y=4/n, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Observation')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance Plot")
    plt.legend()
    plt.show()
    
    return cooks_d
```

### 2. Leverage Points

- Observations with extreme predictor values
- Check using hat values
- High leverage doesn't always mean high influence

```python
def check_leverage(X):
    # Calculate hat values
    hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diagonal(hat_matrix)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(leverage)), leverage, markerfmt='bo')
    plt.axhline(y=2*X.shape[1]/len(X), color='r', linestyle='--', label='Threshold')
    plt.xlabel('Observation')
    plt.ylabel('Leverage')
    plt.title('Leverage Plot')
    plt.legend()
    plt.show()
    
    return leverage
```

### 3. DFBETAS

- Measures impact on regression coefficients
- Identifies observations affecting specific coefficients
- Values > 2/√n are concerning

```python
def calculate_dfbetas(model, X, y):
    n = len(y)
    p = X.shape[1]
    dfbetas = np.zeros((n, p))
    
    # Calculate DFBETAS for each observation and predictor
    for i in range(n):
        # Fit model without observation i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model_i = model.__class__()
        model_i.fit(X[mask], y[mask])
        
        # Calculate difference in coefficients
        diff = model.coef_ - model_i.coef_
        dfbetas[i] = diff / np.sqrt(np.diagonal(np.linalg.inv(X.T @ X)))
    
    return dfbetas
```

## Comprehensive Diagnostic Function

Here's a function that combines all diagnostics:

```python
def run_diagnostics(model, X, y):
    """
    Run comprehensive diagnostics on a regression model.
    
    Parameters:
    model: fitted sklearn regression model
    X: feature matrix
    y: target variable
    """
    # Get predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    print("=== Model Diagnostics ===\n")
    
    # 1. Linearity
    print("Checking linearity...")
    check_linearity(model, X, y)
    
    # 2. Independence
    print("\nChecking independence...")
    check_independence(residuals)
    
    # 3. Homoscedasticity
    print("\nChecking homoscedasticity...")
    check_homoscedasticity(model, X, y)
    
    # 4. Normality
    print("\nChecking normality...")
    check_normality(residuals)
    
    # 5. Influence measures
    print("\nCalculating influence measures...")
    cooks_d = calculate_cooks_distance(model, X, y)
    leverage = check_leverage(X)
    dfbetas = calculate_dfbetas(model, X, y)
    
    # Summary of potential issues
    print("\n=== Summary of Potential Issues ===")
    print(f"Number of high leverage points: {sum(leverage > 2*X.shape[1]/len(X))}")
    print(f"Number of influential points (Cook's D): {sum(cooks_d > 4/len(y))}")
    print(f"Number of large DFBETAS: {sum(np.abs(dfbetas) > 2/np.sqrt(len(y)))}")
    
    return {
        'residuals': residuals,
        'cooks_distance': cooks_d,
        'leverage': leverage,
        'dfbetas': dfbetas
    }
```

## Practice Exercise

Try this hands-on exercise:

```python
# Generate sample data
np.random.seed(42)
n_samples = 100

# Create predictors with an outlier
X = np.random.normal(0, 1, (n_samples, 2))
X[0] = [5, 5]  # Add an outlier

# Create response with non-constant variance
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, np.abs(X[:, 0]), n_samples)

# Fit model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Run diagnostics
diagnostics = run_diagnostics(model, X, y)

# Your tasks:
# 1. Interpret the diagnostic plots
# 2. Identify potential problems
# 3. Suggest improvements
# 4. Implement solutions
# 5. Re-run diagnostics to verify improvements
```

## Common Problems and Solutions

1. **Non-linearity**
   - Transform variables (log, square root, etc.)
   - Consider polynomial terms
   - Use non-linear models

2. **Heteroscedasticity**
   - Transform response variable
   - Use weighted least squares
   - Consider robust regression

3. **Non-normal Residuals**
   - Transform response variable
   - Use robust regression
   - Consider non-parametric methods

4. **Autocorrelation**
   - Add time-related predictors
   - Use time series models
   - Consider GLS (Generalized Least Squares)

5. **Influential Points**
   - Investigate unusual observations
   - Consider robust regression
   - Document and justify any removals

## Key Takeaways

1. Always check model assumptions
2. Use multiple diagnostic tools
3. Consider the context when interpreting results
4. Document any violations and solutions
5. Be transparent about limitations

## Next Steps

Now that you understand model diagnostics, you can:

1. Apply these techniques to your own models
2. Learn about robust regression methods
3. Explore advanced diagnostic techniques
4. Study remedial measures for assumption violations

## Additional Resources

- [STHDA Regression Diagnostics](https://www.sthda.com/english/articles/39-regression-model-diagnostics/)
- [Penn State Statistics](https://online.stat.psu.edu/stat504/lesson/7/7.2)
- [UCLA Stats](https://stats.oarc.ucla.edu/stata/webbooks/reg/chapter2/stata-webbooksregressionwith-statachapter-2-regression-diagnostics/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Perplexity AI](https://www.perplexity.ai/) - For quick statistical questions and clarifications
