# Polynomial Regression

## Introduction
Polynomial regression extends linear regression to capture non-linear relationships by modeling the relationship between the independent variable X and the dependent variable Y as an nth degree polynomial.

### Mathematical Foundation
The general form of a polynomial regression model of degree n is:

$$Y = \beta_0 + \beta_1X + \beta_2X^2 + ... + \beta_nX^n + \epsilon$$

where:
- Y is the dependent variable
- X is the independent variable
- β₀ is the intercept
- β₁, β₂, ..., βₙ are the coefficients
- ε is the error term, assumed to be normally distributed: ε ~ N(0, σ²)

The model can be written in matrix form as:

$$Y = XB + E$$

where:
Y = [y₁, y₂, ..., yₘ]ᵀ
X = [1, x, x², ..., xⁿ]
B = [β₀, β₁, β₂, ..., βₙ]ᵀ
E = [ε₁, ε₂, ..., εₘ]ᵀ

The coefficients are estimated using the least squares method, minimizing:

$$RSS = \sum_{i=1}^m (y_i - (\beta_0 + \beta_1x_i + \beta_2x_i^2 + ... + \beta_nx_i^n))^2$$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def generate_sample_data(n=100, seed=42):
    """Generate sample data with non-linear relationship"""
    np.random.seed(seed)
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y = 0.5 * X.ravel()**2 + 2 * X.ravel() + 1 + np.random.normal(0, 0.5, n)
    return X, y
```

## Understanding Polynomial Terms

### 1. Creating Polynomial Features
```python
def create_polynomial_features(X, degree=2):
    """Create polynomial features up to specified degree"""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Get feature names
    feature_names = ['x']
    for d in range(2, degree + 1):
        feature_names.append(f'x^{d}')
    
    return pd.DataFrame(X_poly, columns=feature_names)
```

### 2. Visualizing Polynomial Fits
```python
def plot_polynomial_fits(X, y, max_degree=4):
    """Plot data with polynomial fits of different degrees"""
    plt.figure(figsize=(15, 5))
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    
    for i, degree in enumerate([1, 2, 3, max_degree], 1):
        plt.subplot(1, 4, i)
        
        # Fit polynomial regression
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        
        # Plot
        plt.scatter(X, y, alpha=0.5)
        plt.plot(X_plot, y_plot, color='r', label=f'Degree {degree}')
        plt.title(f'Polynomial Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('polynomial_fits.png')
    plt.close()
```

## Model Building

### 1. Fitting Polynomial Regression
```python
def fit_polynomial_regression(X, y, degree=2):
    """Fit polynomial regression using statsmodels"""
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Fit model
    model = sm.OLS(y, X_poly).fit()
    
    return {
        'model': model,
        'summary': model.summary(),
        'features': poly_features.get_feature_names_out(),
        'coefficients': pd.DataFrame({
            'feature': poly_features.get_feature_names_out(),
            'coefficient': model.params
        })
    }
```

### 2. Model Selection
```python
def select_polynomial_degree(X, y, max_degree=10):
    """Select optimal polynomial degree using cross-validation"""
    from sklearn.model_selection import cross_val_score
    
    degrees = range(1, max_degree + 1)
    mean_scores = []
    std_scores = []
    
    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_scores.append(-scores.mean())
        std_scores.append(scores.std())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(degrees, mean_scores, yerr=std_scores, marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Selection: Polynomial Degree vs MSE')
    plt.savefig('degree_selection.png')
    plt.close()
    
    # Find optimal degree
    optimal_degree = degrees[np.argmin(mean_scores)]
    
    return {
        'optimal_degree': optimal_degree,
        'mse_scores': mean_scores,
        'std_scores': std_scores
    }
```

## Model Diagnostics

### 1. Residual Analysis
```python
def analyze_polynomial_residuals(model, X, y):
    """Analyze residuals of polynomial regression"""
    # Predictions
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    plt.figure(figsize=(15, 5))
    
    # Residual plot
    plt.subplot(131)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    # Q-Q plot
    plt.subplot(132)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Histogram of residuals
    plt.subplot(133)
    plt.hist(residuals, bins=30, density=True, alpha=0.7)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('polynomial_diagnostics.png')
    plt.close()
    
    return {
        'residuals': residuals,
        'normality_test': stats.normaltest(residuals),
        'homoscedasticity': stats.levene(
            residuals[y_pred < np.median(y_pred)],
            residuals[y_pred >= np.median(y_pred)]
        )
    }
```

### 2. Overfitting Detection
```python
def detect_overfitting(X, y, degrees=[1, 2, 3, 4, 5]):
    """Detect overfitting using training and validation scores"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    train_scores = []
    test_scores = []
    
    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Calculate scores
        train_scores.append(r2_score(y_train, model.predict(X_train)))
        test_scores.append(r2_score(y_test, model.predict(X_test)))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_scores, 'o-', label='Training R²')
    plt.plot(degrees, test_scores, 'o-', label='Test R²')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('Overfitting Detection')
    plt.legend()
    plt.grid(True)
    plt.savefig('overfitting_detection.png')
    plt.close()
    
    return {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'optimal_degree': degrees[np.argmax(test_scores)]
    }
```

## Making Predictions

### 1. Confidence Intervals
```python
def calculate_prediction_intervals(model, X_new, alpha=0.05):
    """Calculate prediction intervals for new observations"""
    # Get predictions
    y_pred = model.predict(X_new)
    
    # Calculate standard errors
    mse = np.sum(model.resid**2) / (len(model.resid) - len(model.params))
    var_pred = mse * (1 + np.diagonal(X_new @ np.linalg.inv(X_new.T @ X_new) @ X_new.T))
    
    # Calculate intervals
    t_value = stats.t.ppf(1 - alpha/2, model.df_resid)
    margin = t_value * np.sqrt(var_pred)
    
    return pd.DataFrame({
        'prediction': y_pred,
        'lower_bound': y_pred - margin,
        'upper_bound': y_pred + margin
    })
```

### 2. Visualization
```python
def plot_predictions(X, y, model, X_new):
    """Visualize predictions with confidence intervals"""
    # Get predictions and intervals
    predictions = calculate_prediction_intervals(model, X_new)
    
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    plt.scatter(X, y, alpha=0.5, label='Data')
    
    # Plot predictions
    plt.plot(X_new, predictions['prediction'], 'r-', label='Prediction')
    plt.fill_between(X_new.ravel(),
                    predictions['lower_bound'],
                    predictions['upper_bound'],
                    alpha=0.2, color='r',
                    label='95% Prediction Interval')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression Predictions')
    plt.legend()
    plt.savefig('polynomial_predictions.png')
    plt.close()
```

## Practice Questions
1. When should you use polynomial regression?
2. How do you choose the optimal polynomial degree?
3. What are the signs of overfitting?
4. How do you interpret polynomial coefficients?
5. What are the limitations of polynomial regression?

## Key Takeaways
1. Polynomial regression captures non-linear relationships
2. Higher degrees can lead to overfitting
3. Model selection is crucial
4. Check for overfitting using validation
5. Consider interpretability vs complexity
