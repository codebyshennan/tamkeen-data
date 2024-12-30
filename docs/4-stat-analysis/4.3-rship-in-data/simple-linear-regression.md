# Simple Linear Regression

## Introduction to Linear Regression
Linear regression models the relationship between a dependent variable (y) and an independent variable (x) by fitting a linear equation.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

def generate_sample_data(n=100, seed=42):
    """Generate sample data for regression examples"""
    np.random.seed(seed)
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + np.random.normal(0, 1, n)
    return x, y
```

## The Linear Model

### 1. Model Components
```python
def explain_model_components(x, y):
    """Demonstrate components of linear regression"""
    # Fit the model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Plot components
    plt.figure(figsize=(10, 6))
    
    # Data points
    plt.scatter(x, y, alpha=0.5, label='Data')
    
    # Regression line
    plt.plot(x, y_pred, 'r-', label='Regression Line')
    
    # Residuals
    for i in range(len(x)):
        plt.vlines(x[i], y[i], y_pred[i], 'g', alpha=0.2)
    
    plt.title('Linear Regression Components')
    plt.legend()
    plt.savefig('regression_components.png')
    plt.close()
    
    return {
        'slope': model.params[1],
        'intercept': model.params[0],
        'equation': f'y = {model.params[1]:.2f}x + {model.params[0]:.2f}'
    }
```

### 2. Fitting the Model
```python
def fit_linear_model(x, y):
    """Fit linear regression model using statsmodels"""
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    return {
        'model': model,
        'summary': model.summary(),
        'params': model.params,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'std_errors': model.bse,
        'p_values': model.pvalues
    }
```

## Model Assessment

### 1. R-squared and Adjusted R-squared
```python
def assess_model_fit(model_results):
    """Assess model fit using R-squared metrics"""
    r2 = model_results.rsquared
    adj_r2 = model_results.rsquared_adj
    
    # Interpret R-squared
    if r2 < 0.3:
        fit_quality = "poor"
    elif r2 < 0.6:
        fit_quality = "moderate"
    else:
        fit_quality = "good"
    
    return {
        'r_squared': r2,
        'adj_r_squared': adj_r2,
        'fit_quality': fit_quality,
        'interpretation': f"The model explains {r2*100:.1f}% of the variance in the data"
    }
```

### 2. Residual Analysis
```python
def analyze_residuals(model_results):
    """Analyze regression residuals"""
    residuals = model_results.resid
    fitted_values = model_results.fittedvalues
    
    plt.figure(figsize=(12, 4))
    
    # Residual plot
    plt.subplot(131)
    plt.scatter(fitted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    
    # Q-Q plot
    plt.subplot(132)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Residual histogram
    plt.subplot(133)
    plt.hist(residuals, bins=20)
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('residual_analysis.png')
    plt.close()
    
    # Statistical tests
    normality_test = stats.normaltest(residuals)
    homoscedasticity = stats.levene(residuals[fitted_values < np.median(fitted_values)],
                                  residuals[fitted_values >= np.median(fitted_values)])
    
    return {
        'normality': {
            'statistic': normality_test[0],
            'p_value': normality_test[1],
            'normal': normality_test[1] > 0.05
        },
        'homoscedasticity': {
            'statistic': homoscedasticity[0],
            'p_value': homoscedasticity[1],
            'homoscedastic': homoscedasticity[1] > 0.05
        }
    }
```

## Making Predictions

### 1. Point Predictions
```python
def make_predictions(model, X_new):
    """Make predictions with confidence intervals"""
    # Add constant if X_new is just the predictor
    if len(X_new.shape) == 1 or X_new.shape[1] == 1:
        X_new = sm.add_constant(X_new)
    
    # Get predictions
    predictions = model.get_prediction(X_new)
    
    return {
        'predictions': predictions.predicted_mean,
        'conf_int': predictions.conf_int(),
        'pred_int': predictions.pred_int()
    }
```

### 2. Confidence and Prediction Intervals
```python
def plot_predictions(model, x, y):
    """Plot regression line with confidence and prediction intervals"""
    X = sm.add_constant(x)
    
    # Get predictions and intervals
    predictions = model.get_prediction(X)
    pred_mean = predictions.predicted_mean
    conf_int = predictions.conf_int()
    pred_int = predictions.pred_int()
    
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, alpha=0.5, label='Data')
    
    # Plot regression line
    plt.plot(x, pred_mean, 'r-', label='Regression Line')
    
    # Plot confidence interval
    plt.fill_between(x, conf_int[:, 0], conf_int[:, 1], 
                    color='r', alpha=0.1, label='95% CI')
    
    # Plot prediction interval
    plt.fill_between(x, pred_int[:, 0], pred_int[:, 1],
                    color='g', alpha=0.1, label='95% PI')
    
    plt.title('Regression with Confidence and Prediction Intervals')
    plt.legend()
    plt.savefig('regression_intervals.png')
    plt.close()
```

## Model Validation

### 1. Cross-Validation
```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def cross_validate_regression(x, y, n_splits=5):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {
        'mse': [],
        'r2': []
    }
    
    for train_idx, test_idx in kf.split(x):
        # Split data
        X_train = sm.add_constant(x[train_idx])
        X_test = sm.add_constant(x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Fit model
        model = sm.OLS(y_train, X_train).fit()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        cv_results['mse'].append(mean_squared_error(y_test, y_pred))
        cv_results['r2'].append(r2_score(y_test, y_pred))
    
    return {
        'mean_mse': np.mean(cv_results['mse']),
        'std_mse': np.std(cv_results['mse']),
        'mean_r2': np.mean(cv_results['r2']),
        'std_r2': np.std(cv_results['r2'])
    }
```

### 2. Model Diagnostics
```python
def model_diagnostics(model_results):
    """Comprehensive model diagnostics"""
    # Basic statistics
    stats = {
        'r_squared': model_results.rsquared,
        'adj_r_squared': model_results.rsquared_adj,
        'f_stat': model_results.fvalue,
        'f_pvalue': model_results.f_pvalue,
        'aic': model_results.aic,
        'bic': model_results.bic
    }
    
    # Influence statistics
    influence = model_results.get_influence()
    stats.update({
        'cooks_distance': influence.cooks_distance[0],
        'leverage': influence.hat_matrix_diag
    })
    
    return stats
```

## Practice Questions
1. What are the key assumptions of linear regression?
2. How do you interpret the R-squared value?
3. When should you use confidence vs prediction intervals?
4. What are the signs of a poor model fit?
5. How do you handle violations of model assumptions?

## Key Takeaways
1. Linear regression assumes a linear relationship
2. Always check model assumptions
3. Use multiple metrics for model assessment
4. Consider both statistical and practical significance
5. Validate your model before using it for predictions
