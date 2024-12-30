# Model Diagnostics

## Introduction
Model diagnostics are crucial for validating regression assumptions and ensuring reliable results. This guide covers essential diagnostic techniques and remedies for common issues.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

def generate_sample_data(n=100, seed=42):
    """Generate sample data with various diagnostic issues"""
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, 2))
    
    # Add heteroscedasticity
    noise = np.random.normal(0, np.exp(X[:, 0]), n)
    y = 2 * X[:, 0] + 3 * X[:, 1] + noise
    
    # Add outliers
    y[0] = y[0] + 10  # Add outlier
    
    return pd.DataFrame({
        'X1': X[:, 0],
        'X2': X[:, 1],
        'y': y
    })
```

## Checking Model Assumptions

### 1. Linearity
```python
def check_linearity(model):
    """Check linearity assumption"""
    fitted_vals = model.fittedvalues
    residuals = model.resid
    
    plt.figure(figsize=(12, 4))
    
    # Residual vs Fitted plot
    plt.subplot(121)
    plt.scatter(fitted_vals, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    
    # Component plus residual plots
    plt.subplot(122)
    sm.graphics.plot_ccpr(model, 'X1')
    plt.title('Component Plus Residual Plot')
    
    plt.tight_layout()
    plt.savefig('linearity_check.png')
    plt.close()
    
    # Ramsey RESET test
    reset_test = sm.stats.diagnostic.linear_reset(model)
    
    return {
        'reset_test_f': reset_test[0],
        'reset_test_p': reset_test[1],
        'linear': reset_test[1] > 0.05
    }
```

### 2. Independence
```python
def check_independence(model):
    """Check independence assumption"""
    residuals = model.resid
    
    # Durbin-Watson test
    dw_stat = sm.stats.stattools.durbin_watson(residuals)
    
    # Plot residuals over time/order
    plt.figure(figsize=(8, 4))
    plt.plot(residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals Over Order')
    plt.xlabel('Order')
    plt.ylabel('Residuals')
    plt.savefig('independence_check.png')
    plt.close()
    
    return {
        'durbin_watson': dw_stat,
        'independent': 1.5 < dw_stat < 2.5
    }
```

### 3. Normality
```python
def check_normality(model):
    """Check normality of residuals"""
    residuals = model.resid
    
    plt.figure(figsize=(12, 4))
    
    # Q-Q plot
    plt.subplot(121)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Histogram
    plt.subplot(122)
    plt.hist(residuals, bins=30, density=True, alpha=0.7)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('normality_check.png')
    plt.close()
    
    # Statistical tests
    shapiro_test = stats.shapiro(residuals)
    normaltest = stats.normaltest(residuals)
    
    return {
        'shapiro_stat': shapiro_test[0],
        'shapiro_p': shapiro_test[1],
        'dagostino_stat': normaltest[0],
        'dagostino_p': normaltest[1],
        'normal': shapiro_test[1] > 0.05
    }
```

### 4. Homoscedasticity
```python
def check_homoscedasticity(model):
    """Check homoscedasticity assumption"""
    fitted_vals = model.fittedvalues
    residuals = model.resid
    
    plt.figure(figsize=(8, 4))
    plt.scatter(fitted_vals, np.abs(residuals), alpha=0.5)
    plt.title('Scale-Location Plot')
    plt.xlabel('Fitted values')
    plt.ylabel('|Residuals|')
    plt.savefig('homoscedasticity_check.png')
    plt.close()
    
    # Breusch-Pagan test
    bp_test = het_breuschpagan(residuals, model.model.exog)
    
    return {
        'bp_stat': bp_test[0],
        'bp_p': bp_test[1],
        'homoscedastic': bp_test[1] > 0.05
    }
```

## Influence Analysis

### 1. Outliers and Leverage Points
```python
def analyze_influence(model):
    """Analyze influential observations"""
    influence = model.get_influence()
    
    # Studentized residuals
    student_resid = influence.resid_studentized_internal
    
    # Leverage
    leverage = influence.hat_matrix_diag
    
    # Cook's distance
    cooks_d = influence.cooks_distance[0]
    
    plt.figure(figsize=(12, 4))
    
    # Leverage vs Studentized Residuals
    plt.subplot(121)
    plt.scatter(leverage, student_resid, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Leverage vs Studentized Residuals')
    plt.xlabel('Leverage')
    plt.ylabel('Studentized Residuals')
    
    # Cook's distance plot
    plt.subplot(122)
    plt.stem(range(len(cooks_d)), cooks_d, markerfmt=',')
    plt.title("Cook's Distance")
    plt.xlabel('Observation')
    plt.ylabel("Cook's Distance")
    
    plt.tight_layout()
    plt.savefig('influence_analysis.png')
    plt.close()
    
    return pd.DataFrame({
        'studentized_residuals': student_resid,
        'leverage': leverage,
        'cooks_distance': cooks_d,
        'outlier': np.abs(student_resid) > 3,
        'high_leverage': leverage > 2 * model.model.exog.shape[1] / len(leverage),
        'influential': cooks_d > 4 / len(cooks_d)
    })
```

### 2. Multicollinearity
```python
def check_multicollinearity(X):
    """Check for multicollinearity"""
    # Correlation matrix
    corr_matrix = X.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # VIF
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                       for i in range(X_with_const.shape[1])]
    
    return {
        'correlation_matrix': corr_matrix,
        'vif_data': vif_data,
        'high_vif_vars': vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
    }
```

## Remedial Measures

### 1. Handling Non-linearity
```python
def handle_nonlinearity(data, target, features):
    """Handle non-linear relationships"""
    transformed_data = data.copy()
    
    for feature in features:
        # Try common transformations
        transformed_data[f'{feature}_squared'] = data[feature]**2
        transformed_data[f'{feature}_log'] = np.log1p(data[feature] - data[feature].min() + 1)
        transformed_data[f'{feature}_sqrt'] = np.sqrt(data[feature] - data[feature].min())
    
    # Fit models with different transformations
    results = {}
    for feature in features:
        # Original
        model_orig = sm.OLS(data[target], 
                          sm.add_constant(data[feature])).fit()
        
        # Squared
        model_sq = sm.OLS(data[target], 
                         sm.add_constant(transformed_data[f'{feature}_squared'])).fit()
        
        # Log
        model_log = sm.OLS(data[target], 
                          sm.add_constant(transformed_data[f'{feature}_log'])).fit()
        
        # Square root
        model_sqrt = sm.OLS(data[target], 
                           sm.add_constant(transformed_data[f'{feature}_sqrt'])).fit()
        
        results[feature] = {
            'original_r2': model_orig.rsquared,
            'squared_r2': model_sq.rsquared,
            'log_r2': model_log.rsquared,
            'sqrt_r2': model_sqrt.rsquared
        }
    
    return results
```

### 2. Handling Heteroscedasticity
```python
def handle_heteroscedasticity(model):
    """Apply corrections for heteroscedasticity"""
    # White's heteroscedasticity-consistent standard errors
    robust_cov = sm.stats.sandwich_covariance.cov_hc3(model)
    robust_std_err = np.sqrt(np.diag(robust_cov))
    
    # WLS estimation
    weights = 1 / (model.resid**2)
    wls_model = sm.WLS(model.model.endog, 
                      model.model.exog, 
                      weights=weights).fit()
    
    return {
        'original_std_errors': model.bse,
        'robust_std_errors': robust_std_err,
        'wls_results': wls_model
    }
```

### 3. Handling Outliers
```python
def handle_outliers(model, max_studentized_resid=3):
    """Handle outliers in regression"""
    influence = model.get_influence()
    student_resid = influence.resid_studentized_internal
    
    # Remove outliers
    clean_data = pd.DataFrame({
        'y': model.model.endog,
        'X': model.model.exog[:, 1]  # Assuming one predictor
    })
    clean_data = clean_data[np.abs(student_resid) <= max_studentized_resid]
    
    # Refit model
    clean_model = sm.OLS(clean_data['y'], 
                        sm.add_constant(clean_data['X'])).fit()
    
    return {
        'original_params': model.params,
        'clean_params': clean_model.params,
        'n_outliers': sum(np.abs(student_resid) > max_studentized_resid),
        'clean_model': clean_model
    }
```

## Practice Questions
1. What are the key assumptions of linear regression?
2. How do you identify influential observations?
3. When should you use robust regression methods?
4. What are the consequences of violating model assumptions?
5. How do you choose between different remedial measures?

## Key Takeaways
1. Always check model assumptions
2. Use multiple diagnostic tools
3. Consider the impact of influential observations
4. Apply appropriate remedial measures
5. Document diagnostic findings and actions taken
