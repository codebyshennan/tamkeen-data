# Multiple Linear Regression

## Introduction
Multiple linear regression extends simple linear regression to include multiple independent variables (predictors).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def generate_sample_data(n=100, seed=42):
    """Generate sample data for multiple regression"""
    np.random.seed(seed)
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = 0.5 * X1 + np.random.normal(0, 0.5, n)  # Correlated with X1
    y = 2 * X1 + 3 * X2 + 0.5 * X3 + np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'y': y
    })
```

## Building the Model

### 1. Model Specification
```python
def fit_multiple_regression(data, target, features):
    """
    Fit multiple regression model
    
    Parameters:
    data: DataFrame containing all variables
    target: Name of target variable
    features: List of feature names
    """
    # Prepare data
    X = sm.add_constant(data[features])
    y = data[target]
    
    # Fit model
    model = sm.OLS(y, X).fit()
    
    return {
        'model': model,
        'summary': model.summary(),
        'params': model.params,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'aic': model.aic,
        'bic': model.bic
    }
```

### 2. Feature Selection
```python
def stepwise_selection(data, target, features, threshold_in=0.05, threshold_out=0.1):
    """
    Perform stepwise feature selection
    
    Parameters:
    data: DataFrame containing all variables
    target: Name of target variable
    features: List of initial features to consider
    threshold_in: P-value threshold for inclusion
    threshold_out: P-value threshold for removal
    """
    included = []
    while True:
        changed = False
        
        # Forward step
        excluded = list(set(features) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(data[target], 
                         sm.add_constant(data[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            
        # Backward step
        model = sm.OLS(data[target], sm.add_constant(data[included])).fit()
        # Remove variables with p-value above threshold_out
        pvalues = model.pvalues.iloc[1:]  # Skip constant
        worst_pval = pvalues.max()
        
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            
        if not changed:
            break
            
    return included
```

## Model Diagnostics

### 1. Multicollinearity
```python
def check_multicollinearity(X):
    """Check for multicollinearity using VIF"""
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    
    return {
        'vif_data': vif_data,
        'high_vif': vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
    }
```

### 2. Residual Analysis
```python
def analyze_multiple_regression_residuals(model):
    """Comprehensive residual analysis for multiple regression"""
    residuals = model.resid
    fitted = model.fittedvalues
    
    plt.figure(figsize=(15, 5))
    
    # Residuals vs Fitted
    plt.subplot(131)
    plt.scatter(fitted, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    
    # Q-Q Plot
    plt.subplot(132)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Scale-Location Plot
    plt.subplot(133)
    plt.scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5)
    plt.title('Scale-Location Plot')
    plt.xlabel('Fitted Values')
    plt.ylabel('√|Residuals|')
    
    plt.tight_layout()
    plt.savefig('multiple_regression_diagnostics.png')
    plt.close()
    
    # Statistical tests
    tests = {
        'normality': stats.normaltest(residuals),
        'homoscedasticity': stats.levene(residuals[fitted < np.median(fitted)],
                                       residuals[fitted >= np.median(fitted)])
    }
    
    return tests
```

## Model Interpretation

### 1. Coefficient Analysis
```python
def analyze_coefficients(model):
    """Analyze and interpret regression coefficients"""
    results = pd.DataFrame({
        'coefficient': model.params,
        'std_error': model.bse,
        't_value': model.tvalues,
        'p_value': model.pvalues,
        'conf_int_lower': model.conf_int()[0],
        'conf_int_upper': model.conf_int()[1]
    })
    
    # Add significance indicators
    results['significant'] = results['p_value'] < 0.05
    
    # Standardize coefficients
    X = model.model.exog
    y = model.model.endog
    standardized_model = sm.OLS(
        (y - y.mean()) / y.std(),
        (X - X.mean(0)) / X.std(0)
    ).fit()
    
    results['standardized_coef'] = standardized_model.params
    
    return results
```

### 2. Partial Regression Plots
```python
def plot_partial_regression(data, target, features):
    """Create partial regression plots"""
    fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 4))
    
    for i, feature in enumerate(features):
        # Residuals of y ~ other features
        other_features = [f for f in features if f != feature]
        y_model = sm.OLS(data[target], 
                        sm.add_constant(data[other_features])).fit()
        y_residuals = y_model.resid
        
        # Residuals of feature ~ other features
        x_model = sm.OLS(data[feature], 
                        sm.add_constant(data[other_features])).fit()
        x_residuals = x_model.resid
        
        # Plot
        if len(features) > 1:
            ax = axes[i]
        else:
            ax = axes
            
        ax.scatter(x_residuals, y_residuals, alpha=0.5)
        ax.set_xlabel(f'{feature} (residuals)')
        ax.set_ylabel(f'{target} (residuals)' if i == 0 else '')
        ax.set_title(f'Partial Regression Plot\n{feature}')
        
        # Add regression line
        z = np.polyfit(x_residuals, y_residuals, 1)
        p = np.poly1d(z)
        ax.plot(x_residuals, p(x_residuals), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('partial_regression_plots.png')
    plt.close()
```

## Model Validation

### 1. Cross-Validation
```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def cross_validate_multiple_regression(data, target, features, n_splits=5):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {
        'mse': [],
        'r2': []
    }
    
    for train_idx, test_idx in kf.split(data):
        # Split data
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Fit model
        model = sm.OLS(train_data[target], 
                      sm.add_constant(train_data[features])).fit()
        
        # Make predictions
        X_test = sm.add_constant(test_data[features])
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        cv_results['mse'].append(
            mean_squared_error(test_data[target], y_pred))
        cv_results['r2'].append(
            r2_score(test_data[target], y_pred))
    
    return {
        'mean_mse': np.mean(cv_results['mse']),
        'std_mse': np.std(cv_results['mse']),
        'mean_r2': np.mean(cv_results['r2']),
        'std_r2': np.std(cv_results['r2'])
    }
```

### 2. Prediction
```python
def make_predictions_with_intervals(model, new_data):
    """Make predictions with confidence and prediction intervals"""
    X_new = sm.add_constant(new_data)
    
    # Get predictions
    predictions = model.get_prediction(X_new)
    
    return pd.DataFrame({
        'prediction': predictions.predicted_mean,
        'conf_int_lower': predictions.conf_int()[:, 0],
        'conf_int_upper': predictions.conf_int()[:, 1],
        'pred_int_lower': predictions.pred_int()[:, 0],
        'pred_int_upper': predictions.pred_int()[:, 1]
    })
```

## Practice Questions
1. How do you handle correlated predictors in multiple regression?
2. What are the key assumptions of multiple regression?
3. How do you interpret standardized coefficients?
4. When should you use stepwise selection?
5. How do you handle categorical variables in multiple regression?

## Key Takeaways
1. Multiple regression extends simple regression to multiple predictors
2. Check for multicollinearity among predictors
3. Use partial regression plots for visualization
4. Consider both statistical and practical significance
5. Validate model assumptions and performance
