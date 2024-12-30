# Logistic Regression Fundamentals

## Introduction
Logistic regression is a statistical model used for binary classification problems, extending linear regression to scenarios where the dependent variable is categorical. Unlike linear regression, which predicts continuous values, logistic regression models the probability of an observation belonging to a particular class.

### Mathematical Foundation
The logistic regression model transforms a linear combination of features into a probability using the logistic (sigmoid) function:

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_pX_p)}} = \frac{1}{1 + e^{-X^T\beta}}$$

where:
- P(Y=1|X) is the probability of the positive class given features X
- β₀ is the intercept term
- β₁, ..., βₚ are the coefficients for features X₁, ..., Xₚ
- X^Tβ represents the linear combination of features and coefficients

The logit transformation converts probabilities to log-odds:

$$logit(p) = ln(\frac{p}{1-p}) = \beta_0 + \beta_1X_1 + ... + \beta_pX_p$$

This transformation is crucial because it:
1. Maps probabilities (0 to 1) to the entire real line (-∞ to +∞)
2. Creates a linear relationship with the predictors
3. Makes the model parameters interpretable as log-odds ratios

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def generate_sample_data(n=1000, seed=42):
    """Generate sample data for logistic regression"""
    np.random.seed(seed)
    
    # Generate features
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    
    # Generate probabilities using logistic function
    z = 1.5 * X1 + 2 * X2
    prob = 1 / (1 + np.exp(-z))
    
    # Generate binary outcomes
    y = (prob > 0.5).astype(int)
    
    return pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'y': y
    })
```

## The Logistic Model

### 1. Understanding the Logit Function
```python
def plot_logistic_curve():
    """Visualize the logistic function"""
    x = np.linspace(-6, 6, 100)
    y = 1 / (1 + np.exp(-x))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('Logistic Function')
    plt.xlabel('z = β₀ + β₁X₁ + β₂X₂ + ...')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    plt.savefig('logistic_curve.png')
    plt.close()
```

### 2. Fitting the Model
```python
def fit_logistic_model(X, y):
    """Fit logistic regression model using statsmodels"""
    # Add constant
    X = sm.add_constant(X)
    
    # Fit model
    model = sm.Logit(y, X).fit()
    
    return {
        'model': model,
        'summary': model.summary(),
        'params': model.params,
        'pvalues': model.pvalues,
        'conf_int': model.conf_int()
    }
```

## Model Interpretation

### 1. Odds Ratios
```python
def interpret_odds_ratios(model):
    """Calculate and interpret odds ratios"""
    odds_ratios = np.exp(model.params)
    conf_int = np.exp(model.conf_int())
    
    interpretation = pd.DataFrame({
        'Odds_Ratio': odds_ratios,
        'CI_Lower': conf_int[0],
        'CI_Upper': conf_int[1]
    })
    
    # Add percent change interpretation
    interpretation['Percent_Change'] = (odds_ratios - 1) * 100
    
    return interpretation
```

### 2. Predicted Probabilities
```python
def calculate_predicted_probs(model, X):
    """Calculate predicted probabilities"""
    X_with_const = sm.add_constant(X)
    predicted_probs = model.predict(X_with_const)
    
    return predicted_probs
```

## Model Evaluation

### 1. Classification Metrics
```python
def evaluate_classification(y_true, y_pred_prob, threshold=0.5):
    """Calculate classification metrics"""
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(y_true, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'roc_auc': roc_auc
    }
```

### 2. Model Diagnostics
```python
def logistic_model_diagnostics(model, X, y):
    """Perform diagnostics for logistic regression"""
    # Predicted probabilities
    y_pred_prob = model.predict(sm.add_constant(X))
    
    # Pearson residuals
    pearson_resid = (y - y_pred_prob) / np.sqrt(y_pred_prob * (1 - y_pred_prob))
    
    # Deviance residuals
    deviance_resid = np.sqrt(2 * (y * np.log((y + 1e-10) / y_pred_prob) + 
                                 (1-y) * np.log((1-y + 1e-10) / (1-y_pred_prob))))
    deviance_resid *= np.where(y > y_pred_prob, 1, -1)
    
    plt.figure(figsize=(12, 4))
    
    # Residual plot
    plt.subplot(131)
    plt.scatter(y_pred_prob, pearson_resid, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Pearson Residuals vs Fitted')
    plt.xlabel('Fitted Probabilities')
    plt.ylabel('Pearson Residuals')
    
    # Influence plot
    plt.subplot(132)
    influence = model.get_influence()
    plt.scatter(range(len(y)), influence.hat_matrix_diag, alpha=0.5)
    plt.title('Leverage Values')
    plt.xlabel('Observation')
    plt.ylabel('Leverage')
    
    # QQ plot of deviance residuals
    plt.subplot(133)
    stats.probplot(deviance_resid, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Deviance Residuals')
    
    plt.tight_layout()
    plt.savefig('logistic_diagnostics.png')
    plt.close()
    
    return {
        'pearson_residuals': pearson_resid,
        'deviance_residuals': deviance_resid,
        'leverage': influence.hat_matrix_diag
    }
```

## Model Selection and Validation

### 1. Feature Selection
```python
def stepwise_logistic_selection(X, y, threshold=0.05):
    """Perform stepwise feature selection"""
    included = []
    while True:
        changed = False
        
        # Forward step
        excluded = list(set(X.columns) - set(included))
        
        if len(excluded) > 0:
            pvalues = pd.Series(index=excluded)
            for feature in excluded:
                features = included + [feature]
                X_subset = sm.add_constant(X[features])
                model = sm.Logit(y, X_subset).fit(disp=0)
                pvalues[feature] = model.pvalues[feature]
            
            best_pvalue = pvalues.min()
            if best_pvalue < threshold:
                best_feature = pvalues.idxmin()
                included.append(best_feature)
                changed = True
        
        if not changed:
            break
    
    return included
```

### 2. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

def cross_validate_logistic(X, y, cv=5):
    """Perform cross-validation"""
    model = LogisticRegression()
    
    # Calculate different metrics
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    auc_roc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    
    return {
        'accuracy': accuracy.mean(),
        'auc_roc': auc_roc.mean(),
        'precision': precision.mean(),
        'recall': recall.mean()
    }
```

## Practice Questions
1. When should you use logistic regression?
2. How do you interpret odds ratios?
3. What metrics are important for model evaluation?
4. How do you handle class imbalance?
5. What are the assumptions of logistic regression?

## Key Takeaways
1. Logistic regression predicts probabilities
2. Interpret results using odds ratios
3. Evaluate using multiple metrics
4. Consider class imbalance
5. Validate assumptions and performance
