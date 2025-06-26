#!/usr/bin/env python3
"""
Comprehensive script to generate all visualizations for ROC-AUC and Metrics documentation.
This script creates all the charts, graphs, and code outputs referenced in the markdown files.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from itertools import cycle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def ensure_assets_dir():
    """Ensure assets directory exists"""
    import os
    if not os.path.exists('assets'):
        os.makedirs('assets')

def generate_basic_roc_curve():
    """Generate basic ROC curve for binary classification"""
    print("Generating basic ROC curve...")
    
    # Create sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, 
                             n_informative=15, n_redundant=5,
                             random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Alternative: Direct AUC calculation
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc_score:.3f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Basic Binary Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/roc_curve_basic_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_train, X_test, y_train, y_test

def generate_multiple_models_roc():
    """Generate ROC curves comparing multiple models"""
    print("Generating multiple models ROC comparison...")
    
    # Create dataset
    X, y = make_classification(n_samples=1000, n_features=20, 
                             n_informative=15, n_redundant=5,
                             random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # Add random classifier line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multiple Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/roc_multiple_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_multiclass_roc():
    """Generate multi-class ROC curves"""
    print("Generating multi-class ROC curves...")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Binarize the output for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot multi-class ROC curves
    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'red', 'green'])
    class_names = iris.target_names
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves (Iris Dataset)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/roc_multiclass_iris.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_thresholds(y_true, y_pred_proba, thresholds=None):
    """Analyze model performance across different thresholds."""
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    metrics = {
        'threshold': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'fpr': [],
        'tpr': []
    }
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        if len(np.unique(y_pred)) > 1:  # Avoid division by zero
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            precision = recall = f1 = 0
        
        # Calculate TPR and FPR
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics['threshold'].append(threshold)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['fpr'].append(fpr)
        metrics['tpr'].append(tpr)
    
    return metrics

def generate_threshold_analysis():
    """Generate threshold analysis plots"""
    print("Generating threshold analysis...")
    
    # Create dataset and train model
    X, y = make_classification(n_samples=1000, n_features=20, 
                             n_informative=15, n_redundant=5,
                             random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Analyze thresholds
    threshold_metrics = analyze_thresholds(y_test, y_pred_proba)
    
    # Plot threshold analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Precision, Recall, F1 vs Threshold
    ax1.plot(threshold_metrics['threshold'], threshold_metrics['precision'], 
             label='Precision', linewidth=2)
    ax1.plot(threshold_metrics['threshold'], threshold_metrics['recall'], 
             label='Recall', linewidth=2)
    ax1.plot(threshold_metrics['threshold'], threshold_metrics['f1'], 
             label='F1-Score', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, and F1-Score vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: TPR and FPR vs Threshold
    ax2.plot(threshold_metrics['threshold'], threshold_metrics['tpr'], 
             label='True Positive Rate', linewidth=2)
    ax2.plot(threshold_metrics['threshold'], threshold_metrics['fpr'], 
             label='False Positive Rate', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Rate')
    ax2.set_title('TPR and FPR vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/threshold_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_credit_risk_analysis():
    """Generate comprehensive credit risk analysis"""
    print("Generating credit risk analysis...")
    
    # Create realistic credit risk dataset
    np.random.seed(42)
    n_samples = 2000
    
    # Generate correlated features
    data = {
        'age': np.random.normal(35, 12, n_samples),
        'income': np.random.lognormal(10.5, 0.8, n_samples),  # Log-normal for income
        'credit_score': np.random.normal(650, 120, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'employment_years': np.random.exponential(5, n_samples),
        'num_credit_accounts': np.random.poisson(3, n_samples),
        'credit_utilization': np.random.beta(2, 3, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Clip values to realistic ranges
    df['age'] = np.clip(df['age'], 18, 80)
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    df['employment_years'] = np.clip(df['employment_years'], 0, 40)
    df['credit_utilization'] = np.clip(df['credit_utilization'], 0, 1)
    
    # Create target variable (loan default) with realistic relationships
    default_probability = (
        -0.01 * df['credit_score'] +
        -0.00001 * df['income'] +
        0.5 * df['debt_to_income'] +
        0.8 * df['credit_utilization'] +
        -0.02 * df['employment_years'] +
        0.05 * df['num_credit_accounts'] +
        5  # Base probability
    )
    
    # Convert to probability and create binary target
    default_prob = 1 / (1 + np.exp(-default_probability))
    y = np.random.binomial(1, default_prob, n_samples)
    
    print(f"Default rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot comprehensive analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve - Credit Risk Model')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Threshold analysis
    threshold_metrics = analyze_thresholds(y_test, y_pred_proba, np.linspace(0, 1, 101))
    
    ax2.plot(threshold_metrics['threshold'], threshold_metrics['precision'], 
             label='Precision', linewidth=2)
    ax2.plot(threshold_metrics['threshold'], threshold_metrics['recall'], 
             label='Recall (TPR)', linewidth=2)
    ax2.plot(threshold_metrics['threshold'], threshold_metrics['f1'], 
             label='F1-Score', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Performance Metrics vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Distribution of predicted probabilities
    ax3.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Default', density=True)
    ax3.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Default', density=True)
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Predicted Probabilities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Feature importance
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    feature_names = df.columns
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    ax4.bar(range(len(feature_importance)), feature_importance[sorted_idx])
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Importance')
    ax4.set_title('Feature Importance')
    ax4.set_xticks(range(len(feature_importance)))
    ax4.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/credit_risk_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print model performance summary
    print(f"\nCredit Risk Model Performance:")
    print(f"AUC Score: {roc_auc:.3f}")
    print(f"Number of test samples: {len(y_test)}")
    print(f"Actual default rate: {y_test.mean():.2%}")
    print(f"Predicted default rate (threshold=0.5): {(y_pred_proba >= 0.5).mean():.2%}")

def bootstrap_auc(y_true, y_pred_proba, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval for AUC."""
    n_samples = len(y_true)
    bootstrap_aucs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]
        
        # Calculate AUC for bootstrap sample
        try:
            auc_boot = roc_auc_score(y_boot, y_pred_boot)
            bootstrap_aucs.append(auc_boot)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))
    
    return np.mean(bootstrap_aucs), lower, upper, bootstrap_aucs

def generate_bootstrap_confidence():
    """Generate bootstrap confidence interval visualization"""
    print("Generating bootstrap confidence interval analysis...")
    
    # Create dataset and train model
    X, y = make_classification(n_samples=1000, n_features=20, 
                             n_informative=15, n_redundant=5,
                             random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate confidence interval
    auc_mean, auc_lower, auc_upper, bootstrap_aucs = bootstrap_auc(y_test, y_pred_proba)
    
    # Plot bootstrap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_aucs, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    plt.axvline(auc_mean, color='red', linestyle='--', linewidth=2, label=f'Mean AUC: {auc_mean:.3f}')
    plt.axvline(auc_lower, color='orange', linestyle='--', linewidth=2, label=f'95% CI Lower: {auc_lower:.3f}')
    plt.axvline(auc_upper, color='orange', linestyle='--', linewidth=2, label=f'95% CI Upper: {auc_upper:.3f}')
    plt.xlabel('AUC Score')
    plt.ylabel('Density')
    plt.title('Bootstrap Distribution of AUC Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/bootstrap_auc_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"AUC: {auc_mean:.3f} (95% CI: {auc_lower:.3f} - {auc_upper:.3f})")

def demonstrate_precision_recall_tradeoff():
    """Demonstrate precision-recall trade-off"""
    print("Generating precision-recall trade-off demonstration...")
    
    # Simulate different threshold scenarios
    thresholds = np.linspace(0.1, 0.9, 9)
    scenarios = {
        'Conservative Model': {'precision': [0.95, 0.92, 0.88, 0.82, 0.75, 0.68, 0.60, 0.52, 0.45],
                              'recall': [0.20, 0.35, 0.48, 0.62, 0.73, 0.81, 0.87, 0.91, 0.94]},
        'Aggressive Model': {'precision': [0.60, 0.58, 0.55, 0.52, 0.48, 0.44, 0.40, 0.36, 0.32],
                            'recall': [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]},
        'Balanced Model': {'precision': [0.80, 0.78, 0.75, 0.72, 0.68, 0.64, 0.60, 0.55, 0.50],
                          'recall': [0.50, 0.58, 0.65, 0.71, 0.76, 0.80, 0.84, 0.87, 0.90]}
    }
    
    plt.figure(figsize=(12, 8))
    
    for model_name, metrics in scenarios.items():
        plt.plot(metrics['recall'], metrics['precision'], 'o-', 
                label=f'{model_name}', linewidth=2, markersize=6)
    
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision vs Recall Trade-off for Different Model Types', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add annotations
    plt.annotate('High Precision\nLow Recall\n(Conservative)', 
                xy=(0.3, 0.9), fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.annotate('Low Precision\nHigh Recall\n(Aggressive)', 
                xy=(0.9, 0.4), fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    plt.annotate('Balanced\nPrecision & Recall', 
                xy=(0.7, 0.7), fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.savefig('assets/precision_recall_tradeoff_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_confusion_matrix_example():
    """Generate confusion matrix example"""
    print("Generating confusion matrix example...")
    
    # Create sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, 
                             n_informative=15, n_redundant=5,
                             random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix Example')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix_example.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_regression_examples():
    """Generate regression examples"""
    print("Generating regression examples...")
    
    # Create regression dataset
    X, y = make_regression(n_samples=1000, n_features=20, 
                          n_informative=15, noise=0.1,
                          random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"R-squared Score: {r2:.3f}")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Predictions vs True Values
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title('Regression Predictions vs True Values')
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predictions')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/regression_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_cross_validation_example():
    """Generate cross-validation example"""
    print("Generating cross-validation example...")
    
    # Create dataset
    X, y = make_classification(n_samples=1000, n_features=20, 
                             n_informative=15, n_redundant=5,
                             random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"Cross-validation AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Plot CV scores
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(cv_scores) + 1), cv_scores, alpha=0.7)
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {cv_scores.mean():.3f}')
    plt.xlabel('Fold')
    plt.ylabel('AUC Score')
    plt.title('Cross-Validation AUC Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/cross_validation_example.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_metrics_comparison_table():
    """Generate metrics comparison visualization"""
    print("Generating metrics comparison table...")
    
    # Create sample data for different scenarios
    scenarios = {
        'Balanced Dataset': {
            'Accuracy': 0.85,
            'Precision': 0.83,
            'Recall': 0.87,
            'F1-Score': 0.85,
            'ROC-AUC': 0.92
        },
        'Imbalanced Dataset (10:1)': {
            'Accuracy': 0.91,
            'Precision': 0.45,
            'Recall': 0.78,
            'F1-Score': 0.57,
            'ROC-AUC': 0.84
        },
        'Conservative Model': {
            'Accuracy': 0.88,
            'Precision': 0.95,
            'Recall': 0.65,
            'F1-Score': 0.77,
            'ROC-AUC': 0.89
        },
        'Aggressive Model': {
            'Accuracy': 0.82,
            'Precision': 0.68,
            'Recall': 0.95,
            'F1-Score': 0.79,
            'ROC-AUC': 0.88
        }
    }
    
    # Create DataFrame
    df = pd.DataFrame(scenarios).T
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='RdYlGn', center=0.5, 
                fmt='.2f', cbar_kws={'label': 'Score'})
    plt.title('Model Performance Metrics Comparison', fontsize=16)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Scenarios', fontsize=12)
    plt.tight_layout()
    plt.savefig('assets/metrics_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all visualizations"""
    print("Starting visualization generation...")
    print("=" * 50)
    
    # Ensure assets directory exists
    ensure_assets_dir()
    
    # Generate all visualizations
    try:
        generate_basic_roc_curve()
        print("✓ Basic ROC curve generated")
        
        generate_multiple_models_roc()
        print("✓ Multiple models ROC comparison generated")
        
        generate_multiclass_roc()
        print("✓ Multi-class ROC curves generated")
        
        generate_threshold_analysis()
        print("✓ Threshold analysis generated")
        
        generate_credit_risk_analysis()
        print("✓ Credit risk analysis generated")
        
        generate_bootstrap_confidence()
        print("✓ Bootstrap confidence interval generated")
        
        demonstrate_precision_recall_tradeoff()
        print("✓ Precision-recall trade-off demonstration generated")
        
        generate_confusion_matrix_example()
        print("✓ Confusion matrix example generated")
        
        generate_regression_examples()
        print("✓ Regression examples generated")
        
        generate_cross_validation_example()
        print("✓ Cross-validation example generated")
        
        generate_metrics_comparison_table()
        print("✓ Metrics comparison table generated")
        
        print("=" * 50)
        print("All visualizations generated successfully!")
        print("Check the 'assets/' directory for the generated images.")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
