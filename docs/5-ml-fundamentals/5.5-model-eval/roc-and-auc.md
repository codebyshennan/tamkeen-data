# ROC and AUC

## What are ROC and AUC? 🤔

Think of ROC (Receiver Operating Characteristic) and AUC (Area Under the Curve) as a way to measure how good your model is at distinguishing between different classes. It's like having a radar system that needs to distinguish between friendly and enemy aircraft - you want to minimize false alarms while catching all real threats.

### Why ROC and AUC Matter 🌟

Imagine you're a doctor diagnosing a disease. You want to:

- Correctly identify all patients with the disease (high sensitivity)
- Avoid false alarms for healthy patients (high specificity)
- Find the right balance between these two goals

ROC and AUC help us find this balance and measure how well our model performs across different thresholds.

## Real-World Analogies 📚

### The Airport Security Analogy

Think of ROC and AUC like airport security:

- True Positives: Correctly identifying dangerous items
- False Positives: Flagging safe items as dangerous
- True Negatives: Correctly identifying safe items
- False Negatives: Missing dangerous items

The ROC curve shows how the security system performs at different sensitivity levels.

### The Weather Forecast Analogy

ROC and AUC are like weather forecasting:

- True Positives: Correctly predicting rain when it rains
- False Positives: Predicting rain when it's sunny
- True Negatives: Correctly predicting sunshine
- False Negatives: Missing rain predictions

The AUC tells us how good our weather forecaster is overall.

## Understanding ROC and AUC 🎯

### 1. ROC Curve

This shows the trade-off between sensitivity and specificity.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

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

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('assets/roc_curve.png')
    plt.show()

plot_roc_curve(fpr, tpr, roc_auc)
```

### 2. AUC Score

This measures the overall ability of the model to distinguish between classes.

```python
from sklearn.metrics import roc_auc_score

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.3f}")

# Visualize AUC
def plot_auc_interpretation():
    plt.figure(figsize=(10, 8))
    
    # Plot perfect classifier
    plt.plot([0, 0, 1], [0, 1, 1], 'g--', label='Perfect Classifier (AUC = 1.0)')
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier (AUC = 0.5)')
    
    # Plot our model
    plt.plot(fpr, tpr, 'b-', label=f'Our Model (AUC = {auc_score:.2f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Interpretation')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('assets/auc_interpretation.png')
    plt.show()

plot_auc_interpretation()
```

### 3. Threshold Selection

This helps us choose the best operating point for our model.

```python
def plot_threshold_analysis(y_true, y_pred_proba):
    # Calculate metrics for different thresholds
    thresholds = np.linspace(0, 1, 100)
    fprs = []
    tprs = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fprs.append(fpr)
        tprs.append(tpr)
    
    # Plot threshold analysis
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, fprs, label='False Positive Rate')
    plt.plot(thresholds, tprs, label='True Positive Rate')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('Threshold Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/threshold_analysis.png')
    plt.show()

plot_threshold_analysis(y_test, y_pred_proba)
```

## Common Mistakes to Avoid ⚠️

1. **Threshold Selection**
   - Using default threshold (0.5)
   - Not considering business costs
   - Not validating on new data

2. **AUC Interpretation**
   - Assuming high AUC means good model
   - Not considering class imbalance
   - Not looking at ROC curve shape

3. **Model Comparison**
   - Comparing AUC without context
   - Not considering computational cost
   - Not considering interpretability

## Practical Example: Credit Risk Prediction 💳

Let's see how ROC and AUC help in a credit risk prediction task:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create credit risk dataset
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.normal(35, 10, n_samples)
income = np.random.exponential(50000, n_samples)
credit_score = np.random.normal(700, 100, n_samples)

X = np.column_stack([age, income, credit_score])
y = (credit_score + income/1000 + age > 800).astype(int)  # Binary target

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Get probability predictions
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate and plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, roc_auc)

# Analyze thresholds
plot_threshold_analysis(y_test, y_pred_proba)
```

## Best Practices 🌟

### 1. ROC and AUC Analysis

```python
def analyze_roc_auc(y_true, y_pred_proba):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc)
    
    # Analyze thresholds
    plot_threshold_analysis(y_true, y_pred_proba)
    
    # Print AUC score
    print(f"AUC Score: {roc_auc:.3f}")
    
    return roc_auc

analyze_roc_auc(y_test, y_pred_proba)
```

## Additional Resources 📚

1. **Online Courses**
   - Coursera: Machine Learning by Andrew Ng
   - edX: Introduction to Machine Learning

2. **Books**
   - "Introduction to Machine Learning with Python" by Andreas Müller
   - "Hands-On Machine Learning with Scikit-Learn" by Aurélien Géron

3. **Documentation**
   - [Scikit-learn ROC Curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
   - [AUC Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

## Next Steps 🚀

Ready to learn more? Check out:

1. [Model Metrics](./metrics.md) to understand other evaluation metrics
2. [Cross Validation](./cross-validation.md) to properly evaluate your model
3. [Model Selection](./model-selection.md) to choose the best model for your problem
