# Model Evaluation Metrics

## What are Evaluation Metrics? 🤔

Think of evaluation metrics as the "scorecard" or "report card" for your machine learning model. Just like how a teacher uses different tests and assignments to evaluate a student's performance, we use different metrics to evaluate how well our model is performing.

### Why Metrics Matter 🌟

Imagine you're a doctor diagnosing patients. You wouldn't just look at one symptom - you'd consider multiple factors like temperature, blood pressure, and lab results. Similarly, in machine learning, we need multiple metrics to get a complete picture of our model's performance.

## Real-World Analogies 📚

### The Sports Analogy

Think of model evaluation like evaluating a sports team:

- Accuracy is like the win-loss record
- Precision is like the percentage of shots that hit the target
- Recall is like the percentage of opportunities that were converted
- F1-score is like the overall team performance rating

### The Weather Forecast Analogy

Model evaluation is like weather forecasting:

- Accuracy is like how often the forecast is correct
- Precision is like how specific the forecast is
- Recall is like how well we catch all the important weather events
- ROC curve is like the trade-off between false alarms and missed events

## Classification Metrics 🎯

### 1. Accuracy

This is like the percentage of correct answers on a test.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.savefig('assets/confusion_matrix.png')
    plt.show()

plot_confusion_matrix(y_test, y_pred)
```

### 2. Precision and Recall

These are like the balance between being thorough and being accurate.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Visualize trade-off
def plot_precision_recall_tradeoff(y_true, y_pred_proba):
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('assets/precision_recall_curve.png')
    plt.show()

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
plot_precision_recall_tradeoff(y_test, y_pred_proba)
```

### 3. ROC Curve and AUC

This is like the trade-off between sensitivity and specificity.

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
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

plot_roc_curve(y_test, y_pred_proba)
```

## Regression Metrics 📈

### 1. Mean Squared Error (MSE)

This is like the average squared difference between predictions and actual values.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")

# Visualize predictions
def plot_regression_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Regression Predictions vs True Values')
    plt.grid(True)
    plt.savefig('assets/regression_predictions.png')
    plt.show()

plot_regression_predictions(y_test, y_pred)
```

### 2. R-squared Score

This is like the percentage of variance explained by the model.

```python
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared Score: {r2:.3f}")

# Visualize residuals
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.savefig('assets/residual_plot.png')
    plt.show()

plot_residuals(y_test, y_pred)
```

## Common Mistakes to Avoid ⚠️

1. **Using Wrong Metrics**
   - Using accuracy for imbalanced data
   - Using MSE for classification
   - Not considering business context

2. **Ignoring Data Distribution**
   - Not checking class imbalance
   - Not considering outliers
   - Not validating assumptions

3. **Overlooking Model Limitations**
   - Not considering model bias
   - Not checking for overfitting
   - Not validating on new data

## Practical Example: Credit Risk Prediction 💳

Let's see how different metrics help evaluate a credit risk model:

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

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Plot ROC curve
plot_roc_curve(y_test, y_pred_proba)
```

## Best Practices 🌟

### 1. Choosing the Right Metrics

```python
def evaluate_model(y_true, y_pred, y_pred_proba):
    # Classification metrics
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.3f}")
    
    # Plot ROC curve
    plot_roc_curve(y_true, y_pred_proba)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

evaluate_model(y_test, y_pred, y_pred_proba)
```

## Additional Resources 📚

1. **Online Courses**
   - Coursera: Machine Learning by Andrew Ng
   - edX: Introduction to Machine Learning

2. **Books**
   - "Introduction to Machine Learning with Python" by Andreas Müller
   - "Hands-On Machine Learning with Scikit-Learn" by Aurélien Géron

3. **Documentation**
   - [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
   - [Precision-Recall Trade-off](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

## Next Steps 🚀

Ready to learn more? Check out:

1. [Model Selection](./model-selection.md) to choose the best model for your problem
2. [Cross Validation](./cross-validation.md) to properly evaluate your model
3. [Hyperparameter Tuning](./hyperparameter-tuning.md) to optimize your model's performance
