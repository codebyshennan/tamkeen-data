# Cross Validation

Imagine testing a recipe by having different people cook it - you'd get a better idea of how well it works in general. Cross validation does the same thing for machine learning models! Let's learn how to properly evaluate model performance. 🧪

## Understanding Cross Validation 🎯

Cross validation helps us:
1. Estimate model performance
2. Detect overfitting
3. Compare models fairly
4. Validate model stability

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Perform cross-validation
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## K-Fold Cross Validation 📊

```python
from sklearn.model_selection import KFold
import seaborn as sns

def visualize_kfold(X, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    plt.figure(figsize=(15, n_splits*2))
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        plt.subplot(n_splits, 1, i+1)
        plt.scatter(range(len(X)), [i]*len(X), c='lightgray', alpha=0.5)
        plt.scatter(train_idx, [i]*len(train_idx), c='blue', label='Train')
        plt.scatter(val_idx, [i]*len(val_idx), c='red', label='Validation')
        plt.title(f'Fold {i+1}')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualize K-Fold splits
visualize_kfold(X)
```

## Stratified K-Fold 🎯

Maintains class distribution in each fold:

```python
from sklearn.model_selection import StratifiedKFold

def compare_fold_distributions(X, y, n_splits=5):
    # Regular K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot regular K-Fold distributions
    for i, (_, val_idx) in enumerate(kf.split(X)):
        ax1.bar(i, y[val_idx].mean(), label=f'Fold {i+1}')
    ax1.axhline(y.mean(), color='r', linestyle='--', label='Overall mean')
    ax1.set_title('Regular K-Fold Class Distribution')
    ax1.legend()
    
    # Plot Stratified K-Fold distributions
    for i, (_, val_idx) in enumerate(skf.split(X, y)):
        ax2.bar(i, y[val_idx].mean(), label=f'Fold {i+1}')
    ax2.axhline(y.mean(), color='r', linestyle='--', label='Overall mean')
    ax2.set_title('Stratified K-Fold Class Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

compare_fold_distributions(X, y)
```

## Time Series Cross Validation 📈

For temporal data:

```python
from sklearn.model_selection import TimeSeriesSplit

def visualize_timeseries_cv(X, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    plt.figure(figsize=(15, n_splits*2))
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        plt.subplot(n_splits, 1, i+1)
        plt.scatter(range(len(X)), [i]*len(X), c='lightgray', alpha=0.5)
        plt.scatter(train_idx, [i]*len(train_idx), c='blue', label='Train')
        plt.scatter(val_idx, [i]*len(val_idx), c='red', label='Validation')
        plt.title(f'Time Series Split {i+1}')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualize Time Series splits
visualize_timeseries_cv(X)
```

## Leave-One-Out Cross Validation 🎲

For small datasets:

```python
from sklearn.model_selection import LeaveOneOut

def demonstrate_loo_cv(X, y, n_samples=10):
    # Use subset for demonstration
    X_subset = X[:n_samples]
    y_subset = y[:n_samples]
    
    loo = LeaveOneOut()
    scores = []
    
    for train_idx, val_idx in loo.split(X_subset):
        X_train, X_val = X_subset[train_idx], X_subset[val_idx]
        y_train, y_val = y_subset[train_idx], y_subset[val_idx]
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
    
    plt.figure(figsize=(10, 5))
    plt.plot(scores, 'bo-')
    plt.title('Leave-One-Out CV Scores')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

demonstrate_loo_cv(X, y)
```

## Group Cross Validation 👥

For grouped data:

```python
from sklearn.model_selection import GroupKFold

def demonstrate_group_cv():
    # Create sample grouped data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 1, 1, 2, 2])
    groups = np.array([0, 0, 1, 1, 2, 2])  # Two samples per group
    
    group_kfold = GroupKFold(n_splits=3)
    
    plt.figure(figsize=(15, 6))
    for i, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
        plt.subplot(3, 1, i+1)
        plt.scatter(range(len(X)), [i]*len(X), c='lightgray', alpha=0.5)
        plt.scatter(train_idx, [i]*len(train_idx), c='blue', label='Train')
        plt.scatter(val_idx, [i]*len(val_idx), c='red', label='Validation')
        plt.title(f'Group {i+1}')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

demonstrate_group_cv()
```

## Real-World Example: Credit Risk 💳

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

# Perform stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_val, y_val)
    scores.append(score)
    print(f"Fold {fold+1}: {score:.3f}")

print(f"\nMean CV score: {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")
```

## Best Practices 🌟

### 1. Choosing Number of Folds
```python
def choose_optimal_k(X, y, k_range=range(2, 11)):
    scores = []
    stds = []
    
    for k in k_range:
        cv_scores = cross_val_score(
            LogisticRegression(),
            X, y,
            cv=k
        )
        scores.append(cv_scores.mean())
        stds.append(cv_scores.std())
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(k_range, scores, yerr=stds, fmt='o-')
    plt.xlabel('Number of Folds')
    plt.ylabel('Cross-validation Score')
    plt.title('Impact of K on Cross-validation')
    plt.grid(True)
    plt.show()
```

### 2. Handling Imbalanced Data
```python
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

def cv_with_sampling(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Apply SMOTE only to training data
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train, y_train
        )
        
        model = LogisticRegression()
        model.fit(X_train_resampled, y_train_resampled)
        scores.append(model.score(X_val, y_val))
    
    return scores
```

### 3. Cross-validation with Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

def cv_with_pipeline(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(k=10)),
        ('classifier', LogisticRegression())
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=5)
    return scores
```

## Common Pitfalls and Solutions 🚧

1. **Data Leakage**
   - Keep preprocessing inside CV
   - Use pipelines
   - Validate feature selection

2. **Temporal Dependencies**
   - Use time series CV
   - Maintain order
   - Consider lag features

3. **Group Dependencies**
   - Use group K-fold
   - Maintain group integrity
   - Document relationships

## Next Steps

Now that you understand cross-validation, let's explore [Hyperparameter Tuning](./hyperparameter-tuning.md) to optimize your models!
