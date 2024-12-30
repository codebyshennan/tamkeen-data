# Decision Trees 🌳

Decision trees are intuitive models that make predictions by recursively splitting data based on feature values. Think of playing "20 Questions" - each question narrows down the possibilities until you reach a final answer!

## Mathematical Foundation 📐

### Splitting Criteria

#### 1. Gini Impurity (Classification)
For a node with probability $p_i$ for class $i$:

$$\text{Gini} = 1 - \sum_{i=1}^c p_i^2$$

#### 2. Entropy (Classification)
Information gain based on entropy:

$$\text{Entropy} = -\sum_{i=1}^c p_i \log_2(p_i)$$
$$\text{Information Gain} = \text{Entropy}_{\text{parent}} - \sum_{j=1}^m \frac{N_j}{N} \text{Entropy}_{\text{child}_j}$$

#### 3. Mean Squared Error (Regression)
For a node with values $y_i$ and mean $\bar{y}$:

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \bar{y})^2$$

## Tree Construction Algorithm 🔨

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler

class DecisionTreeVisualizer:
    def __init__(self, max_depth=3):
        self.scaler = StandardScaler()
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        )
        
    def fit_and_visualize(self, X, y, feature_names=None):
        """Fit tree and create visualizations"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit tree
        self.tree.fit(X_scaled, y)
        
        # Plot tree structure
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.tree,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title('Decision Tree Structure')
        plt.show()
        
        # Plot decision boundary if 2D
        if X.shape[1] == 2:
            self._plot_decision_boundary(X_scaled, y)
            
        # Plot feature importance
        if feature_names is not None:
            self._plot_feature_importance(feature_names)
            
    def _plot_decision_boundary(self, X, y):
        """Plot decision boundary and data points"""
        # Create mesh grid
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )
        
        # Get predictions
        Z = self.tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title('Decision Boundaries')
        plt.show()
        
    def _plot_feature_importance(self, feature_names):
        """Visualize feature importance"""
        importances = self.tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)),
                importances[indices])
        plt.xticks(
            range(len(importances)),
            [feature_names[i] for i in indices],
            rotation=45
        )
        plt.tight_layout()
        plt.show()
```

## Real-World Applications 🌍

### 1. Credit Risk Assessment
```python
class CreditRiskClassifier:
    def __init__(self):
        self.tree = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced'
        )
        
    def preprocess_features(self, X):
        """Prepare features for credit risk assessment"""
        # Calculate financial ratios
        X['debt_to_income'] = X['total_debt'] / X['annual_income']
        X['payment_to_income'] = X['monthly_payment'] / (X['annual_income']/12)
        X['credit_utilization'] = X['current_balance'] / X['credit_limit']
        
        # Create risk indicators
        X['missed_payments_flag'] = (X['missed_payments'] > 0).astype(int)
        X['high_utilization'] = (X['credit_utilization'] > 0.7).astype(int)
        
        return X
        
    def fit(self, X, y):
        """Train the credit risk model"""
        X_processed = self.preprocess_features(X)
        self.tree.fit(X_processed, y)
        
    def predict_risk(self, X):
        """Predict credit risk with probabilities"""
        X_processed = self.preprocess_features(X)
        probabilities = self.tree.predict_proba(X_processed)
        return {
            'risk_score': probabilities[:, 1],
            'prediction': self.tree.predict(X_processed)
        }
```

### 2. Disease Diagnosis
```python
class MedicalDiagnosisTree:
    def __init__(self):
        self.tree = DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=30,
            min_samples_leaf=10
        )
        self.symptom_encoder = None
        
    def encode_symptoms(self, symptoms):
        """Convert symptom descriptions to binary features"""
        from sklearn.preprocessing import MultiLabelBinarizer
        if self.symptom_encoder is None:
            self.symptom_encoder = MultiLabelBinarizer()
            return self.symptom_encoder.fit_transform(symptoms)
        return self.symptom_encoder.transform(symptoms)
        
    def fit(self, symptoms, diagnoses):
        """Train the diagnosis model"""
        X = self.encode_symptoms(symptoms)
        self.tree.fit(X, diagnoses)
        
    def diagnose(self, symptoms):
        """Predict diagnosis and confidence"""
        X = self.encode_symptoms([symptoms])
        probabilities = self.tree.predict_proba(X)
        diagnosis = self.tree.predict(X)
        
        return {
            'diagnosis': diagnosis[0],
            'confidence': np.max(probabilities),
            'differential_diagnoses': [
                (self.tree.classes_[i], prob)
                for i, prob in enumerate(probabilities[0])
                if prob > 0.1
            ]
        }
```

## Optimization Techniques 🔧

### 1. Pre-pruning Parameters
```python
class TreeOptimizer:
    def __init__(self):
        self.best_params = None
        self.best_score = 0
        
    def optimize_parameters(self, X, y):
        """Find optimal tree parameters"""
        from sklearn.model_selection import GridSearchCV
        
        # Parameter grid
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            DecisionTreeClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        return grid_search.best_estimator_
```

### 2. Cost-Complexity Pruning
```python
def optimize_alpha(X, y):
    """Find optimal complexity parameter"""
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create tree
    tree = DecisionTreeClassifier()
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    
    # Test different alphas
    alphas = path.ccp_alphas
    scores = []
    
    for alpha in alphas:
        tree = DecisionTreeClassifier(ccp_alpha=alpha)
        tree.fit(X_train, y_train)
        scores.append(tree.score(X_val, y_val))
        
    # Plot alpha vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, scores, marker='o')
    plt.xlabel('Alpha (complexity parameter)')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy vs Alpha')
    plt.xscale('log')
    plt.grid(True)
    plt.show()
    
    # Return best alpha
    return alphas[np.argmax(scores)]
```

## Best Practices and Pitfalls 💡

### 1. Handling Missing Values
```python
def handle_missing_values(X):
    """Strategy for missing values in trees"""
    # For numerical features
    numerical_imputer = SimpleImputer(
        strategy='mean',
        add_indicator=True  # Creates binary indicator for missingness
    )
    
    # For categorical features
    categorical_imputer = SimpleImputer(
        strategy='most_frequent',
        add_indicator=True
    )
    
    return numerical_imputer, categorical_imputer
```

### 2. Handling Imbalanced Data
```python
def handle_imbalance(X, y):
    """Methods to handle class imbalance"""
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    
    # Combine over and under sampling
    steps = [
        ('over', SMOTE(sampling_strategy=0.8)),
        ('under', RandomUnderSampler(sampling_strategy=0.9))
    ]
    pipeline = Pipeline(steps=steps)
    
    return pipeline.fit_resample(X, y)
```

### 3. Feature Engineering
```python
def engineer_features(X):
    """Create interaction features"""
    from sklearn.preprocessing import PolynomialFeatures
    
    # Add interaction terms
    poly = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )
    
    return poly.fit_transform(X)
```

## Common Pitfalls and Solutions ⚠️

1. **Overfitting**
   - Use max_depth and min_samples constraints
   - Apply cost-complexity pruning
   - Cross-validate parameters

2. **Instability**
   - Use ensemble methods (Random Forests)
   - Increase min_samples_leaf
   - Average multiple trees

3. **Feature Scaling**
   - Trees don't require scaling
   - But scaling helps visualization
   - Keep original values for interpretability

## Next Steps 📚

Now that you understand Decision Trees, explore ensemble methods in [Supervised Learning Part 2](../5.3-supervised-learning-2/README.md) to learn about Random Forests and Gradient Boosting!
