# Implementing Decision Trees with Scikit-learn 💻

Let's learn how to implement decision trees for both classification and regression tasks using scikit-learn.

## Basic Classification Example 🎯

### Simple Disease Diagnosis

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import matplotlib.pyplot as plt

# Create sample dataset
X = np.array([
    [101, 1, 1],  # [temperature, cough, fatigue]
    [99, 0, 0],
    [102, 1, 1],
    [98, 0, 1],
    [100, 1, 0]
])
y = ['sick', 'healthy', 'sick', 'healthy', 'healthy']

# Create and train model
def train_disease_classifier():
    """Train a simple disease classifier"""
    # Create model
    clf = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    # Train model
    clf.fit(X, y)
    
    # Visualize tree
    plt.figure(figsize=(15, 10))
    plot_tree(
        clf,
        feature_names=['temperature', 'cough', 'fatigue'],
        class_names=['healthy', 'sick'],
        filled=True,
        rounded=True
    )
    plt.title('Disease Diagnosis Decision Tree')
    plt.show()
    
    return clf

# Make predictions
model = train_disease_classifier()
new_patient = np.array([[100, 1, 1]])
prediction = model.predict(new_patient)
print(f"Diagnosis: {prediction[0]}")
```

## Regression Example 📈

### House Price Prediction

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

class HousePricePredictor:
    def __init__(self, max_depth=3):
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
    def train(self, X, y):
        """Train the price predictor"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training R² Score: {train_score:.3f}")
        print(f"Testing R² Score: {test_score:.3f}")
        
    def predict(self, features):
        """Predict house price"""
        return self.model.predict(features)

# Example usage
X = np.array([
    [1400, 3, 10],  # [size, bedrooms, age]
    [1600, 3, 8],
    [1700, 4, 15],
    [1875, 4, 5],
    [1100, 2, 20]
])
y = np.array([250000, 280000, 300000, 350000, 200000])

predictor = HousePricePredictor()
predictor.train(X, y)
```

## Tree Visualization 🎨

### Basic Tree Plot

```python
def visualize_tree(model, feature_names, class_names=None):
    """Create detailed tree visualization"""
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Decision Tree Visualization')
    plt.show()
```

### Feature Importance Plot

```python
def plot_feature_importance(model, feature_names):
    """Visualize feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), 
            importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45)
    plt.tight_layout()
    plt.show()
```

## Decision Boundary Visualization 🎯

```python
def plot_decision_boundary(model, X, y):
    """Plot decision boundary for 2D data"""
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    # Make predictions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Tree Decision Boundary')
    plt.show()
```

## Cross-Validation 📊

```python
from sklearn.model_selection import cross_val_score

def evaluate_with_cv(X, y, max_depths=[3, 5, 7, 10]):
    """Evaluate model with cross-validation"""
    results = []
    
    for depth in max_depths:
        model = DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(model, X, y, cv=5)
        
        results.append({
            'max_depth': depth,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        })
        
    # Plot results
    depths = [r['max_depth'] for r in results]
    means = [r['mean_score'] for r in results]
    stds = [r['std_score'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(depths, means, yerr=stds, marker='o')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Cross-validation Score')
    plt.title('Model Performance vs Tree Depth')
    plt.grid(True)
    plt.show()
    
    return results
```

## Complete Pipeline Example 🔄

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class DecisionTreePipeline:
    def __init__(self, max_depth=3):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        
    def train_and_evaluate(self, X, y):
        """Train and evaluate the pipeline"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(
            self.pipeline,
            X_test,
            y_test,
            cmap=plt.cm.Blues,
            normalize='true'
        )
        plt.title('Normalized Confusion Matrix')
        plt.show()
        
        return self.pipeline
```

## Best Practices 📚

### 1. Data Preprocessing

```python
def preprocess_data(X, categorical_features=[]):
    """Preprocess data for decision trees"""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), 
             categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor
```

### 2. Model Selection

```python
def select_best_model(X, y):
    """Select best tree configuration"""
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    tree = DecisionTreeClassifier()
    grid_search = GridSearchCV(
        tree, param_grid, cv=5,
        scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_
```

## Next Steps 📚

Now that you can implement decision trees:
1. Explore [advanced techniques](4-advanced.md)
2. Learn about [real-world applications](5-applications.md)
3. Practice with different datasets
4. Experiment with tree parameters

Remember:
- Start with simple trees
- Use cross-validation
- Visualize your trees
- Monitor for overfitting
