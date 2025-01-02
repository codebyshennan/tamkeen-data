# Advanced Decision Tree Techniques 🚀

Let's explore advanced concepts and optimizations that can improve your decision tree implementations.

## Tree Pruning Techniques 🌿

> **Pruning** is the process of reducing the size of a decision tree by removing sections that provide little predictive power.

### 1. Pre-pruning (Early Stopping)

```python
from sklearn.tree import DecisionTreeClassifier

class PrePrunedTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=3,              # Limit tree depth
            min_samples_split=20,     # Min samples to split
            min_samples_leaf=5,       # Min samples in leaf
            max_features='sqrt',      # Consider subset of features
            min_impurity_decrease=0.01  # Min improvement needed
        )
        
    def analyze_params(self, X, y):
        """Analyze impact of different parameters"""
        params = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [10, 20, 50],
            'min_samples_leaf': [5, 10, 20]
        }
        
        results = {}
        for param, values in params.items():
            scores = []
            for value in values:
                # Update parameter
                setattr(self.model, param, value)
                # Evaluate
                score = cross_val_score(
                    self.model, X, y, cv=5
                ).mean()
                scores.append(score)
            results[param] = scores
            
        self._plot_param_impact(results)
        
    def _plot_param_impact(self, results):
        """Plot parameter impact on performance"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (param, scores) in zip(axes, results.items()):
            ax.plot(scores, marker='o')
            ax.set_title(f'Impact of {param}')
            ax.grid(True)
        plt.tight_layout()
        plt.show()
```

### 2. Post-pruning (Cost-Complexity Pruning)

```python
def optimize_ccp_alpha(X, y):
    """Find optimal complexity parameter"""
    # Create tree
    tree = DecisionTreeClassifier()
    
    # Get path
    path = tree.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    
    # Train trees with different alphas
    trees = []
    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        tree.fit(X, y)
        trees.append(tree)
    
    # Plot number of nodes vs alpha
    node_counts = [tree.tree_.node_count for tree in trees]
    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas, node_counts, marker='o')
    plt.xlabel('ccp_alpha')
    plt.ylabel('Number of nodes')
    plt.title('Tree Size vs Alpha')
    plt.show()
    
    return trees, ccp_alphas
```

## Advanced Tree Growing 🌳

### 1. Custom Splitting Criteria

```python
from sklearn.tree._criterion import Criterion
import numpy as np

class CustomCriterion:
    """Example of custom splitting criterion"""
    def node_impurity(self, y):
        """Calculate node impurity"""
        # Example: Modified Gini
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 3)  # Cubic term
        
    def children_impurity(self, y_left, y_right):
        """Calculate children impurity"""
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        
        # Weighted impurity
        return (
            (n_left/n_total) * self.node_impurity(y_left) +
            (n_right/n_total) * self.node_impurity(y_right)
        )
```

### 2. Dynamic Feature Selection

```python
class DynamicFeatureSelector:
    def __init__(self, n_features=10):
        self.n_features = n_features
        
    def select_features(self, X, y, depth):
        """Select features based on tree depth"""
        if depth < 3:
            # Use all features at top levels
            return list(range(X.shape[1]))
        else:
            # Use feature importance for deeper levels
            importances = self._get_feature_importance(X, y)
            return np.argsort(importances)[-self.n_features:]
            
    def _get_feature_importance(self, X, y):
        """Calculate feature importance"""
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)
        return tree.feature_importances_
```

## Ensemble Preview 🎭

> **Ensemble Methods** combine multiple decision trees to create more robust models.

### 1. Random Forest Preview

```python
from sklearn.ensemble import RandomForestClassifier

def compare_with_forest(X, y):
    """Compare single tree vs random forest"""
    # Single tree
    tree = DecisionTreeClassifier(max_depth=3)
    tree_scores = cross_val_score(tree, X, y, cv=5)
    
    # Random forest
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=3
    )
    forest_scores = cross_val_score(forest, X, y, cv=5)
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.boxplot([tree_scores, forest_scores], 
                labels=['Single Tree', 'Random Forest'])
    plt.title('Performance Comparison')
    plt.ylabel('Accuracy')
    plt.show()
```

### 2. Gradient Boosting Preview

```python
from sklearn.ensemble import GradientBoostingClassifier

def preview_boosting(X, y):
    """Preview gradient boosting performance"""
    boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    
    # Train and track performance
    train_scores = []
    test_scores = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    
    for i in range(1, 101):
        boosting.n_estimators = i
        boosting.fit(X_train, y_train)
        train_scores.append(
            boosting.score(X_train, y_train)
        )
        test_scores.append(
            boosting.score(X_test, y_test)
        )
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='Train')
    plt.plot(test_scores, label='Test')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Gradient Boosting Learning Curves')
    plt.legend()
    plt.show()
```

## Advanced Visualization 📊

### 1. Interactive Tree Explorer

```python
def create_interactive_tree(model, feature_names):
    """Create interactive tree visualization"""
    from dtreeviz.trees import dtreeviz
    
    viz = dtreeviz(
        model,
        X_train,
        y_train,
        target_name='target',
        feature_names=feature_names,
        class_names=list(model.classes_)
    )
    
    return viz
```

### 2. Path Highlighter

```python
def highlight_decision_path(model, X, feature_names):
    """Highlight decision path for a sample"""
    # Get decision path
    path = model.decision_path(X)
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True
    )
    
    # Highlight path
    for node_id in path.indices:
        plt.plot(node_id, 'ro')
        
    plt.title('Decision Path Visualization')
    plt.show()
```

## Performance Optimization 🏃‍♂️

### 1. Memory-Efficient Implementation

```python
class MemoryEfficientTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.model = None
        
    def fit(self, X, y, batch_size=1000):
        """Train tree in batches"""
        models = []
        
        # Train on batches
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth
            )
            tree.fit(batch_X, batch_y)
            models.append(tree)
        
        # Combine predictions
        self.model = self._combine_trees(models)
        
    def _combine_trees(self, models):
        """Combine multiple trees into one"""
        # Implementation depends on specific needs
        return models[0]  # Simplified example
```

### 2. Parallel Processing

```python
from joblib import Parallel, delayed

def parallel_cross_validation(X, y, n_folds=5):
    """Perform cross-validation in parallel"""
    def evaluate_fold(train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    
    # Generate fold indices
    kf = KFold(n_splits=n_folds)
    
    # Parallel evaluation
    scores = Parallel(n_jobs=-1)(
        delayed(evaluate_fold)(train_idx, test_idx)
        for train_idx, test_idx in kf.split(X)
    )
    
    return np.mean(scores), np.std(scores)
```

## Next Steps 📚

After mastering these advanced techniques:
1. Learn about [real-world applications](5-applications.md)
2. Study ensemble methods in depth
3. Practice with complex datasets
4. Experiment with custom implementations

Remember:
- Balance complexity with interpretability
- Use pruning judiciously
- Consider computational costs
- Validate thoroughly
