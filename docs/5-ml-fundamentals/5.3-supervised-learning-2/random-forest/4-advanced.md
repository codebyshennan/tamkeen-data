# Advanced Random Forest Techniques

Let's explore advanced concepts and techniques to take your Random Forest models to the next level!

## Ensemble Optimization

### 1. Stacking with Random Forests

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base estimators
estimators = [
    ('rf1', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ('rf2', RandomForestClassifier(n_estimators=100, max_features='sqrt')),
    ('rf3', RandomForestClassifier(n_estimators=100, min_samples_leaf=5))
]

# Create stacked model
stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# Train stacked model
stacked_model.fit(X_train, y_train)
```

### 2. Weighted Voting

```python
def weighted_voting_predict(models, weights, X):
    """Implement weighted voting for ensemble"""
    predictions = np.array([
        model.predict_proba(X) for model in models
    ])
    
    # Weight each model's predictions
    weighted_pred = np.average(
        predictions,
        weights=weights,
        axis=0
    )
    
    return np.argmax(weighted_pred, axis=1)
```

## Advanced Feature Engineering

### 1. Automated Feature Interactions

```python
from itertools import combinations

def create_feature_interactions(X, degree=2):
    """Create all possible feature interactions up to specified degree"""
    X = X.copy()
    feature_names = list(X.columns)
    
    for d in range(2, degree + 1):
        for combo in combinations(feature_names, d):
            # Create interaction feature
            name = '*'.join(combo)
            X[name] = 1
            for feature in combo:
                X[name] *= X[feature]
    
    return X
```

### 2. Feature Selection with Permutation Importance

```python
from sklearn.inspection import permutation_importance

def analyze_permutation_importance(model, X, y):
    """Calculate permutation importance with cross-validation"""
    result = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df
```

## Optimization Techniques

### 1. Dynamic Feature Selection

```python
class DynamicFeatureSelector:
    """Dynamically select features based on importance threshold"""
    def __init__(self, base_model, threshold=0.01):
        self.base_model = base_model
        self.threshold = threshold
        self.selected_features = None
    
    def fit(self, X, y):
        # Train base model
        self.base_model.fit(X, y)
        
        # Get feature importance
        importances = self.base_model.feature_importances_
        
        # Select features above threshold
        self.selected_features = X.columns[importances > self.threshold]
        
        # Retrain on selected features
        self.base_model.fit(X[self.selected_features], y)
        
        return self
    
    def predict(self, X):
        return self.base_model.predict(X[self.selected_features])
```

### 2. Memory-Efficient Implementation

```python
class MemoryEfficientRF:
    """Memory-efficient Random Forest implementation"""
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.trees = []
        
    def fit(self, X, y, batch_size=10):
        """Train trees in batches to save memory"""
        for i in range(0, self.n_estimators, batch_size):
            # Train batch of trees
            batch_trees = [
                RandomForestClassifier(n_estimators=1)
                for _ in range(min(batch_size, 
                                 self.n_estimators - i))
            ]
            
            # Fit each tree
            for tree in batch_trees:
                # Bootstrap sample
                idx = np.random.choice(
                    len(X), size=len(X), replace=True
                )
                tree.fit(X.iloc[idx], y.iloc[idx])
            
            self.trees.extend(batch_trees)
    
    def predict(self, X):
        """Predict using majority vote"""
        predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
```

## Advanced Evaluation Metrics

### 1. Custom Evaluation Framework

```python
class AdvancedRFEvaluator:
    """Advanced evaluation metrics for Random Forest"""
    def __init__(self, model):
        self.model = model
        
    def evaluate_stability(self, X, y, n_iterations=10):
        """Evaluate feature importance stability"""
        importance_matrices = []
        
        for _ in range(n_iterations):
            # Bootstrap sample
            idx = np.random.choice(len(X), size=len(X))
            X_boot, y_boot = X.iloc[idx], y.iloc[idx]
            
            # Fit model and get importance
            self.model.fit(X_boot, y_boot)
            importance_matrices.append(
                self.model.feature_importances_
            )
        
        # Calculate stability metrics
        importance_std = np.std(importance_matrices, axis=0)
        stability_score = 1 / (1 + np.mean(importance_std))
        
        return stability_score
    
    def feature_importance_confidence(self, X, y, 
                                    confidence_level=0.95):
        """Calculate confidence intervals for feature importance"""
        n_bootstrap = 1000
        n_features = X.shape[1]
        
        # Bootstrap feature importances
        importances = np.zeros((n_bootstrap, n_features))
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), size=len(X))
            X_boot, y_boot = X.iloc[idx], y.iloc[idx]
            
            # Get feature importance
            self.model.fit(X_boot, y_boot)
            importances[i] = self.model.feature_importances_
        
        # Calculate confidence intervals
        lower = np.percentile(importances, 
                            (1 - confidence_level) / 2 * 100, 
                            axis=0)
        upper = np.percentile(importances,
                            (1 + confidence_level) / 2 * 100,
                            axis=0)
        
        return pd.DataFrame({
            'feature': X.columns,
            'importance_mean': np.mean(importances, axis=0),
            'importance_lower': lower,
            'importance_upper': upper
        })
```

## Interpretability Techniques

### 1. Partial Dependence Plots

```python
from sklearn.inspection import partial_dependence

def plot_partial_dependence(model, X, feature_names):
    """Create partial dependence plots for specified features"""
    fig, axes = plt.subplots(
        len(feature_names), 1,
        figsize=(10, 5*len(feature_names))
    )
    
    for idx, feature in enumerate(feature_names):
        # Calculate partial dependence
        pdp = partial_dependence(
            model, X, [feature],
            kind='average'
        )
        
        # Plot
        axes[idx].plot(pdp[1][0], pdp[0][0])
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Partial dependence')
        
    plt.tight_layout()
    plt.show()
```

### 2. SHAP Values

```python
import shap

def analyze_shap_values(model, X):
    """Analyze SHAP values for feature importance"""
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # Plot summary
    shap.summary_plot(shap_values, X)
    
    # Return SHAP values for further analysis
    return shap_values
```

## Production Deployment

### 1. Model Versioning

```python
class VersionedRandomForest:
    """Random Forest with versioning capabilities"""
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        self.version = 1
        self.history = {}
        
    def fit(self, X, y):
        # Train model
        self.model.fit(X, y)
        
        # Save version info
        self.history[self.version] = {
            'timestamp': pd.Timestamp.now(),
            'n_samples': len(X),
            'feature_importance': dict(zip(
                X.columns,
                self.model.feature_importances_
            ))
        }
        
        self.version += 1
        return self
    
    def save_version(self, path):
        """Save model with version information"""
        save_dict = {
            'model': self.model,
            'version': self.version,
            'history': self.history
        }
        joblib.dump(save_dict, path)
```

### 2. Online Learning

```python
class OnlineRandomForest:
    """Random Forest with online learning capabilities"""
    def __init__(self, n_estimators=100, buffer_size=1000):
        self.n_estimators = n_estimators
        self.buffer_size = buffer_size
        self.buffer_X = []
        self.buffer_y = []
        self.model = None
        
    def partial_fit(self, X, y):
        """Update model with new data"""
        # Add to buffer
        self.buffer_X.extend(X.values)
        self.buffer_y.extend(y.values)
        
        # If buffer is full, retrain
        if len(self.buffer_X) >= self.buffer_size:
            # Convert to arrays
            X_train = np.array(self.buffer_X)
            y_train = np.array(self.buffer_y)
            
            # Train new model
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators
            ).fit(X_train, y_train)
            
            # Clear buffer
            self.buffer_X = []
            self.buffer_y = []
        
        return self
```

## Next Steps

Ready to see Random Forests in action? Continue to [Applications](5-applications.md) to explore real-world use cases!
