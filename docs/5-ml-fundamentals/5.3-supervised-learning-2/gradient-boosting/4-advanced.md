# Advanced Gradient Boosting Techniques 🚀

Let's explore advanced concepts and techniques to take your Gradient Boosting models to the next level!

## Advanced Model Architectures 🏗️

### 1. Multi-Output Gradient Boosting
```python
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

def train_multi_output_model(X, y_multiple):
    """Train model for multiple outputs"""
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
    )
    model.fit(X, y_multiple)
    return model
```

### 2. Hierarchical Gradient Boosting
```python
class HierarchicalGBM:
    """Hierarchical Gradient Boosting for nested categories"""
    def __init__(self, levels):
        self.levels = levels
        self.models = {}
        
    def fit(self, X, y_hierarchy):
        """Train hierarchical models"""
        for level in self.levels:
            # Train model for current level
            self.models[level] = XGBClassifier()
            self.models[level].fit(
                X, 
                y_hierarchy[level],
                sample_weight=self._get_weights(level, y_hierarchy)
            )
    
    def _get_weights(self, level, y_hierarchy):
        """Get sample weights based on hierarchy"""
        weights = np.ones(len(y_hierarchy))
        if level > 0:
            # Increase weights for samples that were correct
            # at previous level
            prev_correct = (
                self.models[level-1].predict(X) == 
                y_hierarchy[level-1]
            )
            weights[prev_correct] *= 2
        return weights
```

## Advanced Loss Functions 📉

### 1. Custom Loss Function
```python
def custom_objective(y_true, y_pred):
    """Custom objective function for XGBoost"""
    # Calculate gradients
    grad = 2 * (y_pred - y_true)
    
    # Calculate hessians
    hess = 2 * np.ones_like(y_pred)
    
    return grad, hess

# Use custom objective
params = {
    'objective': custom_objective,
    'max_depth': 3
}
```

### 2. Weighted Loss
```python
def weighted_log_loss(y_true, y_pred, weights):
    """Weighted logarithmic loss"""
    return -np.mean(
        weights * (
            y_true * np.log(y_pred) + 
            (1 - y_true) * np.log(1 - y_pred)
        )
    )
```

## Advanced Feature Engineering 🔧

### 1. Automated Feature Interactions
```python
def create_interactions(X, degree=2):
    """Create feature interactions up to specified degree"""
    from itertools import combinations
    
    X = X.copy()
    features = list(X.columns)
    
    for d in range(2, degree + 1):
        for combo in combinations(features, d):
            name = '*'.join(combo)
            X[name] = 1
            for feature in combo:
                X[name] *= X[feature]
    
    return X
```

### 2. Time-Based Features
```python
def create_time_features(df, date_column):
    """Create time-based features"""
    df = df.copy()
    df['hour'] = df[date_column].dt.hour
    df['day'] = df[date_column].dt.day
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    df['dayofweek'] = df[date_column].dt.dayofweek
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    return df
```

## Advanced Training Techniques 🎓

### 1. Learning Rate Scheduling
```python
class LearningRateScheduler:
    """Dynamic learning rate scheduler"""
    def __init__(self, initial_lr=0.1, decay=0.995):
        self.initial_lr = initial_lr
        self.decay = decay
        self.iteration = 0
    
    def __call__(self):
        """Calculate current learning rate"""
        lr = self.initial_lr * (self.decay ** self.iteration)
        self.iteration += 1
        return lr

# Use with XGBoost
scheduler = LearningRateScheduler()
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    callbacks=[
        xgb.callback.reset_learning_rate(scheduler)
    ]
)
```

### 2. Custom Splitting Criteria
```python
def custom_split_evaluator(y_left, y_right):
    """Custom split evaluation function"""
    # Calculate statistics for left and right nodes
    n_left = len(y_left)
    n_right = len(y_right)
    
    # Calculate impurity reduction
    impurity_left = calculate_impurity(y_left)
    impurity_right = calculate_impurity(y_right)
    
    # Weight by sample size
    weighted_impurity = (
        (n_left * impurity_left + n_right * impurity_right) /
        (n_left + n_right)
    )
    
    return weighted_impurity
```

## Advanced Regularization 🎛️

### 1. Feature-Level Regularization
```python
class FeatureRegularizer:
    """Apply different regularization to different features"""
    def __init__(self, feature_penalties):
        self.feature_penalties = feature_penalties
    
    def __call__(self, weights):
        """Calculate regularization term"""
        reg_term = 0
        for feature, penalty in self.feature_penalties.items():
            reg_term += penalty * np.abs(weights[feature])
        return reg_term
```

### 2. Adaptive Regularization
```python
class AdaptiveRegularizer:
    """Adjust regularization based on validation performance"""
    def __init__(self, initial_lambda=1.0):
        self.lambda_ = initial_lambda
        self.best_score = float('inf')
    
    def update(self, val_score):
        """Update regularization strength"""
        if val_score < self.best_score:
            self.best_score = val_score
            self.lambda_ *= 0.9  # Reduce regularization
        else:
            self.lambda_ *= 1.1  # Increase regularization
```

## Advanced Model Analysis 📊

### 1. Partial Dependence Analysis
```python
def calculate_partial_dependence(model, X, feature, grid_points=50):
    """Calculate partial dependence for a feature"""
    # Create grid of values
    feature_values = np.linspace(
        X[feature].min(),
        X[feature].max(),
        grid_points
    )
    
    # Calculate predictions
    predictions = []
    for value in feature_values:
        X_modified = X.copy()
        X_modified[feature] = value
        pred = model.predict(X_modified)
        predictions.append(pred.mean())
    
    return feature_values, predictions
```

### 2. SHAP Value Analysis
```python
import shap

def analyze_shap_interactions(model, X):
    """Analyze feature interactions using SHAP"""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Calculate interaction values
    interaction_values = explainer.shap_interaction_values(X)
    
    # Create interaction matrix
    n_features = X.shape[1]
    interaction_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            interaction_matrix[i, j] = np.abs(
                interaction_values[:, i, j]
            ).mean()
    
    return pd.DataFrame(
        interaction_matrix,
        index=X.columns,
        columns=X.columns
    )
```

## Production Deployment 🚀

### 1. Model Versioning
```python
class VersionedGBM:
    """Gradient Boosting with versioning"""
    def __init__(self):
        self.versions = {}
        self.current_version = 0
    
    def train_new_version(self, X, y, params):
        """Train and store new model version"""
        model = xgb.train(params, xgb.DMatrix(X, y))
        
        self.current_version += 1
        self.versions[self.current_version] = {
            'model': model,
            'params': params,
            'timestamp': pd.Timestamp.now(),
            'metrics': self._calculate_metrics(model, X, y)
        }
        
        return self.current_version
    
    def _calculate_metrics(self, model, X, y):
        """Calculate model metrics"""
        predictions = model.predict(xgb.DMatrix(X))
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions)
        }
```

### 2. Online Learning
```python
class OnlineGBM:
    """Gradient Boosting with online learning"""
    def __init__(self, base_model, buffer_size=1000):
        self.base_model = base_model
        self.buffer_size = buffer_size
        self.buffer_X = []
        self.buffer_y = []
    
    def partial_fit(self, X, y):
        """Update model with new data"""
        # Add to buffer
        self.buffer_X.extend(X)
        self.buffer_y.extend(y)
        
        # Retrain if buffer is full
        if len(self.buffer_X) >= self.buffer_size:
            X_train = np.vstack(self.buffer_X)
            y_train = np.array(self.buffer_y)
            
            self.base_model.fit(
                X_train, y_train,
                xgb_model=self.base_model.get_booster()
            )
            
            # Clear buffer
            self.buffer_X = []
            self.buffer_y = []
```

## Next Steps 🎯

Ready to see Gradient Boosting in action? Continue to [Applications](5-applications.md) to explore real-world use cases!
