# Implementing SVM with Scikit-learn 💻

Let's learn how to implement SVM for both classification and regression tasks using scikit-learn.

## Basic Classification Example 🎯

### Binary Classification

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np

# Create a simple pipeline
def create_svm_classifier(kernel='rbf', C=1.0):
    """Create SVM classification pipeline"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel=kernel,
            C=C,
            random_state=42
        ))
    ])

# Example usage
X = np.array([
    [1, 2], [2, 3], [3, 4], [2, 1],  # Class 0
    [5, 6], [6, 7], [7, 8], [6, 5]   # Class 1
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = create_svm_classifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

### Multiclass Classification

```python
def create_multiclass_classifier():
    """Create multiclass SVM classifier"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            decision_function_shape='ovo',  # one-vs-one
            probability=True
        ))
    ])

# Example: Iris Classification
from sklearn.datasets import load_iris

def iris_classification_example():
    """Complete example using Iris dataset"""
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2
    )
    
    # Create and train model
    model = create_multiclass_classifier()
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Classification Report:")
    print(classification_report(
        y_test,
        model.predict(X_test),
        target_names=iris.target_names
    ))
    
    return model
```

## Regression with SVM 📈

> **SVR (Support Vector Regression)** uses the same principles as SVM, but for predicting continuous values.

```python
from sklearn.svm import SVR

def create_svm_regressor(kernel='rbf'):
    """Create SVM regression pipeline"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(
            kernel=kernel,
            C=1.0,
            epsilon=0.1
        ))
    ])

# Example: Housing Price Prediction
def housing_price_example():
    """Predict housing prices using SVR"""
    # Sample data: [size, bedrooms, age]
    X = np.array([
        [1400, 3, 10],
        [1600, 3, 8],
        [1700, 4, 15],
        [1875, 4, 5],
        [1100, 2, 20]
    ])
    
    # Prices (in thousands)
    y = np.array([250, 280, 300, 350, 200])
    
    # Create and train model
    model = create_svm_regressor()
    model.fit(X, y)
    
    # Example prediction
    new_house = np.array([[1500, 3, 12]])
    prediction = model.predict(new_house)
    print(f"Predicted price: ${prediction[0]:.2f}k")
    
    return model
```

## Parameter Tuning 🔧

### Grid Search for Optimal Parameters

```python
from sklearn.model_selection import GridSearchCV

def optimize_svm_parameters(X, y, cv=5):
    """Find optimal SVM parameters"""
    # Parameter grid
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 1],
        'svm__kernel': ['rbf', 'linear', 'poly']
    }
    
    # Create base model
    model = create_svm_classifier()
    
    # Grid search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit and print results
    grid_search.fit(X, y)
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    return grid_search.best_estimator_
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

def evaluate_svm_model(model, X, y, cv=5):
    """Evaluate model using cross-validation"""
    # Calculate scores
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='accuracy'
    )
    
    print("Cross-validation scores:", scores)
    print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Feature Engineering 🛠️

### Text Classification Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer

class SVMTextClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )),
            ('scaler', StandardScaler(with_mean=False)),
            ('svm', SVC(kernel='linear'))
        ])
        
    def train(self, texts, labels):
        """Train the text classifier"""
        self.pipeline.fit(texts, labels)
        
    def predict(self, texts):
        """Make predictions"""
        return self.pipeline.predict(texts)
        
    def analyze_features(self):
        """Analyze important features"""
        tfidf = self.pipeline.named_steps['tfidf']
        svm = self.pipeline.named_steps['svm']
        
        # Get feature names and coefficients
        feature_names = tfidf.get_feature_names_out()
        coefficients = svm.coef_[0]
        
        # Sort by importance
        importance = np.abs(coefficients)
        sorted_idx = np.argsort(importance)
        pos = sorted_idx[-10:]  # Top 10 features
        
        return [(feature_names[i], coefficients[i]) 
                for i in pos]
```

## Handling Imbalanced Data ⚖️

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def create_balanced_svm():
    """Create SVM pipeline for imbalanced data"""
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE()),
        ('svm', SVC(
            class_weight='balanced',
            probability=True
        ))
    ])

# Example usage
def handle_imbalanced_data(X, y):
    """Train SVM on imbalanced dataset"""
    # Create and train model
    model = create_balanced_svm()
    model.fit(X, y)
    
    # Evaluate with appropriate metrics
    from sklearn.metrics import balanced_accuracy_score
    y_pred = model.predict(X)
    print("Balanced accuracy:", 
          balanced_accuracy_score(y, y_pred))
```

## Best Practices 📚

### 1. Data Preprocessing

```python
def preprocess_data(X, categorical_features=[]):
    """Preprocess data for SVM"""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), 
             [i for i in range(X.shape[1]) 
              if i not in categorical_features]),
            ('cat', OneHotEncoder(drop='first'), 
             categorical_features)
        ])
    
    return preprocessor
```

### 2. Model Selection

```python
def select_best_model(X, y):
    """Select best SVM configuration"""
    models = {
        'linear': create_svm_classifier(kernel='linear'),
        'rbf': create_svm_classifier(kernel='rbf'),
        'poly': create_svm_classifier(kernel='poly')
    }
    
    results = {}
    for name, model in models.items():
        score = cross_val_score(model, X, y, cv=5).mean()
        results[name] = score
        
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"Best model: {best_model[0]} "
          f"(score: {best_model[1]:.3f})")
    
    return models[best_model[0]]
```

## Common Pitfalls and Solutions ⚠️

1. **Memory Issues**
   ```python
   from sklearn.svm import LinearSVC
   
   # Use LinearSVC for large datasets
   model = LinearSVC(dual=False)
   ```

2. **Slow Training**
   ```python
   # Use smaller subset for parameter tuning
   X_sample, _, y_sample, _ = train_test_split(
       X, y, train_size=0.1, random_state=42
   )
   best_params = optimize_svm_parameters(X_sample, y_sample)
   ```

3. **Poor Performance**
   ```python
   # Try different preprocessing
   from sklearn.preprocessing import RobustScaler
   
   pipeline = Pipeline([
       ('scaler', RobustScaler()),  # More robust to outliers
       ('svm', SVC())
   ])
   ```

## Next Steps 📚

Now that you can implement SVM:
1. Explore [advanced techniques](4-advanced.md)
2. Learn about [real-world applications](5-applications.md)
3. Practice with different datasets
4. Experiment with various kernels

Remember:
- Always scale your features
- Use cross-validation
- Start with simple kernels
- Monitor training time
