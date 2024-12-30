# Scikit-learn Pipelines

Imagine building a car assembly line - each step needs to happen in the right order, and you want to be able to replicate the process exactly. That's what scikit-learn pipelines do for machine learning workflows! Let's learn how to build efficient and reproducible pipelines. 🏭

## Understanding Pipelines 🎯

Pipelines help us:
1. Ensure preprocessing steps are consistent
2. Prevent data leakage
3. Simplify model deployment
4. Make code more maintainable

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create sample dataset
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.normal(35, 10, n_samples)
income = np.random.exponential(50000, n_samples)
credit_score = np.random.normal(700, 100, n_samples)

X = np.column_stack([age, income, credit_score])
y = (credit_score + income/1000 + age > 800).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create simple pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit and evaluate
pipeline.fit(X_train, y_train)
print(f"Pipeline score: {pipeline.score(X_test, y_test):.3f}")
```

## Building Complex Pipelines 🔧

### Feature Unions
Combine multiple feature processing steps:

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

def create_feature_union_pipeline():
    # Create feature processors
    feature_processing = FeatureUnion([
        ('pca', PCA(n_components=2)),
        ('select_best', SelectKBest(k=2))
    ])
    
    # Create full pipeline
    pipeline = Pipeline([
        ('features', feature_processing),
        ('classifier', LogisticRegression())
    ])
    
    return pipeline

# Create and use pipeline
union_pipeline = create_feature_union_pipeline()
union_pipeline.fit(X_train, y_train)
print(f"Feature union score: {union_pipeline.score(X_test, y_test):.3f}")
```

### Custom Transformers
Create your own preprocessing steps:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        # Calculate z-scores for each feature
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        # Replace outliers with mean values
        z_scores = np.abs((X - self.mean_) / self.std_)
        mask = z_scores > self.threshold
        X_copy = X.copy()
        X_copy[mask] = np.take(self.mean_, range(X.shape[1]))
        return X_copy

# Use custom transformer in pipeline
pipeline_with_custom = Pipeline([
    ('outlier_handler', OutlierHandler(threshold=3)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline_with_custom.fit(X_train, y_train)
print(f"Custom pipeline score: {pipeline_with_custom.score(X_test, y_test):.3f}")
```

## Real-World Example: Text Classification 📝

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import re

# Sample text data
texts = [
    "Machine learning is fascinating",
    "Deep neural networks are powerful",
    "Data science is growing rapidly",
    "AI transforms industries"
]
labels = [1, 1, 1, 1]  # Positive class for all

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Create text processing pipeline
text_pipeline = Pipeline([
    ('preprocessor', FunctionTransformer(lambda x: [preprocess_text(text) for text in x])),
    ('vectorizer', TfidfVectorizer(max_features=1000)),
    ('classifier', LogisticRegression())
])

# Split text data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Fit pipeline
text_pipeline.fit(X_train, y_train)

# Make predictions
predictions = text_pipeline.predict(X_test)
```

## Pipeline Persistence 💾

Save and load pipelines:

```python
import joblib

def save_pipeline(pipeline, filename):
    """Save pipeline to file"""
    joblib.dump(pipeline, filename)

def load_pipeline(filename):
    """Load pipeline from file"""
    return joblib.load(filename)

# Example usage
save_pipeline(pipeline, 'model_pipeline.joblib')
loaded_pipeline = load_pipeline('model_pipeline.joblib')
```

## Advanced Techniques 🚀

### 1. Memory Caching
```python
from sklearn.pipeline import Pipeline
from sklearn.externals import Memory

# Set up caching
memory = Memory(location='./cachedir', verbose=0)

# Create pipeline with caching
cached_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
], memory=memory)
```

### 2. Parameter Grid Search
```python
from sklearn.model_selection import GridSearchCV

# Define parameters for multiple steps
param_grid = {
    'scaler__with_mean': [True, False],
    'pca__n_components': [2, 3, 4],
    'classifier__C': [0.1, 1.0, 10.0]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 3. Column Transformer
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1]),  # Numerical columns
        ('cat', OneHotEncoder(), [2])       # Categorical columns
    ])

# Create pipeline with column transformer
column_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

## Best Practices 🌟

### 1. Naming Conventions
```python
# Use descriptive names for steps
pipeline = Pipeline([
    ('missing_handler', SimpleImputer()),
    ('feature_scaler', StandardScaler()),
    ('dim_reducer', PCA()),
    ('classifier', LogisticRegression())
])
```

### 2. Error Handling
```python
class RobustTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        try:
            # Transformation logic
            return transformed_X
        except Exception as e:
            print(f"Error in transformation: {e}")
            # Return safe fallback
            return X
```

### 3. Validation
```python
from sklearn.model_selection import cross_val_score

def validate_pipeline(pipeline, X, y, cv=5):
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv)
    
    # Print results
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Common Pitfalls and Solutions 🚧

1. **Data Leakage**
   - Keep all preprocessing in pipeline
   - Use ColumnTransformer for mixed data
   - Validate transformation order

2. **Memory Issues**
   - Use memory caching
   - Implement batch processing
   - Monitor memory usage

3. **Performance**
   - Profile pipeline steps
   - Optimize transformers
   - Use parallel processing

## Next Steps

Now that you understand scikit-learn pipelines, try the [assignment](./assignment.md) to practice building efficient machine learning workflows!
