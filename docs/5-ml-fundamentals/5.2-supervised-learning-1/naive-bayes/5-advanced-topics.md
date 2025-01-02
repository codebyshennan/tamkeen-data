# Advanced Topics in Naive Bayes 🚀

## Feature Engineering Techniques 🛠️

### 1. Text Feature Engineering

> **Feature Engineering** is the process of creating new features from existing data to improve model performance.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.stem import WordNetLemmatizer

class AdvancedTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, text):
        """Advanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
        
        # Lemmatization (convert words to base form)
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

class CustomVectorizer(TfidfVectorizer):
    def __init__(self, preprocessor=None, **kwargs):
        super().__init__(preprocessor=preprocessor, **kwargs)
        
    def get_feature_names_out(self):
        """Get feature names for interpretation"""
        return super().get_feature_names_out()

# Example usage
preprocessor = AdvancedTextPreprocessor()
pipeline = Pipeline([
    ('vectorizer', CustomVectorizer(
        preprocessor=preprocessor.preprocess,
        ngram_range=(1, 3),
        max_features=1000
    )),
    ('classifier', MultinomialNB())
])
```

### 2. Numerical Feature Engineering

> **Feature Scaling** is particularly important for Gaussian Naive Bayes to ensure all features contribute appropriately to the model.

```python
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer

def create_advanced_numerical_pipeline():
    """Create pipeline with advanced numerical preprocessing"""
    numeric_features = ['age', 'income', 'credit_score']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', PowerTransformer(method='yeo-johnson'), numeric_features)
        ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
```

## Handling Missing Data 🕳️

### 1. Advanced Imputation Strategies

> **Imputation** is the process of replacing missing values with substituted values.

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class AdvancedImputer:
    def __init__(self, strategy='knn'):
        self.strategy = strategy
        
    def get_imputer(self):
        """Get appropriate imputer based on strategy"""
        if self.strategy == 'knn':
            return KNNImputer(n_neighbors=5)
        elif self.strategy == 'iterative':
            return IterativeImputer(max_iter=10)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

# Example usage
def create_pipeline_with_imputation():
    """Create pipeline with advanced imputation"""
    return Pipeline([
        ('imputer', AdvancedImputer(strategy='iterative').get_imputer()),
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])
```

## Ensemble Methods 🤝

### 1. Voting Classifier

> **Ensemble Methods** combine multiple models to create a more robust classifier.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

def create_naive_bayes_ensemble():
    """Create an ensemble of different Naive Bayes classifiers"""
    classifiers = [
        ('multinomial', MultinomialNB()),
        ('gaussian', GaussianNB()),
        ('bernoulli', BernoulliNB())
    ]
    
    return VotingClassifier(
        estimators=classifiers,
        voting='soft'  # Use probability estimates
    )
```

### 2. Stacking

```python
from sklearn.ensemble import StackingClassifier

def create_stacked_classifier():
    """Create a stacked classifier with Naive Bayes models"""
    estimators = [
        ('mnb', MultinomialNB()),
        ('gnb', GaussianNB()),
        ('bnb', BernoulliNB())
    ]
    
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
```

## Production Deployment 🌐

### 1. Model Serialization

```python
import joblib
import json

class ModelSerializer:
    def __init__(self, model, metadata=None):
        self.model = model
        self.metadata = metadata or {}
        
    def save(self, path):
        """Save model and metadata"""
        model_path = f"{path}/model.joblib"
        meta_path = f"{path}/metadata.json"
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f)
            
    @classmethod
    def load(cls, path):
        """Load model and metadata"""
        model = joblib.load(f"{path}/model.joblib")
        
        with open(f"{path}/metadata.json", 'r') as f:
            metadata = json.load(f)
            
        return cls(model, metadata)
```

### 2. Model Monitoring

```python
import numpy as np
from datetime import datetime

class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.timestamps = []
        self.feature_stats = {}
        
    def log_prediction(self, features, prediction, actual=None):
        """Log prediction details"""
        self.predictions.append({
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now()
        })
        
    def check_drift(self, window_size=100):
        """Check for concept drift"""
        if len(self.predictions) < window_size:
            return False
            
        recent = self.predictions[-window_size:]
        
        # Calculate drift metrics
        drift_score = self._calculate_drift_score(recent)
        
        return drift_score > 0.1  # Threshold for drift detection
        
    def _calculate_drift_score(self, predictions):
        """Calculate drift score based on prediction patterns"""
        # Implementation depends on specific needs
        pass
```

## Advanced Optimization Techniques 🔧

### 1. Hyperparameter Optimization

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def optimize_naive_bayes(X, y):
    """Optimize Naive Bayes hyperparameters"""
    # Parameter space
    param_dist = {
        'vectorizer__max_features': randint(100, 10000),
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'classifier__alpha': uniform(0.1, 2.0)
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Random search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    search.fit(X, y)
    return search.best_estimator_
```

### 2. Custom Probability Calibration

> **Probability Calibration** ensures that the predicted probabilities accurately reflect the true likelihood of each class.

```python
from sklearn.calibration import CalibratedClassifierCV

def create_calibrated_classifier(base_model):
    """Create a calibrated version of the classifier"""
    return CalibratedClassifierCV(
        base_model,
        method='sigmoid',  # or 'isotonic'
        cv=5
    )
```

## Performance Optimization 🏃‍♂️

### 1. Memory Efficiency

```python
from sklearn.feature_extraction.text import HashingVectorizer

def create_memory_efficient_pipeline():
    """Create a memory-efficient pipeline for large datasets"""
    return Pipeline([
        ('vectorizer', HashingVectorizer(
            n_features=2**10,  # Smaller feature space
            alternate_sign=False  # Non-negative features
        )),
        ('classifier', MultinomialNB())
    ])
```

### 2. Computation Efficiency

```python
import numpy as np
from joblib import Parallel, delayed

class EfficientNaiveBayes:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        
    def parallel_predict(self, X, batch_size=1000):
        """Make predictions in parallel"""
        def predict_batch(batch):
            return self.model.predict_proba(batch)
            
        # Split into batches
        batches = np.array_split(X, len(X) // batch_size + 1)
        
        # Parallel prediction
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_batch)(batch) for batch in batches
        )
        
        return np.vstack(results)
```

## Next Steps 🎯

After mastering these advanced topics:
1. Explore other algorithms in the [supervised learning](../README.md) section
2. Practice with real-world datasets
3. Participate in machine learning competitions
4. Stay updated with the latest research and techniques

Remember:
- Advanced techniques should be used judiciously
- Always validate assumptions
- Monitor model performance in production
- Keep learning and experimenting!
