# Advanced Topics in Naive Bayes

## Welcome to Advanced Naive Bayes! 🎯

Now that you've mastered the basics, let's explore some advanced techniques that will make your Naive Bayes models even better. Think of this as adding special tools to your machine learning toolbox!

## 1. Feature Engineering: Making Your Data Work Better

### What is Feature Engineering?

Feature engineering is like being a chef who transforms basic ingredients into a delicious meal. You take your raw data and transform it into features that help your model make better predictions.

### Text Feature Engineering

Let's say you're building a spam detector. Instead of just using raw words, you can create smarter features:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer

class SmartTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, text):
        """Transform text into better features"""
        # Convert to lowercase
        text = text.lower()
        
        # Keep important punctuation (! ? .)
        text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
        
        # Convert words to their base form
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

# Use in your pipeline
pipeline = Pipeline([
    ('preprocessor', SmartTextPreprocessor()),
    ('vectorizer', TfidfVectorizer(
        ngram_range=(1, 3),  # Look at words, pairs, and triplets
        max_features=1000    # Keep the most important words
    )),
    ('classifier', MultinomialNB())
])
```

### Numerical Feature Engineering

When working with numbers (like age or income), you can transform them to better fit the Gaussian distribution:

```python
from sklearn.preprocessing import PowerTransformer

def transform_numerical_features():
    """Create better numerical features"""
    return Pipeline([
        ('transformer', PowerTransformer(
            method='yeo-johnson'  # Handles positive and negative numbers
        )),
        ('classifier', GaussianNB())
    ])
```

## 2. Handling Missing Data: Don't Let Gaps Stop You

### Why Missing Data Matters

Imagine you're a doctor with incomplete patient records. You can't just ignore missing information - you need to handle it smartly!

### Smart Ways to Handle Missing Data

```python
from sklearn.impute import KNNImputer

class SmartDataImputer:
    def __init__(self, strategy='knn'):
        self.strategy = strategy
        
    def impute(self, data):
        """Fill in missing values intelligently"""
        if self.strategy == 'knn':
            # Use similar patients to fill in missing values
            imputer = KNNImputer(n_neighbors=5)
        else:
            # Use iterative approach
            imputer = IterativeImputer(max_iter=10)
            
        return imputer.fit_transform(data)

# Use in your pipeline
pipeline = Pipeline([
    ('imputer', SmartDataImputer(strategy='knn')),
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])
```

## 3. Ensemble Methods: Teamwork Makes the Dream Work

### What are Ensembles?

An ensemble is like a team of experts working together. Instead of relying on one model, we combine multiple models to get better predictions.

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

def create_naive_bayes_team():
    """Create a team of Naive Bayes models"""
    models = [
        ('multinomial', MultinomialNB()),  # For text
        ('gaussian', GaussianNB()),        # For numbers
        ('bernoulli', BernoulliNB())       # For yes/no features
    ]
    
    return VotingClassifier(
        estimators=models,
        voting='soft'  # Use probability estimates
    )
```

### Stacking Classifier

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def create_stacked_model():
    """Create a stacked model with Naive Bayes"""
    base_models = [
        ('mnb', MultinomialNB()),
        ('gnb', GaussianNB()),
        ('bnb', BernoulliNB())
    ]
    
    return StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5  # Use 5-fold cross-validation
    )
```

## 4. Model Deployment: Taking Your Model to the Real World

### Saving Your Model

```python
import joblib
import json

class ModelSaver:
    def __init__(self, model, info=None):
        self.model = model
        self.info = info or {}
        
    def save(self, folder):
        """Save model and its information"""
        # Save the model
        joblib.dump(self.model, f"{folder}/model.joblib")
        
        # Save additional information
        with open(f"{folder}/model_info.json", 'w') as f:
            json.dump(self.info, f)
            
    @classmethod
    def load(cls, folder):
        """Load a saved model"""
        model = joblib.load(f"{folder}/model.joblib")
        with open(f"{folder}/model_info.json", 'r') as f:
            info = json.load(f)
        return cls(model, info)
```

### Monitoring Your Model

```python
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.timestamps = []
        
    def track_prediction(self, features, prediction, actual=None):
        """Keep track of model predictions"""
        self.predictions.append({
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'time': datetime.now()
        })
        
    def check_performance(self, window=100):
        """Check recent model performance"""
        if len(self.predictions) < window:
            return "Not enough data"
            
        recent = self.predictions[-window:]
        accuracy = sum(1 for p in recent if p['prediction'] == p['actual']) / window
        return f"Recent accuracy: {accuracy:.2%}"
```

## 5. Hyperparameter Tuning: Finding the Best Settings

### What are Hyperparameters?

Hyperparameters are like the settings on your camera. You need to adjust them to get the best results for each situation.

### Finding the Best Settings

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def find_best_settings(X, y):
    """Find the best hyperparameters"""
    # Define what settings to try
    param_options = {
        'vectorizer__max_features': randint(100, 10000),
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'classifier__alpha': uniform(0.1, 2.0)
    }
    
    # Create the model
    model = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Search for best settings
    search = RandomizedSearchCV(
        model, param_options,
        n_iter=20,  # Try 20 different combinations
        cv=5,       # Use 5-fold cross-validation
        scoring='accuracy'
    )
    
    # Find the best settings
    search.fit(X, y)
    return search.best_params_
```

## Common Advanced Challenges and Solutions

### 1. Dealing with Class Imbalance

When one class is much more common than others:

```python
# Use class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
model = MultinomialNB(class_prior=class_weights)
```

### 2. Handling High-Dimensional Data

When you have too many features:

```python
# Use feature selection
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=1000)  # Keep top 1000 features
X_new = selector.fit_transform(X, y)
```

### 3. Improving Numeric Stability

When dealing with very small probabilities:

```python
# Use log probabilities
log_probs = model.predict_log_proba(X)
predictions = np.argmax(log_probs, axis=1)
```

## Next Steps 📚

Ready to become a Naive Bayes expert? Try these challenges:

1. Implement feature engineering in your own project
2. Experiment with different ensemble methods
3. Deploy a model and monitor its performance
4. Try hyperparameter tuning on a real dataset

Remember: The best way to learn is by doing! Start with small experiments and gradually tackle more complex problems.
