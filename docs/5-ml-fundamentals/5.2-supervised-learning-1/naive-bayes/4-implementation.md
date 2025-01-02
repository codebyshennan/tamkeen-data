# Implementing Naive Bayes in Practice 💻

Let's put our knowledge into practice with real-world implementations using scikit-learn. We'll cover common use cases and best practices.

## Text Classification Example 📝

### Spam Detection System

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset
emails = [
    "Get rich quick! Buy now!",
    "Meeting at 3pm tomorrow",
    "Win a free iPhone today!",
    "Project deadline reminder",
    "Free money, act now!",
    "Team lunch next week"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Create text classification pipeline
def create_text_classifier():
    """Create a pipeline for text classification"""
    return Pipeline([
        ('vectorizer', TfidfVectorizer(
            # Convert text to lowercase
            lowercase=True,
            # Remove common words like 'the', 'is'
            stop_words='english',
            # Include 1-word and 2-word phrases
            ngram_range=(1, 2),
            # Ignore rare words (appear in < 2 documents)
            min_df=2
        )),
        ('classifier', MultinomialNB(
            # Smoothing parameter
            alpha=1.0
        ))
    ])

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.2, random_state=42
)

# Create and train the model
model = create_text_classifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Example prediction
new_email = ["Congratulations! You've won a prize!"]
prediction = model.predict(new_email)
probability = model.predict_proba(new_email)

print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Not Spam'}")
print(f"Confidence: {max(probability[0]):.2%}")
```

> **TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects how important a word is to a document in a collection of documents.

## Medical Diagnosis Example 🏥

### Disease Prediction System

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Sample dataset
# Features: [temperature, heart_rate, blood_pressure, age]
patient_data = [
    [38.5, 90, 140, 45],
    [37.0, 70, 120, 30],
    [39.0, 95, 150, 55],
    [36.8, 75, 125, 35]
]
# Labels: 1 for sick, 0 for healthy
conditions = [1, 0, 1, 0]

def create_medical_classifier():
    """Create a pipeline for medical diagnosis"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    patient_data, conditions, test_size=0.2, random_state=42
)

# Create and train model
model = create_medical_classifier()
model.fit(X_train, y_train)

# Example prediction
new_patient = [[38.2, 85, 135, 40]]
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print(f"Diagnosis: {'Sick' if prediction[0] == 1 else 'Healthy'}")
print(f"Confidence: {max(probability[0]):.2%}")
```

## Product Categorization Example 🛍️

### Multi-class Classification

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Sample dataset
products = [
    "blue cotton t-shirt size M",
    "leather wallet black",
    "running shoes size 10",
    "denim jeans blue",
    "sports water bottle"
]
categories = ['Clothing', 'Accessories', 'Shoes', 'Clothing', 'Sports']

def create_product_classifier():
    """Create a pipeline for product categorization"""
    return Pipeline([
        ('vectorizer', CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('classifier', MultinomialNB())
    ])

# Encode categories
label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    products, encoded_categories, test_size=0.2, random_state=42
)

# Create and train model
model = create_product_classifier()
model.fit(X_train, y_train)

# Example prediction
new_product = ["white cotton socks pack"]
prediction = model.predict(new_product)
category = label_encoder.inverse_transform(prediction)

print(f"Predicted Category: {category[0]}")
```

## Best Practices and Tips 💡

### 1. Data Preprocessing

```python
def preprocess_text(text):
    """Common text preprocessing steps"""
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Example usage in a pipeline
class TextPreprocessor:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [preprocess_text(text) for text in X]

# Add to pipeline
pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
```

### 2. Handling Imbalanced Data

```python
from sklearn.utils.class_weight import compute_class_weight

def handle_imbalanced_data(X, y):
    """Handle imbalanced classes"""
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    
    # Create model with class weights
    model = MultinomialNB(class_prior=class_weights)
    
    return model
```

### 3. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y, cv=5):
    """Evaluate model using cross-validation"""
    
    # Calculate scores
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='accuracy'
    )
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
```

### 4. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, chi2

def select_best_features(X, y, k=10):
    """Select top k features using chi-square test"""
    
    # Create feature selector
    selector = SelectKBest(chi2, k=k)
    
    # Fit and transform
    X_new = selector.fit_transform(X, y)
    
    return X_new, selector
```

## Common Challenges and Solutions 🔧

### 1. Zero Probability Problem

```python
# Solution: Use additive (Laplace) smoothing
model = MultinomialNB(alpha=1.0)  # alpha is the smoothing parameter
```

### 2. High Dimensionality

```python
# Solution: Use feature selection or dimensionality reduction
from sklearn.decomposition import TruncatedSVD

def reduce_dimensions(X, n_components=100):
    """Reduce dimensionality using LSA"""
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd
```

### 3. Numeric Stability

```python
# Solution: Use log probabilities
def predict_log_proba(model, X):
    """Use log probabilities for numeric stability"""
    log_probs = model.predict_log_proba(X)
    return np.exp(log_probs)  # Convert back if needed
```

## Next Steps 📚

Now that you can implement Naive Bayes:
1. Explore [advanced topics](5-advanced-topics.md) for optimization
2. Try implementing hybrid solutions
3. Practice with real-world datasets

Remember:
- Start simple and iterate
- Monitor model performance
- Validate assumptions
- Test thoroughly before deployment
