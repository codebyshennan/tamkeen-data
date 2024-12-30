# Naive Bayes Classification: A Deep Dive 🧮

## 1. Understanding Bayes' Theorem Visually

### The Core Equation Explained
```mermaid
graph LR
    A[Prior P(y)] --> D[Posterior P(y|X)]
    B[Likelihood P(X|y)] --> D
    C[Evidence P(X)] --> D
    D --> E[Final Classification]
```

Let's break down the equation $P(y|X) = \frac{P(X|y)P(y)}{P(X)}$ with a real-world example:

**Email Spam Classification Example:**
- Prior P(y): 30% of all emails are spam
- Likelihood P(X|y): 80% of spam emails contain the word "free"
- Evidence P(X): 40% of all emails contain the word "free"

```python
# Probability calculation example
prior_spam = 0.30  # P(y)
likelihood = 0.80  # P(X|y)
evidence = 0.40    # P(X)

posterior = (likelihood * prior_spam) / evidence
print(f"Probability email is spam: {posterior:.2%}")  # 60%
```

### Feature Independence Visualization
```mermaid
graph TD
    A[Class Label] --> B[Feature 1]
    A --> C[Feature 2]
    A --> D[Feature 3]
    B -.x.- C
    B -.x.- D
    C -.x.- D
    
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#bbf,stroke:#333
```
*Dotted X lines indicate assumed independence between features*

## 2. Types of Naive Bayes with Detailed Examples

### 2.1 Gaussian Naive Bayes
Used for continuous data that follows a normal distribution.

```python
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 100

# Create two classes with different distributions
class1_data = np.random.normal(loc=0, scale=1, size=(n_samples//2, 2))
class2_data = np.random.normal(loc=2, scale=1.5, size=(n_samples//2, 2))

X = np.vstack([class1_data, class2_data])
y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))

# Train model
model = GaussianNB()
model.fit(X, y)

# Visualize decision boundary
def plot_decision_boundary(X, y, model):
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Gaussian Naive Bayes Decision Boundary')
```

### 2.2 Multinomial Naive Bayes
Perfect for text classification with word frequencies.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Example: Document Classification
documents = [
    'python programming code development',
    'machine learning data science',
    'web development html css',
    'deep learning neural networks',
    'database sql queries'
]
labels = ['programming', 'data_science', 'web', 'data_science', 'database']

# Create document-term matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Visualize feature importance
def plot_feature_importance(vectorizer, model):
    feature_names = vectorizer.get_feature_names_out()
    for idx, label in enumerate(model.classes_):
        top_features = np.argsort(model.feature_log_prob_[idx])[-5:]
        print(f"\nTop features for {label}:")
        for feature_idx in top_features:
            print(f"{feature_names[feature_idx]}: {model.feature_log_prob_[idx][feature_idx]:.3f}")
```

### 2.3 Bernoulli Naive Bayes
Ideal for binary features (presence/absence).

```python
from sklearn.naive_bayes import BernoulliNB
import seaborn as sns

# Example: Movie Genre Classification with Binary Features
features = ['action', 'romance', 'comedy', 'special_effects', 'dialogue']
X = np.array([
    [1, 0, 0, 1, 0],  # Action movie
    [0, 1, 0, 0, 1],  # Romance movie
    [0, 0, 1, 0, 1],  # Comedy movie
    [1, 0, 0, 1, 0],  # Action movie
    [0, 1, 1, 0, 1]   # Romantic comedy
])
y = ['Action', 'Romance', 'Comedy', 'Action', 'Comedy']

# Visualize feature patterns
def plot_feature_patterns(X, y, features):
    plt.figure(figsize=(10, 6))
    sns.heatmap(X, annot=True, cmap='YlOrRd', 
                xticklabels=features, 
                yticklabels=y)
    plt.title('Feature Patterns by Genre')
```

## 3. Advanced Feature Engineering Techniques

### 3.1 Text Preprocessing Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                preprocessor=self.preprocess_text,
                ngram_range=(1, 2),
                max_features=1000,
                stop_words=stopwords.words('english')
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
```

### 3.2 Feature Selection Visualization
```python
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

def visualize_feature_importance(X, y, feature_names):
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y)
    
    # Sort features by importance
    sorted_idx = np.argsort(mi_scores)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(pos, mi_scores[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Mutual Information Score')
    plt.title('Feature Importance Ranking')
```

## 4. Performance Optimization

### 4.1 Cross-Validation with Learning Curves
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
```

### 4.2 Handling Class Imbalance
```python
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

def visualize_class_distribution(y):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Calculate and display class weights
    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    print("\nClass Weights:")
    for class_label, weight in zip(np.unique(y), weights):
        print(f"Class {class_label}: {weight:.2f}")
```

## 5. Common Pitfalls and Solutions

### 5.1 Zero Probability Problem
```python
# Example showing the zero probability problem
from sklearn.naive_bayes import MultinomialNB

# Data with zero frequency for some feature
X_zero_prob = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 0]
])
y_zero_prob = [0, 1, 1]

# Compare different smoothing parameters
alphas = [0.0, 0.1, 1.0, 10.0]
for alpha in alphas:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_zero_prob, y_zero_prob)
    print(f"\nAlpha = {alpha}")
    print("Feature probabilities:")
    print(model.feature_log_prob_)
```

### 5.2 Feature Independence Violation
```python
import seaborn as sns

def check_feature_correlation(X, feature_names):
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(X.T)
    sns.heatmap(correlation_matrix, 
                annot=True, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
```

## 6. Real-world Application Example: Sentiment Analysis

```python
class SentimentAnalyzer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df=2,
                max_df=0.95
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
    
    def analyze_sentiment(self, text, threshold=0.8):
        """Analyze sentiment with confidence score"""
        prob = self.pipeline.predict_proba([text])[0]
        sentiment = 'Positive' if prob[1] > 0.5 else 'Negative'
        confidence = max(prob)
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'is_reliable': confidence >= threshold
        }
    
    def visualize_confidence_distribution(self, texts):
        """Plot confidence score distribution"""
        probs = self.pipeline.predict_proba(texts)
        confidences = np.max(probs, axis=1)
        
        plt.figure(figsize=(8, 5))
        plt.hist(confidences, bins=20)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
```

This enhanced explanation provides:
1. Visual representations of key concepts
2. Detailed code examples with comments
3. Real-world applications
4. Performance optimization techniques
5. Common pitfalls and their solutions
6. Interactive visualizations for better understanding

The mermaid diagrams and code examples can be run to generate visual insights into how Naive Bayes works in practice.
