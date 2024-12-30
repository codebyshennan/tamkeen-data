# Naive Bayes Classification 🧮

Naive Bayes is a probabilistic classifier based on Bayes' Theorem with an assumption of independence between features. Think of it as a doctor diagnosing patients - different symptoms (features) contribute independently to the probability of various diseases (classes).

## Mathematical Foundation 📐

### Bayes' Theorem
The core equation:

$$P(y|X) = \frac{P(X|y)P(y)}{P(X)}$$

Where:
- $P(y|X)$ is the posterior probability of class $y$ given features $X$
- $P(X|y)$ is the likelihood of features $X$ given class $y$
- $P(y)$ is the prior probability of class $y$
- $P(X)$ is the evidence (normalization factor)

### The "Naive" Assumption
Features are conditionally independent given the class:

$$P(X|y) = P(x_1|y) \times P(x_2|y) \times ... \times P(x_n|y)$$

This leads to the classification rule:

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i|y)$$

## Types of Naive Bayes 🔍

### 1. Gaussian Naive Bayes
For continuous features following normal distribution:

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

```python
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Example: Student Performance Prediction
X = np.array([
    [5, 8, 75],  # [study_hours, sleep_hours, previous_score]
    [4, 7, 65],
    [3, 6, 55],
    [8, 9, 85]
])
y = np.array([1, 1, 0, 1])  # 1=pass, 0=fail

# Create and train model
model = GaussianNB()
model.fit(X, y)

# Predict for a new student
new_student = np.array([[6, 8, 70]])
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print(f"Will student pass? {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability of passing: {probability[0][1]:.2%}")
```

### 2. Multinomial Naive Bayes
For discrete features (e.g., word counts):

$$P(x_i|y) = \frac{\text{count}(x_i, y) + \alpha}{\sum_{w \in V} (\text{count}(w, y) + \alpha)}$$

Where $\alpha$ is the smoothing parameter (Laplace smoothing).

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Example: Email Classification
emails = [
    'win money lottery prize',
    'meeting schedule tomorrow',
    'claim free gift now',
    'project deadline update'
]
labels = [1, 0, 1, 0]  # 1=spam, 0=not spam

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1, 2)  # Include both unigrams and bigrams
)
X = vectorizer.fit_transform(emails)

# Train model with Laplace smoothing
model = MultinomialNB(alpha=1.0)
model.fit(X, labels)

# Predict new email
new_email = ['urgent claim your prize money now']
X_new = vectorizer.transform(new_email)
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)

print(f"Is spam? {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability of spam: {probability[0][1]:.2%}")
```

### 3. Bernoulli Naive Bayes
For binary features:

$$P(x_i|y) = P(i|y)^{x_i} (1-P(i|y))^{1-x_i}$$

```python
from sklearn.naive_bayes import BernoulliNB

# Example: Movie Genre Classification
X = np.array([
    [1, 1, 0, 0, 1],  # Action movie features
    [0, 0, 1, 1, 0],  # Romance movie features
    [1, 0, 0, 1, 1]   # Mixed features
])
# Features: [explosions, fights, romance, kissing, special_effects]
y = ['Action', 'Romance', 'Action']

# Train model
model = BernoulliNB(alpha=1.0)
model.fit(X, y)

# Predict new movie
new_movie = np.array([[1, 1, 0, 1, 1]])
prediction = model.predict(new_movie)
probability = model.predict_proba(new_movie)

print(f"Predicted genre: {prediction[0]}")
print(f"Probability distribution:")
for genre, prob in zip(model.classes_, probability[0]):
    print(f"{genre}: {prob:.2%}")
```

## Advanced Applications 🚀

### Sentiment Analysis with Feature Engineering
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

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
        
    def train(self, texts, labels):
        """Train the sentiment analyzer"""
        self.pipeline.fit(texts, labels)
        
    def predict(self, texts):
        """Predict sentiment with confidence"""
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        return list(zip(predictions, probabilities))
        
    def evaluate(self, texts, labels, cv=5):
        """Evaluate model performance"""
        scores = cross_val_score(
            self.pipeline, texts, labels, 
            cv=cv, scoring='accuracy'
        )
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }

# Example usage
reviews = [
    "absolutely fantastic movie, loved every minute",
    "terrible waste of time and money",
    "great acting but boring plot",
    "outstanding performance by the entire cast",
    "disappointing ending, wouldn't recommend"
]
sentiments = [1, 0, 0, 1, 0]  # 1=positive, 0=negative

# Create and train analyzer
analyzer = SentimentAnalyzer()
analyzer.train(reviews, sentiments)

# Evaluate performance
performance = analyzer.evaluate(reviews, sentiments)
print(f"Mean Accuracy: {performance['mean_accuracy']:.2%}")
print(f"Standard Deviation: {performance['std_accuracy']:.2%}")

# Analyze new reviews
new_reviews = [
    "incredible film, highly recommend",
    "waste of potential, poor direction"
]

results = analyzer.predict(new_reviews)
for review, (pred, probs) in zip(new_reviews, results):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = probs[1] if pred == 1 else probs[0]
    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2%}")
```

## Optimization Techniques 🔧

### 1. Feature Selection
```python
from sklearn.feature_selection import mutual_info_classif

def select_best_features(X, y, k=10):
    """Select top k features using mutual information"""
    mi_scores = mutual_info_classif(X, y)
    selected_features = np.argsort(mi_scores)[-k:]
    return selected_features, mi_scores[selected_features]
```

### 2. Handling Class Imbalance
```python
from sklearn.utils.class_weight import compute_class_weight

def get_balanced_model(X, y):
    """Create model with balanced class weights"""
    weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    return MultinomialNB(class_prior=weights)
```

### 3. Numeric Feature Discretization
```python
from sklearn.preprocessing import KBinsDiscretizer

def discretize_features(X, n_bins=5):
    """Discretize continuous features"""
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode='ordinal',
        strategy='quantile'
    )
    return discretizer.fit_transform(X)
```

## Best Practices and Common Pitfalls 💡

### Best Practices
1. Data Preprocessing
   - Handle missing values
   - Normalize text (for text classification)
   - Remove irrelevant features

2. Model Selection
   - Use GaussianNB for continuous features
   - Use MultinomialNB for count data
   - Use BernoulliNB for binary features

3. Parameter Tuning
   - Adjust smoothing parameter ($\alpha$)
   - Try different feature selection methods
   - Consider class weights for imbalanced data

### Common Pitfalls
1. Zero Probability Problem
   - Solution: Use Laplace smoothing
   - Adjust $\alpha$ parameter

2. Feature Independence Assumption
   - Solution: Feature selection
   - Remove highly correlated features

3. Numeric Feature Handling
   - Solution: Discretization
   - Use appropriate scaling

## Next Steps 📚

Now that you understand Naive Bayes, let's explore [k-Nearest Neighbors](./knn.md) to learn about instance-based learning!
