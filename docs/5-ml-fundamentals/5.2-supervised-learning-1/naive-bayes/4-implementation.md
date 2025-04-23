# Implementing Naive Bayes in Practice

## Welcome to Hands-On Naive Bayes! 🎯

Now that you understand the theory, let's roll up our sleeves and implement Naive Bayes in real projects. We'll start with simple examples and gradually build up to more complex applications.

## Setting Up Your Environment

First, let's make sure you have everything you need:

```python
# Install required packages
!pip install scikit-learn pandas numpy matplotlib

# Import essential libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

## Project 1: Spam Email Classifier 📧

### Understanding the Problem

Imagine you're building a spam filter for your email. You want to automatically identify which emails are spam and which are legitimate. This is a perfect job for Naive Bayes!

### Step 1: Prepare Your Data

```python
# Sample dataset - in real life, you'd have many more emails!
emails = [
    "Get rich quick! Buy now!",
    "Meeting at 3pm tomorrow",
    "Win a free iPhone today!",
    "Project deadline reminder",
    "Free money, act now!",
    "Team lunch next week"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.2, random_state=42
)
```

### Step 2: Create a Text Processing Pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def create_spam_classifier():
    """Create a pipeline for spam detection"""
    return Pipeline([
        # Convert text to numbers
        ('vectorizer', TfidfVectorizer(
            lowercase=True,      # Convert to lowercase
            stop_words='english', # Remove common words
            ngram_range=(1, 2),  # Look at single words and pairs
            min_df=2            # Ignore very rare words
        )),
        # Use Multinomial NB for text classification
        ('classifier', MultinomialNB(
            alpha=1.0  # Smoothing parameter
        ))
    ])
```

### Step 3: Train and Evaluate the Model

```python
# Create and train the model
model = create_spam_classifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

### Step 4: Use Your Model

```python
# Test with new emails
new_emails = [
    "Congratulations! You've won a prize!",
    "Team meeting scheduled for Friday"
]

# Make predictions
predictions = model.predict(new_emails)
probabilities = model.predict_proba(new_emails)

# Print results
for email, pred, prob in zip(new_emails, predictions, probabilities):
    print(f"\nEmail: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Not Spam'}")
    print(f"Confidence: {max(prob):.2%}")
```

## Project 2: Medical Diagnosis System 🏥

### Understanding the Problem

Let's build a system that helps doctors predict whether a patient has a certain disease based on their symptoms and test results.

### Step 1: Prepare Your Data

```python
# Sample patient data
# Features: [temperature, heart_rate, blood_pressure, age]
patient_data = [
    [38.5, 90, 140, 45],
    [37.0, 70, 120, 30],
    [39.0, 95, 150, 55],
    [36.8, 75, 125, 35]
]
# Labels: 1 for sick, 0 for healthy
conditions = [1, 0, 1, 0]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    patient_data, conditions, test_size=0.2, random_state=42
)
```

### Step 2: Create a Medical Diagnosis Pipeline

```python
from sklearn.preprocessing import StandardScaler

def create_medical_classifier():
    """Create a pipeline for medical diagnosis"""
    return Pipeline([
        # Scale the features (important for Gaussian NB)
        ('scaler', StandardScaler()),
        # Use Gaussian NB for numerical data
        ('classifier', GaussianNB())
    ])
```

### Step 3: Train and Evaluate the Model

```python
# Create and train the model
model = create_medical_classifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

### Step 4: Use Your Model

```python
# New patient data
new_patient = [[38.2, 85, 135, 40]]

# Make prediction
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

# Print results
print(f"Diagnosis: {'Sick' if prediction[0] == 1 else 'Healthy'}")
print(f"Confidence: {max(probability[0]):.2%}")
```

## Project 3: Product Categorization System 🛍️

### Understanding the Problem

Let's build a system that automatically categorizes products based on their descriptions. This is useful for e-commerce websites.

### Step 1: Prepare Your Data

```python
# Sample product data
products = [
    "blue cotton t-shirt size M",
    "leather wallet black",
    "running shoes size 10",
    "denim jeans blue",
    "sports water bottle"
]
categories = ['Clothing', 'Accessories', 'Shoes', 'Clothing', 'Sports']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    products, categories, test_size=0.2, random_state=42
)
```

### Step 2: Create a Product Categorization Pipeline

```python
from sklearn.preprocessing import LabelEncoder

def create_product_classifier():
    """Create a pipeline for product categorization"""
    return Pipeline([
        # Convert text to word counts
        ('vectorizer', CountVectorizer(
            ngram_range=(1, 2),  # Look at words and pairs
            stop_words='english'  # Remove common words
        )),
        # Use Multinomial NB for text classification
        ('classifier', MultinomialNB())
    ])
```

### Step 3: Train and Evaluate the Model

```python
# Create and train the model
model = create_product_classifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

### Step 4: Use Your Model

```python
# New product
new_product = ["white cotton socks pack"]

# Make prediction
prediction = model.predict(new_product)

# Print result
print(f"Predicted Category: {prediction[0]}")
```

## Best Practices and Tips 💡

### 1. Data Preprocessing

Always preprocess your data properly:

- For text: clean, normalize, and vectorize
- For numbers: scale and handle outliers
- For categories: encode properly

### 2. Model Evaluation

Use multiple metrics to evaluate your model:

- Accuracy: Overall correctness
- Precision: How many predicted positives are actually positive
- Recall: How many actual positives are correctly predicted
- F1-score: Balance between precision and recall

### 3. Common Pitfalls to Avoid

1. **Forgetting to Scale Numerical Features**

   ```python
   # Always do this for Gaussian NB
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Ignoring Class Imbalance**

   ```python
   # Handle imbalanced classes
   from sklearn.utils.class_weight import compute_class_weight
   class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
   model = MultinomialNB(class_prior=class_weights)
   ```

3. **Not Using Cross-Validation**

   ```python
   # Always use cross-validation
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X, y, cv=5)
   print(f"Mean accuracy: {scores.mean():.2f}")
   ```

## Next Steps 📚

Ready to take your Naive Bayes skills to the next level? Check out the [Advanced Topics](5-advanced-topics.md) section to learn about:

- Feature engineering techniques
- Handling missing data
- Ensemble methods
- Model deployment

Remember: Practice makes perfect! Try implementing these examples and then modify them for your own projects.
