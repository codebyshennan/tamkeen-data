# 5.2 Supervised Learning Part 1

**After this submodule:** you can use the lessons linked below and complete the exercises that match **5.2 Supervised Learning Part 1** in your course schedule.

## Overview

This submodule introduces four classical **supervised** algorithms: [Naive Bayes](naive-bayes/1-introduction.md), [k-nearest neighbors](knn/1-introduction.md), [support vector machines](svm/1-introduction.md), and [decision trees](decision-trees/1-introduction.md). You will get enough intuition to choose a first model and know where to read next. **Prerequisites:** [5.1 Introduction to ML](../5.1-intro-to-ml/README.md) (workflow, features, bias-variance); comfort with pandas and train/test splits.

## Why this matters

These methods still appear in production, teaching, and interviews. Understanding their assumptions (e.g., independence for Naive Bayes, distance geometry for kNN, margins and kernels for SVM, axis-aligned splits for trees) helps you debug failures and pair algorithms with the right metrics.

Welcome to the first part of supervised learning! Here we'll explore fundamental algorithms that form the backbone of machine learning. Think of these algorithms as different tools in your ML toolkit - each with its own strengths and ideal use cases.

## Helpful video

Crash Course AI: supervised learning for classical algorithms.

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qVRBYAdLAo" title="Supervised Learning: Crash Course AI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Learning Objectives

By the end of this section, you will be able to:

1. Understand and implement Naive Bayes classifiers for text classification
2. Master k-Nearest Neighbors (kNN) for both classification and regression tasks
3. Apply Support Vector Machines (SVM) for complex decision boundaries
4. Build and interpret Decision Trees for transparent decision-making
5. Choose the optimal algorithm for different problem types

## Algorithm Overview

### 1. [Naive Bayes](naive-bayes/1-introduction.md)

Probabilistic classifier based on Bayes' Theorem:

$$P(y|X) = \frac{P(X|y)P(y)}{P(X)}$$

Perfect for:

* Text classification (spam detection, sentiment analysis)
* High-dimensional data
* Real-time prediction needs
* When independence assumption holds

### 2. [k-Nearest Neighbors](knn/1-introduction.md)

Instance-based learning using distance metrics:

$$\text{distance}(p,q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$

Ideal for:

* Recommendation systems
* Pattern recognition
* Anomaly detection
* When data is well-clustered

### 3. [Support Vector Machines](svm/1-introduction.md)

Finds optimal hyperplane with maximum margin:

$$\min_{w,b} \frac{1}{2}||w||^2 \text{ subject to } y_i(w^Tx_i + b) \geq 1$$

Best for:

* Complex classification tasks
* Non-linear decision boundaries
* High-dimensional spaces
* When clear margin of separation exists

### 4. [Decision Trees](decision-trees/1-introduction.md)

Hierarchical decisions using information theory:

$$\text{Information Gain} = H(\text{parent}) - \sum_{j=1}^m \frac{N_j}{N} H(\text{child}_j)$$

Excellent for:

* Interpretable models
* Mixed data types
* Feature importance analysis
* When non-linear relationships exist

## Algorithm Selection Guide

Use the sketch below as a **first guess**, then validate with cross-validation and baselines (logistic regression, linear SVM) as in [5.5 Model evaluation](../5.5-model-eval/).

### Classification Tasks

#### Heuristic mapping from data traits to a first algorithm

```python
def select_classifier(data_characteristics):
    if data_characteristics.text_data:
        return "Naive Bayes"
    elif data_characteristics.need_interpretability:
        return "Decision Tree"
    elif data_characteristics.high_dimensional:
        return "SVM"
    elif data_characteristics.well_clustered:
        return "kNN"
    else:
        return "Try multiple and compare"
```

### Performance Comparison

| Algorithm      | Training speed | Prediction speed | Interpretability | Memory usage |
| -------------- | -------------- | ---------------- | ---------------- | ------------ |
| Naive Bayes    | Usually fast   | Fast             | Moderate (coefficients / log-probs) | Low |
| kNN            | Very fast (often just store data) | Slower as $n$ grows | Low (black-box votes) | High (stores training set) |
| SVM            | Can be costly on large $n$ | Moderate | Low-moderate | Moderate |
| Decision Trees | Fast           | Fast             | High (rules)     | Low          |

Values are typical heuristics; always profile and validate on your dataset.

## Prerequisites

Before diving in, ensure you're comfortable with:

### 1. Mathematics

* Basic probability theory
* Linear algebra fundamentals
* Information theory concepts
* Distance metrics

### 2. Programming

#### Core imports for this module

```python
# Essential Python libraries
import numpy as np          # Numerical operations
import pandas as pd         # Data manipulation
import sklearn             # Machine learning tools
import matplotlib.pyplot as plt  # Visualization
```

### 3. Concepts

* Feature engineering
* Model evaluation metrics
* Cross-validation
* Bias-variance tradeoff

## Real-World Applications

### 1. Email Classification

#### Text pipeline sketch: TF-IDF + multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Example: Spam Detection
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()
```



### 2. Medical Diagnosis

#### Nonlinear SVM with RBF kernel (illustrative)

```python
from sklearn.svm import SVC

# Example: Disease Classification
svm_classifier = SVC(kernel='rbf', C=1.0)
```



### 3. Credit Risk Assessment

#### Shallow tree for interpretable credit decisions

```python
from sklearn.tree import DecisionTreeClassifier

# Example: Loan Approval
dt_classifier = DecisionTreeClassifier(max_depth=5)
```



### 4. Recommendation Systems

#### kNN as a lazy learner for similarity-style classification

```python
from sklearn.neighbors import KNeighborsClassifier

# Example: Product Recommendations
knn_classifier = KNeighborsClassifier(n_neighbors=5)
```



## Learning Path

Suggested order follows the nav: start simple and build geometry and splitting ideas.

1. Start with [Naive Bayes](naive-bayes/1-introduction.md)
   * Understand probability basics
   * Learn text classification
   * Master feature independence
2. Move to [k-Nearest Neighbors](knn/1-introduction.md)
   * Grasp distance metrics
   * Understand k selection
   * Handle the curse of dimensionality
3. Progress to [Support Vector Machines](svm/1-introduction.md)
   * Master linear classification
   * Explore kernel methods
   * Optimize hyperparameters
4. Conclude with [Decision Trees](decision-trees/1-introduction.md)
   * Learn tree construction
   * Understand splitting criteria
   * Practice pruning techniques

## Tools and Environment

### Required Libraries

#### Install scientific Python stack (example)

```bash
# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Recommended IDE Setup

#### Notebook starter imports and reproducibility

```python
# Standard imports for all notebooks
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
```

## Best Practices

1. Data Preparation
   * Handle missing values
   * Scale features appropriately
   * Split data properly
2. Model Selection
   * Consider problem characteristics
   * Start simple, increase complexity
   * Use cross-validation
3. Evaluation
   * Choose appropriate metrics
   * Test on holdout set
   * Consider computational costs

## Common Pitfalls

1. Naive Bayes
   * Zero frequency problem
   * Feature independence assumption
   * Numeric precision issues
2. kNN
   * Curse of dimensionality
   * Scale sensitivity
   * Memory requirements
3. SVM
   * Kernel selection
   * Parameter tuning
   * Scaling requirements
4. Decision Trees
   * Overfitting
   * Feature interaction handling
   * Categorical variable splits

## Assignment

Ready to apply your supervised learning knowledge? The questions are in [Module 5 assignment](../assignments/module-assignment.md) (see Section 2 for **5.2**); self-check answers are in [assignments.md](../assignments.md).

The companion notebook demonstrates all four algorithms, including SVM. The coding assignment focuses on Naive Bayes, kNN, and Decision Trees so the assessed work stays short enough for one practice block.

## Ready to Begin?

Start your journey with [Naive Bayes](naive-bayes/1-introduction.md) to build a strong foundation in probabilistic classification. Each algorithm builds upon previous concepts, so following the suggested order will maximize your learning experience.

The best way to learn this is by doing. Each section includes hands-on examples and exercises to reinforce your understanding.
