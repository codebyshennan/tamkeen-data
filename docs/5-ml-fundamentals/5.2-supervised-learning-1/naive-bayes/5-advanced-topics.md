---
reading_minutes: 25
objectives:
  - >-
    Engineer text and numeric features (n-grams, log transforms, binning) to
    lift NB accuracy on real data.
  - >-
    Handle missing values with sklearn imputers inside the pipeline rather than
    ad-hoc dropping.
  - >-
    Combine NB into a `VotingClassifier` or `StackingClassifier` to recover from
    the independence assumption.
  - >-
    Tune `alpha` (smoothing) with `GridSearchCV` and persist the fitted pipeline
    with `joblib` for deployment.
---

# Advanced Topics in Naive Bayes

**After this lesson:** you can explain Advanced Topics in Naive Bayes and try the examples in your own notebook.

## Overview

Discusses **smoothing**, correlated features breaking the assumption, and mitigations, without losing the fast baseline story.

## Welcome to Advanced Naive Bayes

Now that you've mastered the basics, we will look at some advanced techniques that will make your Naive Bayes models even better. Think of this as adding special tools to your machine learning toolbox!

## 1. Feature Engineering: Making Your Data Work Better

### What is Feature Engineering?

Feature engineering is like being a chef who transforms basic ingredients into a delicious meal. You take your raw data and transform it into features that help your model make better predictions.

### Text Feature Engineering

Suppose you're building a spam detector. Instead of just using raw words, you can create smarter features:

#### TF-IDF pipeline with custom preprocessing

Text Normalizer

`normalize_text` lowercases and strips non-letter characters (preserving punctuation like `!?.`) before tokenization, passed as the `preprocessor` callable to `TfidfVectorizer`.

TF-IDF and NB Pipeline

The pipeline chains vectorization (up to trigrams, top 1000 features) with `MultinomialNB`; calling `pipeline.fit` runs both steps in sequence automatically.

### Numerical Feature Engineering

When working with numbers (like age or income), you can transform them to better fit the Gaussian distribution:

#### Power transform + Gaussian NB

```python
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

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

#### KNN imputer + scaler + Gaussian NB (sketch)

Imputer Class

`SmartDataImputer` selects between KNN (neighbor-based) and iterative imputation; both strategies are sklearn's built-in imputers, the class is illustrative of the pattern.

Production Pipeline

The recommended approach is to drop the wrapper class and put `KNNImputer` directly into a `Pipeline` with `StandardScaler` and `GaussianNB` so all steps are cross-validated together.

\`\`\`

## 3. Ensemble Methods: Teamwork Makes the Dream Work

### What are Ensembles?

An ensemble is like a team of experts working together. Instead of relying on one model, we combine multiple models to get better predictions.

### Voting Classifier

#### VotingClassifier with multiple NB variants (illustrative)

\{% highlight python %\} from sklearn.ensemble import VotingClassifier from sklearn.naive\_bayes import GaussianNB, MultinomialNB, BernoulliNB

def create\_naive\_bayes\_team(): """Create a team of Naive Bayes models""" models = \[ ('multinomial', MultinomialNB()), # For text ('gaussian', GaussianNB()), # For numbers ('bernoulli', BernoulliNB()) # For yes/no features ]

```
return VotingClassifier(
    estimators=models,
    voting='soft'  # Use probability estimates
)
```

\{% endhighlight %\}

Three NB Variants

Each Naive Bayes variant targets a different feature type: Multinomial for text counts, Gaussian for continuous values, Bernoulli for binary yes/no features.

Soft Voting

`voting='soft'` averages predicted class probabilities from all estimators rather than majority-voting hard labels - note that in practice each base model needs compatible input features.

### Stacking Classifier

#### StackingClassifier with logistic meta-learner

\{% highlight python %\} from sklearn.ensemble import StackingClassifier from sklearn.linear\_model import LogisticRegression from sklearn.naive\_bayes import GaussianNB, MultinomialNB, BernoulliNB

def create\_stacked\_model(): """Create a stacked model with Naive Bayes""" base\_models = \[ ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()) ]

```
return StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5  # Use 5-fold cross-validation
)
```

\{% endhighlight %\}

Base Models

Three NB variants serve as base learners; stacking passes their out-of-fold predictions as features to the meta-learner rather than averaging them directly.

Logistic Meta-learner

`StackingClassifier` with `cv=5` generates cross-validated predictions from each base model; `LogisticRegression` learns how to combine them optimally.

## 4. Model Deployment: Taking Your Model to the Real World

### Saving Your Model

#### Persist estimator with joblib and sidecar JSON

\{% highlight python %\} import json

import joblib

class ModelSaver: def **init**(self, model, info=None): self.model = model self.info = info or {}

```
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

\{% endhighlight %\}

Class Init

Stores a reference to the fitted model and optional metadata dict (`info`) - the sidecar JSON lets you record version, training date, or feature names alongside the binary model.

Save and Load

`save` writes the estimator with `joblib.dump` and the metadata as JSON; the `@classmethod` `load` reverses both steps, reconstructing the `ModelSaver` instance.

### Monitoring Your Model

#### Track predictions for simple drift-style checks

\{% highlight python %\} from datetime import datetime

class ModelMonitor: def **init**(self): self.predictions = \[] self.timestamps = \[]

```
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

\{% endhighlight %\}

Track Predictions

Each prediction is stored as a dict with features, predicted label, optional true label, and timestamp - collecting these enables rolling accuracy checks without external logging infrastructure.

Rolling Accuracy

`check_performance` slices the last `window` predictions and counts matches against actuals - a quick drift indicator when true labels arrive with delay.

## 5. Hyperparameter Tuning: Finding the Best Settings

### What are Hyperparameters?

Hyperparameters are like the settings on your camera. You need to adjust them to get the best results for each situation.

### Finding the Best Settings

#### RandomizedSearchCV over vectorizer + MultinomialNB

\{% highlight python %\} from sklearn.model\_selection import RandomizedSearchCV from scipy.stats import uniform, randint from sklearn.feature\_extraction.text import TfidfVectorizer from sklearn.naive\_bayes import MultinomialNB from sklearn.pipeline import Pipeline

def find\_best\_settings(X, y): """Find the best hyperparameters""" # Define what settings to try param\_options = { 'vectorizer\_\_max\_features': randint(100, 10000), 'vectorizer\_\_ngram\_range': \[(1, 1), (1, 2), (1, 3)], 'classifier\_\_alpha': uniform(0.1, 2.0) }

```
# Create the model
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Search for best settings
search = RandomizedSearchCV(
    model, param_options,
    n_iter=20,  # Try 20 different combinations
    cv=3,       # Use 3-fold CV (needs enough samples vs n_splits)
    scoring='accuracy'
)

# Find the best settings
search.fit(X, y)
return search.best_params_
```

## Example: small text corpus (enough rows for cv=3)

X\_text = \[ "sports team wins game", "stock market news today", "team scores in final quarter", "finance report earnings beat", "championship final overtime", "investors buy tech shares", "roster injury update", "quarterly revenue growth", "playoff bracket announced", "dividend yield increases", ] y\_text = \[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] best = find\_best\_settings(X\_text, y\_text) \{% endhighlight %\}

Search Space and Pipeline

The param dict samples `max_features` and `alpha` from continuous distributions (`randint`/`uniform`) and tries three ngram ranges; the `vectorizer__` prefix routes params to the correct pipeline step.

Randomized Search

`RandomizedSearchCV` with `n_iter=20` tries 20 random combinations instead of an exhaustive grid - much faster when the search space is large.

Toy Corpus Example

Ten labeled sentences (sports vs finance) illustrate the input format; call `find_best_settings` on your real dataset to get the winning hyperparameter combination via `best_params_`.

## Common Advanced Challenges and Solutions

### 1. Dealing with Class Imbalance

When one class is much more common than others:

#### Normalize balanced weights to `class_prior`

```python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB

y = np.array([0] * 90 + [1] * 10)
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
priors = class_weights / class_weights.sum()
model = MultinomialNB(class_prior=priors)
```

### 2. Handling High-Dimensional Data

When you have too many features:

#### Chi-squared feature selection before NB

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import csr_matrix

rng = np.random.default_rng(0)
X = csr_matrix(rng.integers(1, 10, size=(50, 200)))
y = rng.integers(0, 2, size=50)

selector = SelectKBest(chi2, k=100)  # keep top features for this toy size
X_new = selector.fit_transform(X, y)
```

### 3. Improving Numeric Stability

When dealing with very small probabilities:

#### Argmax on `predict_log_proba`

```python
import numpy as np
from sklearn.naive_bayes import BernoulliNB

X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
y = np.array([0, 1, 0])
model = BernoulliNB().fit(X, y)

log_probs = model.predict_log_proba(X)
predictions = np.argmax(log_probs, axis=1)
```

## Gotchas

* **Using `VotingClassifier` with NB variants that expect different feature types**: The `create_naive_bayes_team` example lists `MultinomialNB`, `GaussianNB`, and `BernoulliNB` in one `VotingClassifier`. This only works if all three receive compatible input. In practice, each variant requires a different feature representation (counts, continuous values, binary), so a naive ensemble on a single feature matrix will raise errors or produce wrong results for at least two of the three.
* **Applying `PowerTransformer` and then expecting `GaussianNB` to be perfectly calibrated**: `PowerTransformer(method='yeo-johnson')` makes features more Gaussian-like but does not guarantee true normality. `GaussianNB` still makes a Gaussian assumption per class; if the transformed distribution is still skewed or bimodal, the probability estimates will be poorly calibrated even after transformation.
* **Using `IterativeImputer` without `enable_iterative_imputer`**: The `IterativeImputer` is experimental in scikit-learn and requires the `enable_iterative_imputer` import guard (`from sklearn.experimental import enable_iterative_imputer`). Omitting it raises an `ImportError` that looks like the class doesn't exist, which confuses learners who see it referenced in the docs.
* **Tuning `alpha` near zero in `RandomizedSearchCV`**: If `uniform(0.1, 2.0)` is replaced with `uniform(0, 2.0)`, the search may sample `alpha` values very close to zero. Near-zero smoothing effectively removes Laplace correction, causing zero-probability terms for unseen tokens and potential numerical instability. Keep `alpha` strictly positive (typically ≥ 0.01).
* **Saving a pipeline with `joblib` but loading it in a different scikit-learn version**: `joblib.dump` serializes the fitted estimator including internal attributes. If the scikit-learn version changes between save and load, internal attribute names may differ and `joblib.load` will either raise an error or silently return an object with broken state. Pin your dependency versions in deployment.
* **Checking rolling accuracy in `ModelMonitor` when `actual` is `None`**: `check_performance` compares `p['prediction'] == p['actual']` for all recent records. If some records were logged without a true label (`actual=None`), those comparisons evaluate to `False` and drag down the accuracy score without any warning. Filter out records where `actual is None` before computing the window accuracy.

## Next Steps

Ready to become a Naive Bayes expert? Try these challenges:

1. Implement feature engineering in your own project
2. Experiment with different ensemble methods
3. Deploy a model and monitor its performance
4. Try hyperparameter tuning on a real dataset

Remember: The best way to learn is by doing! Start with small experiments and gradually tackle more complex problems.
