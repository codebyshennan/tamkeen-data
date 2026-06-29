---
reading_minutes: 22
objectives:
  - "Compose preprocessing + model into a `Pipeline` (or `ColumnTransformer` for mixed-type tables) so `fit` / `predict` run as a single estimator."
  - "Eliminate leakage: every transformer in the pipeline is fit on training folds only when it sits inside `cross_val_score` / `GridSearchCV`."
  - "Persist the whole pipeline with `joblib.dump` so the prediction-time preprocessing exactly matches training."
  - "Use named steps and `set_params(step__hyper=...)` so a single search tunes transformer and model hyperparameters together."
---

# Scikit-learn Pipelines

**After this lesson:** you can explain the core ideas in “Scikit-learn Pipelines” and reproduce the examples here in your own notebook or environment.

## Overview

**Pipelines** bundle preprocessing + model to prevent leakage and serialize a reproducible path to production.


## Understanding Pipelines

Pipelines help us:

1. Ensure preprocessing steps are consistent
2. Prevent data leakage
3. Simplify model deployment
4. Make code more maintainable

{% include model-eval-html-diagram.html diagram="sklearn-pipelines" title="Scikit-learn pipeline workflow diagram" %}

*Without a pipeline, if you `StandardScaler.fit(X_all)` before splitting, test-set statistics leak into the scaler — the pipeline prevents this by fitting each step only on training data.*

#### Minimal `Pipeline`: scale then classify

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>Core sklearn components: <code>Pipeline</code> for chaining, <code>StandardScaler</code> for preprocessing, and <code>LogisticRegression</code> as the model.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Synthetic Dataset</span>
    </div>
    <div class="code-callout__body">
      <p>Generate 1000 samples with age, income, and credit score features; the binary label is derived from a linear threshold on those three features.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-34" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline Definition and Fit</span>
    </div>
    <div class="code-callout__body">
      <p>A two-step pipeline: the scaler's <code>fit</code> is called only on <code>X_train</code> inside <code>pipeline.fit</code>, preventing test statistics from leaking into preprocessing.</p>
    </div>
  </div>
</aside>
</div>

```
Pipeline score: 0.990
```

## Building Complex Pipelines

### Feature Unions

Combine multiple feature processing steps:

#### Parallel feature branches with `FeatureUnion`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p><code>FeatureUnion</code> runs multiple transformers in parallel and concatenates outputs; <code>PCA</code> reduces dimensions, <code>SelectKBest</code> picks the top univariate features.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Parallel Feature Branches</span>
    </div>
    <div class="code-callout__body">
      <p>The <code>FeatureUnion</code> applies PCA and SelectKBest on the same input concurrently, then stacks the resulting columns as a wider feature matrix for the classifier.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-22" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Build and Score</span>
    </div>
    <div class="code-callout__body">
      <p>Instantiate, fit on training data, and evaluate on the held-out test set in three lines — same API as a simple pipeline.</p>
    </div>
  </div>
</aside>
</div>

```
Feature union score: 0.980
```

### Custom Transformers

Create your own preprocessing steps:

#### Custom `TransformerMixin` for outlier clipping

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
        _, col_idx = np.where(mask)
        X_copy[mask] = self.mean_[col_idx]
        return X_copy

# Use custom transformer in pipeline
pipeline_with_custom = Pipeline([
    ('outlier_handler', OutlierHandler(threshold=3)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline_with_custom.fit(X_train, y_train)
print(f"Custom pipeline score: {pipeline_with_custom.score(X_test, y_test):.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Custom Transformer Class</span>
    </div>
    <div class="code-callout__body">
      <p>Inheriting <code>BaseEstimator</code> and <code>TransformerMixin</code> gives <code>get_params</code>, <code>set_params</code>, and <code>fit_transform</code> for free; <code>fit</code> stores per-column mean and std from training data.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outlier Clipping Logic</span>
    </div>
    <div class="code-callout__body">
      <p>Compute z-scores using stored train statistics; values exceeding the threshold are replaced with column means, leaving normal values unchanged.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-29" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline Integration</span>
    </div>
    <div class="code-callout__body">
      <p>The custom step slots in as the first pipeline step; scaling and classification follow in the same <code>fit</code>/<code>predict</code> call.</p>
    </div>
  </div>
</aside>
</div>

```
Custom pipeline score: 0.970
```

## Real-World Example: Text Classification

#### Text preprocessing + TF–IDF + logistic regression in one `Pipeline`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import re

# Sample text data
texts = [
    "Machine learning is fascinating",
    "Deep neural networks are powerful",
    "Data science is growing rapidly",
    "AI transforms industries",
    "The model missed too many fraud cases",
    "This pipeline leaks test data",
    "The baseline accuracy is misleading",
    "Production monitoring caught model drift"
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]

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
print(f"Predictions: {predictions.tolist()}")
print(f"Test labels: {y_test}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Text Data and Imports</span>
    </div>
    <div class="code-callout__body">
      <p>Eight sentences with binary labels; <code>TfidfVectorizer</code> and <code>FunctionTransformer</code> will handle text → numeric conversion inside the pipeline.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Text Preprocessor</span>
    </div>
    <div class="code-callout__body">
      <p>Lowercase and strip non-alpha characters before vectorization — wrapped in a lambda so it processes the full list at once inside the pipeline step.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-27" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Text Pipeline</span>
    </div>
    <div class="code-callout__body">
      <p>Three sequential steps: clean text → TF-IDF sparse matrix → logistic regression, all within one <code>Pipeline</code> so the vectorizer is only fit on training data.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="29-37" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Split and Predict</span>
    </div>
    <div class="code-callout__body">
      <p>Split raw string lists directly — sklearn handles list inputs — then fit and predict exactly as with numeric arrays.</p>
    </div>
  </div>
</aside>
</div>

```
Predictions: [1, 1]
Test labels: [1, 0]
```

## Pipeline Persistence

Save and load pipelines:

#### Serialize a fitted pipeline with `joblib`

**Purpose:** Show the persistence helper pattern; this is no-output because it writes and reads model files.

```python
# no-output
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

## Advanced Techniques

### 1. Memory Caching

#### Cache intermediate pipeline outputs on disk

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Set up caching
memory = Memory(location='./cachedir', verbose=0)

# Create pipeline with caching
cached_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
], memory=memory)

# Fit once to populate the cache; identical inputs reuse cached transforms
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=300, n_features=6, random_state=42)
cached_pipeline.fit(X, y)
print("Pipeline steps:", list(cached_pipeline.named_steps))
print(f"Training accuracy: {cached_pipeline.score(X, y):.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports and Memory</span>
    </div>
    <div class="code-callout__body">
      <p><code>joblib.Memory</code> writes transformer outputs to <code>./cachedir</code>; on subsequent fits with identical input, the cached result is reused instead of recomputing.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Cached Pipeline</span>
    </div>
    <div class="code-callout__body">
      <p>Pass <code>memory=memory</code> to <code>Pipeline</code>; this tells sklearn to serialize each fitted transformer step so repeated grid-search folds skip redundant transforms.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-22" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit and Reuse</span>
    </div>
    <div class="code-callout__body">
      <p>The first fit computes every step and writes its output to the cache; later fits with identical input reuse it instead of recomputing.</p>
    </div>
  </div>
</aside>
</div>

```
Pipeline steps: ['scaler', 'pca', 'classifier']
Training accuracy: 0.930
```

### 2. Parameter Grid Search

#### Tune nested steps with `__` hyperparameter names

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=42)
grid_X_train, grid_X_test, grid_y_train, grid_y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', LogisticRegression(max_iter=1000)),
])

# Define parameters for multiple steps
param_grid = {
    'scaler__with_mean': [True, False],
    'pca__n_components': [2, 3, 4],
    'classifier__C': [0.1, 1.0, 10.0],
}

# Perform grid search
grid_search = GridSearchCV(grid_pipeline, param_grid, cv=5)
grid_search.fit(grid_X_train, grid_y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(grid_X_test, grid_y_test):.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline Definition</span>
    </div>
    <div class="code-callout__body">
      <p>A three-step pipeline — scaler, PCA, and logistic regression — where step names will be used as keys in the parameter grid using the <code>step__param</code> convention.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-18" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Nested Parameter Grid</span>
    </div>
    <div class="code-callout__body">
      <p>Keys like <code>pca__n_components</code> reach inside the named step; <code>GridSearchCV</code> handles all combinations across both preprocessing and model hyperparameters.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-22" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Cross-validated Search</span>
    </div>
    <div class="code-callout__body">
      <p>5-fold CV ensures preprocessing is re-fit each fold, so the selected hyperparameters reflect honest out-of-fold performance.</p>
    </div>
  </div>
</aside>
</div>

```
Best parameters: {'classifier__C': 0.1, 'pca__n_components': 4, 'scaler__with_mean': True}
Best CV score: 0.685
Test score: 0.710
```

### 3. Column Transformer

#### Different transformers per column group

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

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

# Fit on a small mixed-type dataset: 2 numeric columns + 1 categorical column
import numpy as np
X = np.array([[1.2, 3.4, 0], [2.1, 1.0, 1], [0.5, 2.2, 2],
              [1.8, 0.7, 1], [2.5, 3.1, 0], [0.3, 1.5, 2]])
y = np.array([0, 1, 0, 1, 0, 1])
column_pipeline.fit(X, y)
print("Transformed feature count:", preprocessor.fit_transform(X).shape[1])
print(f"Training accuracy: {column_pipeline.score(X, y):.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p><code>ColumnTransformer</code> applies different preprocessing to different column subsets and concatenates the results into a single matrix.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Column-specific Transforms</span>
    </div>
    <div class="code-callout__body">
      <p>Columns <code>[0, 1]</code> get standard-scaled; column <code>[2]</code> gets one-hot encoded — both in parallel, then stacked as the output feature matrix.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-17" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Final Pipeline</span>
    </div>
    <div class="code-callout__body">
      <p>The preprocessor becomes a single named step; the classifier operates on the fully-transformed matrix from both numeric and categorical branches.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-26" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit and Check</span>
    </div>
    <div class="code-callout__body">
      <p>Fitting confirms the wiring: the 3 input columns expand to 5 features (2 scaled numeric + 3 one-hot categories) before the classifier sees them.</p>
    </div>
  </div>
</aside>
</div>

```
Transformed feature count: 5
Training accuracy: 1.000
```

## Best Practices

### 1. Naming Conventions

#### Readable step names for grids and debugging

**Purpose:** Demonstrate descriptive step names that make pipeline grids and error messages easier to read.

```python
# no-output
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Use descriptive names for steps
pipeline = Pipeline([
    ('missing_handler', SimpleImputer()),
    ('feature_scaler', StandardScaler()),
    ('dim_reducer', PCA()),
    ('classifier', LogisticRegression())
])
```

### 2. Error Handling

#### Illustrative try/except inside `transform`

**Purpose:** Sketch a defensive transformer pattern; the placeholder `transformed_X` marks it as illustrative rather than runnable.

```python
# no-output
from sklearn.base import BaseEstimator, TransformerMixin

class RobustTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        try:
            # Transformation logic
            return transformed_X  # replace with real output; illustrative only
        except Exception as e:
            print(f"Error in transformation: {e}")
            # Return safe fallback
            return X
```

### 3. Validation

#### CV helper for any pipeline object

**Purpose:** Wrap cross-validation reporting in one helper that can be reused with any sklearn-compatible pipeline.

```python
# no-output
from sklearn.model_selection import cross_val_score

def validate_pipeline(pipeline, X, y, cv=5):
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv)
    
    # Print results
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Common Pitfalls and Solutions

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

## Gotchas

- **Calling `fit_transform` on the full dataset before the pipeline** — If you run `scaler.fit_transform(X)` on all data and then pass it into a `Pipeline` for cross-validation, the scaler has already seen test folds and leaked their statistics; the pipeline only prevents leakage if it receives raw, untransformed data.
- **Accessing step attributes before fitting** — Calling `pipeline.named_steps['scaler'].mean_` before `pipeline.fit(X_train, y_train)` raises `NotFittedError`; the pipeline steps are cloned and fitted lazily, so attributes like `mean_`, `coef_`, or `feature_importances_` are only available after fitting.
- **Forgetting the double-underscore syntax for nested parameters** — In a grid search over a pipeline, hyperparameter names must be `stepname__param` (e.g., `classifier__max_depth`); a single underscore or a misspelled step name causes a silent `KeyError` or `ValueError` that can look like a data problem.
- **`FeatureUnion` and `ColumnTransformer` do not preserve column names by default** — After transformation, the output is a plain numpy array; any subsequent step that expects a DataFrame with column names (e.g., a custom transformer checking `X.columns`) will break unless you explicitly handle name reconstruction or use `set_output(transform='pandas')` (sklearn ≥ 1.2).
- **Pickling a pipeline that uses a lambda function** — `FunctionTransformer(lambda x: ...)` cannot be pickled with the default Python `pickle` or `joblib`; when deploying or saving the pipeline, replace inline lambdas with named module-level functions to avoid `AttributeError: Can't pickle local object`.
- **Memory caching invalidates when estimator parameters change** — `Pipeline(memory=memory)` caches transformer outputs keyed by the step's parameters; if you change `PCA(n_components=2)` to `PCA(n_components=3)`, the cache is invalidated and all folds recompute; the cache only accelerates runs where parameters are truly unchanged across calls.

## Additional Resources

- [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/pipeline.html)
- [Pipeline Best Practices](https://scikit-learn.org/stable/modules/compose.html)
- [Custom Transformers Guide](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html)

Remember: Pipelines are your best friend for reproducible machine learning workflows!
