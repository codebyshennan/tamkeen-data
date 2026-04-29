---
reading_minutes: 30
objectives:
  - "Use `ExtraTreesClassifier` / `ExtraTreesRegressor` and recognise when extra randomization helps over standard random forests."
  - "Handle class imbalance with `class_weight`, `sample_weight`, and resampling strategies (e.g., SMOTE) inside a forest pipeline."
  - "Plug a random forest into a serving pipeline — estimate model size, run batch inference, and persist with pickle/joblib."
---

# Advanced Random Forest Techniques

**After this lesson:** you can explain the core ideas in “Advanced Random Forest Techniques” and reproduce the examples here in your own notebook or environment.

## Overview

Extra trees, class imbalance handling, and tuning strategies beyond defaults.

## Helpful video

Crash Course AI: supervised learning framing (~15 min).

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qVRBYAdLAo" title="Supervised Learning: Crash Course AI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Ensemble Optimization

{% include mermaid-diagram.html src="5-ml-fundamentals/5.3-supervised-learning-2/random-forest/diagrams/4-advanced-1.mmd" %}

### 1. Stacking with Random Forests

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base estimators
estimators = [
    ('rf1', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ('rf2', RandomForestClassifier(n_estimators=100, max_features='sqrt')),
    ('rf3', RandomForestClassifier(n_estimators=100, min_samples_leaf=5))
]

# Create stacked model
stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# Train stacked model
stacked_model.fit(X_train, y_train)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Three Forest Variants</span>
    </div>
    <div class="code-callout__body">
      <p>Three <code>RandomForestClassifier</code> instances differ in depth, feature sampling strategy, and leaf size — diversity in the base models is key; similar models won't add information when stacked.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Stack and Fit</span>
    </div>
    <div class="code-callout__body">
      <p><code>StackingClassifier</code> generates out-of-fold predictions from each base model with <code>cv=5</code>, then trains <code>LogisticRegression</code> to combine them — a learned ensemble that outperforms majority voting.</p>
    </div>
  </div>
</aside>
</div>

### 2. Weighted Voting

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def weighted_voting_predict(models, weights, X):
    """Implement weighted voting for ensemble"""
    predictions = np.array([
        model.predict_proba(X) for model in models
    ])

    # Weight each model's predictions
    weighted_pred = np.average(
        predictions,
        weights=weights,
        axis=0
    )

    return np.argmax(weighted_pred, axis=1)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Collect Probabilities</span>
    </div>
    <div class="code-callout__body">
      <p>List-comprehension calls <code>predict_proba</code> on each model and stacks results into a 3D array (models × samples × classes) for vectorized aggregation.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Weighted Average</span>
    </div>
    <div class="code-callout__body">
      <p><code>np.average(..., weights=weights, axis=0)</code> combines probability matrices; <code>argmax</code> then picks the class with the highest combined probability as the final prediction.</p>
    </div>
  </div>
</aside>
</div>

## Advanced Feature Engineering

### 1. Automated Feature Interactions

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from itertools import combinations

def create_feature_interactions(X, degree=2):
    """Create all possible feature interactions up to specified degree"""
    X = X.copy()
    feature_names = list(X.columns)

    for d in range(2, degree + 1):
        for combo in combinations(feature_names, d):
            # Create interaction feature
            name = '*'.join(combo)
            X[name] = 1
            for feature in combo:
                X[name] *= X[feature]

    return X
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Copy the dataframe to avoid in-place mutation; collect column names for combinatorial generation up to the specified degree.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Product Columns</span>
    </div>
    <div class="code-callout__body">
      <p>For each feature combination, initialize a new column to 1 then multiply by each component feature; the starred join creates readable column names like <code>age*income</code> for downstream interpretation.</p>
    </div>
  </div>
</aside>
</div>

### 2. Feature Selection with Permutation Importance

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.inspection import permutation_importance

def analyze_permutation_importance(model, X, y):
    """Calculate permutation importance with cross-validation"""
    result = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    return importance_df
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Permutation Importance</span>
    </div>
    <div class="code-callout__body">
      <p><code>permutation_importance</code> shuffles each feature 10 times and measures the drop in model score — features that cause a large drop are important; those that don't can be dropped.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Importance DataFrame</span>
    </div>
    <div class="code-callout__body">
      <p>Package mean and std importance into a sorted DataFrame — the std across 10 repeats shows how stable each feature's importance is, helping distinguish truly important features from noisy ones.</p>
    </div>
  </div>
</aside>
</div>

## Optimization Techniques

### 1. Dynamic Feature Selection

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DynamicFeatureSelector:
    """Dynamically select features based on importance threshold"""
    def __init__(self, base_model, threshold=0.01):
        self.base_model = base_model
        self.threshold = threshold
        self.selected_features = None

    def fit(self, X, y):
        # Train base model
        self.base_model.fit(X, y)

        # Get feature importance
        importances = self.base_model.feature_importances_

        # Select features above threshold
        self.selected_features = X.columns[importances > self.threshold]

        # Retrain on selected features
        self.base_model.fit(X[self.selected_features], y)

        return self

    def predict(self, X):
        return self.base_model.predict(X[self.selected_features])
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Setup</span>
    </div>
    <div class="code-callout__body">
      <p>The selector wraps any <code>base_model</code> that exposes <code>feature_importances_</code>; the <code>threshold</code> controls how aggressively low-importance features are dropped.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Two-Pass Fit</span>
    </div>
    <div class="code-callout__body">
      <p>The model is first fit on all features to compute importances; features above the threshold are retained and the model is refit on only those, reducing noise and inference cost.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-25" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Predict</span>
    </div>
    <div class="code-callout__body">
      <p>Prediction uses only the selected feature subset stored during fit, so test data is automatically filtered to match the training column set.</p>
    </div>
  </div>
</aside>
</div>

### 2. Memory-Efficient Implementation

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class MemoryEfficientRF:
    """Memory-efficient Random Forest implementation"""
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y, batch_size=10):
        """Train trees in batches to save memory"""
        for i in range(0, self.n_estimators, batch_size):
            # Train batch of trees
            batch_trees = [
                RandomForestClassifier(n_estimators=1)
                for _ in range(min(batch_size,
                                 self.n_estimators - i))
            ]

            # Fit each tree
            for tree in batch_trees:
                # Bootstrap sample
                idx = np.random.choice(
                    len(X), size=len(X), replace=True
                )
                tree.fit(X.iloc[idx], y.iloc[idx])

            self.trees.extend(batch_trees)

    def predict(self, X):
        """Predict using majority vote"""
        predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Stores target tree count and an empty list that will hold each individually trained tree; fitting in batches avoids materializing all trees in memory at once.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-25" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Batch Training</span>
    </div>
    <div class="code-callout__body">
      <p>Trees are built in groups of <code>batch_size</code>; each single-estimator RF is fit on a fresh bootstrap sample, then appended to the list, keeping peak memory proportional to one batch.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="27-36" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Majority Vote</span>
    </div>
    <div class="code-callout__body">
      <p>All trees predict; <code>np.apply_along_axis</code> with <code>bincount(...).argmax()</code> picks the most frequent class label across the ensemble for each sample.</p>
    </div>
  </div>
</aside>
</div>

## Advanced Evaluation Metrics

### 1. Custom Evaluation Framework

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class AdvancedRFEvaluator:
    """Advanced evaluation metrics for Random Forest"""
    def __init__(self, model):
        self.model = model

    def evaluate_stability(self, X, y, n_iterations=10):
        """Evaluate feature importance stability"""
        importance_matrices = []

        for _ in range(n_iterations):
            # Bootstrap sample
            idx = np.random.choice(len(X), size=len(X))
            X_boot, y_boot = X.iloc[idx], y.iloc[idx]

            # Fit model and get importance
            self.model.fit(X_boot, y_boot)
            importance_matrices.append(
                self.model.feature_importances_
            )

        # Calculate stability metrics
        importance_std = np.std(importance_matrices, axis=0)
        stability_score = 1 / (1 + np.mean(importance_std))

        return stability_score

    def feature_importance_confidence(self, X, y,
                                    confidence_level=0.95):
        """Calculate confidence intervals for feature importance"""
        n_bootstrap = 1000
        n_features = X.shape[1]

        # Bootstrap feature importances
        importances = np.zeros((n_bootstrap, n_features))

        for i in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), size=len(X))
            X_boot, y_boot = X.iloc[idx], y.iloc[idx]

            # Get feature importance
            self.model.fit(X_boot, y_boot)
            importances[i] = self.model.feature_importances_

        # Calculate confidence intervals
        lower = np.percentile(importances,
                            (1 - confidence_level) / 2 * 100,
                            axis=0)
        upper = np.percentile(importances,
                            (1 + confidence_level) / 2 * 100,
                            axis=0)

        return pd.DataFrame({
            'feature': X.columns,
            'importance_mean': np.mean(importances, axis=0),
            'importance_lower': lower,
            'importance_upper': upper
        })
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Wraps any fitted RF model to add two post-hoc analysis methods: a stability score across bootstrap refits and bootstrap confidence intervals for each feature's importance.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-24" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Stability Score</span>
    </div>
    <div class="code-callout__body">
      <p>The model is refit on 10 bootstrap samples; the standard deviation of importances across runs captures how consistently each feature is ranked, converted to a 0–1 stability score.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="26-57" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Confidence Intervals</span>
    </div>
    <div class="code-callout__body">
      <p>1 000 bootstrap refits build a distribution of importances per feature; percentile-based lower and upper bounds form a 95% CI returned as a tidy DataFrame for reporting.</p>
    </div>
  </div>
</aside>
</div>

## Interpretability Techniques

### 1. Partial Dependence Plots

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.inspection import partial_dependence

def plot_partial_dependence(model, X, feature_names):
    """Create partial dependence plots for specified features"""
    fig, axes = plt.subplots(
        len(feature_names), 1,
        figsize=(10, 5*len(feature_names))
    )

    for idx, feature in enumerate(feature_names):
        # Calculate partial dependence
        pdp = partial_dependence(
            model, X, [feature],
            kind='average'
        )

        # Plot
        axes[idx].plot(pdp[1][0], pdp[0][0])
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Partial dependence')

    plt.tight_layout()
    plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Subplot Grid</span>
    </div>
    <div class="code-callout__body">
      <p>Creates one subplot per feature with 5-inch height each; dynamically scaling the figure height keeps plots readable regardless of how many features are requested.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-24" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">PDP per Feature</span>
    </div>
    <div class="code-callout__body">
      <p><code>partial_dependence(kind='average')</code> returns grid values and average predictions; plotting <code>pdp[1][0]</code> vs <code>pdp[0][0]</code> shows how the marginal prediction changes across the feature's range.</p>
    </div>
  </div>
</aside>
</div>

### 2. SHAP Values

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import shap

def analyze_shap_values(model, X):
    """Analyze SHAP values for feature importance"""
    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)

    # Plot summary
    shap.summary_plot(shap_values, X)

    # Return SHAP values for further analysis
    return shap_values
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">TreeExplainer</span>
    </div>
    <div class="code-callout__body">
      <p><code>shap.TreeExplainer</code> uses a tree-path algorithm to compute exact SHAP values efficiently for tree-based models — much faster than the model-agnostic kernel SHAP approach.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Summary Plot</span>
    </div>
    <div class="code-callout__body">
      <p><code>shap.summary_plot</code> shows a beeswarm of SHAP values per feature — each point is one sample, color encodes feature value, and x-position shows the impact on model output.</p>
    </div>
  </div>
</aside>
</div>

## Production Deployment

### 1. Model Versioning

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class VersionedRandomForest:
    """Random Forest with versioning capabilities"""
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        self.version = 1
        self.history = {}

    def fit(self, X, y):
        # Train model
        self.model.fit(X, y)

        # Save version info
        self.history[self.version] = {
            'timestamp': pd.Timestamp.now(),
            'n_samples': len(X),
            'feature_importance': dict(zip(
                X.columns,
                self.model.feature_importances_
            ))
        }

        self.version += 1
        return self

    def save_version(self, path):
        """Save model with version information"""
        save_dict = {
            'model': self.model,
            'version': self.version,
            'history': self.history
        }
        joblib.dump(save_dict, path)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Setup</span>
    </div>
    <div class="code-callout__body">
      <p>A <code>RandomForestClassifier</code> is created with forwarded kwargs; <code>version</code> and <code>history</code> will track each training run's metadata for auditability.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Versioned Fit</span>
    </div>
    <div class="code-callout__body">
      <p>After training, a snapshot of the timestamp, sample count, and per-feature importances is stored under the current version number; the counter increments for the next call.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-30" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Save Version</span>
    </div>
    <div class="code-callout__body">
      <p><code>joblib.dump</code> serializes the model, current version number, and full training history together so any saved checkpoint can be fully reconstructed later.</p>
    </div>
  </div>
</aside>
</div>

### 2. Online Learning

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class OnlineRandomForest:
    """Random Forest with online learning capabilities"""
    def __init__(self, n_estimators=100, buffer_size=1000):
        self.n_estimators = n_estimators
        self.buffer_size = buffer_size
        self.buffer_X = []
        self.buffer_y = []
        self.model = None

    def partial_fit(self, X, y):
        """Update model with new data"""
        # Add to buffer
        self.buffer_X.extend(X.values)
        self.buffer_y.extend(y.values)

        # If buffer is full, retrain
        if len(self.buffer_X) >= self.buffer_size:
            # Convert to arrays
            X_train = np.array(self.buffer_X)
            y_train = np.array(self.buffer_y)

            # Train new model
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators
            ).fit(X_train, y_train)

            # Clear buffer
            self.buffer_X = []
            self.buffer_y = []

        return self
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Two lists act as a rolling buffer for incoming samples; <code>buffer_size</code> controls how many new points trigger a full model retrain, balancing freshness against compute cost.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-31" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Partial Fit</span>
    </div>
    <div class="code-callout__body">
      <p>Each call appends new rows to the buffer; when the buffer reaches <code>buffer_size</code>, a fresh <code>RandomForestClassifier</code> is trained on all buffered data and the buffer is cleared for the next window.</p>
    </div>
  </div>
</aside>
</div>

## Gotchas

- **`StackingClassifier` leaks if base models are trained on the full training set** — sklearn's `StackingClassifier` uses `cv` to generate out-of-fold predictions for the meta-learner by default; manually fitting base models on the whole training set and then stacking them allows the meta-learner to see training predictions, inflating performance estimates.
- **Permutation importance can be misleading when features are correlated** — shuffling one correlated feature still leaves its information accessible through the correlated partner, so both features will look less important than they truly are; prefer partial dependence plots for correlated settings.
- **The `DynamicFeatureSelector` refits on a subset without re-tuning hyperparameters** — after dropping low-importance features and refitting, the original `max_depth` or `n_estimators` may no longer be optimal for the reduced feature set; the two-pass approach needs its own hyperparameter validation.
- **`OnlineRandomForest.partial_fit` loses all historical data on each buffer flush** — the implementation retrains from scratch on the current buffer window only, discarding older examples; this is not true online learning but windowed batch retraining, which can cause catastrophic forgetting on drifting data.
- **SHAP's `TreeExplainer` returns a list of arrays for multi-class classifiers** — `shap_values` is a list of length `n_classes`, not a single 2D array; passing the raw return value to `shap.summary_plot` for a binary classifier will work, but for multi-class you must index into the list (e.g., `shap_values[1]` for class 1).
- **`partial_dependence` with `kind='average'` averages over the marginal distribution of other features** — this can produce unrealistic feature combinations (e.g., a very high income with a very low credit score) that the model was never trained on, leading to extrapolated PDP curves that don't reflect real-world behaviour.

## Next Steps

Ready to see Random Forests in action? Continue to [Applications](5-applications.md) to explore real-world use cases!
