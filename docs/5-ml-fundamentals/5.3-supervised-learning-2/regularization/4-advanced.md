---
reading_minutes: 20
objectives:
  - "Combine adaptive regularization (Adaptive Lasso, group lasso) with stratified or grouped cross-validation when groups in the data should be respected."
  - "Apply dropout and weight decay in neural networks, and explain why dropout's masking behaves like noisy ensembling."
  - "Decide when to *combine* regularizers (e.g., L2 + dropout in a deep model) and when one is enough."
---

# Advanced Regularization Techniques

**After this lesson:** you can explain Advanced Regularization Techniques and try the examples in your own notebook.

## Overview

Dropout and other NN-centric regularizers vs classical penalties; when to combine approaches.


{% include mermaid-diagram.html src="5-ml-fundamentals/5.3-supervised-learning-2/regularization/diagrams/4-advanced-1.mmd" %}

## Adaptive Regularization

Adaptive regularization is like having a smart teacher who adjusts their teaching style based on each student's needs. Instead of treating all features the same, it gives more attention to the important ones.

### 1. Adaptive Lasso

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class AdaptiveLasso:
    """Adaptive Lasso implementation"""
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.weights = None
        self.lasso = None

    def fit(self, X, y):
        # Initial OLS fit to estimate feature importance
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression()
        ols.fit(X, y)

        # Compute adaptive weights: penalise weak features more
        self.weights = 1 / (np.abs(ols.coef_) ** self.gamma)

        # Scale features by weights, then fit Lasso
        X_weighted = X * self.weights
        self.lasso = Lasso(alpha=self.alpha)
        self.lasso.fit(X_weighted, y)

        return self

    def predict(self, X):
        X_weighted = X * self.weights
        return self.lasso.predict(X_weighted)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Init</span>
    </div>
    <div class="code-callout__body">
      <p>Stores regularization strength (<code>alpha</code>), weight-adjustment power (<code>gamma</code>), and placeholders for the adaptive weights and the inner Lasso model.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Adaptive Weights and Fit</span>
    </div>
    <div class="code-callout__body">
      <p>An OLS pass estimates initial coefficients; features with small coefficients get large weights (1/|β|^γ), making the subsequent Lasso penalise them more aggressively and thus driving them to zero.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-27" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Predict</span>
    </div>
    <div class="code-callout__body">
      <p>Re-applies the same feature scaling learned during <code>fit</code> before passing new data to the inner Lasso, ensuring training and inference use identical transformations.</p>
    </div>
  </div>
</aside>
</div>

### 2. Group Lasso

Group Lasso is like having a team coach who manages groups of players together, rather than individual players. It's useful when you have related features that should be selected or dropped as a group.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def group_lasso_penalty(coef_groups):
    """Calculate group lasso penalty"""
    return sum(
        np.sqrt(len(group)) * np.linalg.norm(group, 2)
        for group in coef_groups
    )

class GroupLasso:
    """Group Lasso implementation"""
    def __init__(self, groups, alpha=1.0):
        self.groups = groups
        self.alpha = alpha

    def fit(self, X, y):
        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)

        for _ in range(1000):
            # Gradient step
            grad = -X.T @ (y - X @ self.coef_)
            self.coef_ -= 0.01 * grad

            # Proximal operator: shrink each group toward zero
            for group in self.groups:
                group_norm = np.linalg.norm(self.coef_[group], 2)
                if group_norm > 0:
                    shrinkage = max(0, 1 - self.alpha / group_norm)
                    self.coef_[group] *= shrinkage

        return self
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Penalty Function</span>
    </div>
    <div class="code-callout__body">
      <p>Computes the Group Lasso penalty as the sum over groups of √(group size) × L2 norm, larger groups are penalised more heavily in proportion to their size.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Proximal Gradient Fit</span>
    </div>
    <div class="code-callout__body">
      <p>Each iteration takes a gradient step on the squared loss, then applies a proximal shrinkage operator per group: if the group's L2 norm is smaller than <code>alpha</code> it collapses the entire group to zero, otherwise it shrinks it uniformly.</p>
    </div>
  </div>
</aside>
</div>

## Advanced Optimization Techniques

### 1. Coordinate Descent

Coordinate descent is like solving a puzzle one piece at a time. Instead of trying to solve everything at once, it focuses on one feature at a time.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def coordinate_descent_lasso(X, y, alpha, max_iter=1000):
    """Coordinate descent for Lasso"""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)

    for _ in range(max_iter):
        coef_old = coef.copy()

        # Update each coordinate one at a time
        for j in range(n_features):
            # Partial residual excluding feature j
            r = y - X @ coef + X[:, j] * coef[j]

            rho = X[:, j] @ r
            if abs(rho) <= alpha:
                coef[j] = 0  # Soft-threshold to zero
            else:
                coef[j] = (
                    np.sign(rho) * (abs(rho) - alpha) /
                    (X[:, j] @ X[:, j])
                )

        if np.allclose(coef, coef_old):
            break

    return coef
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Initialises all coefficients to zero; the outer loop repeats until convergence or <code>max_iter</code> is reached, copying the previous coefficients to detect when updates become negligible.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Soft-threshold Update</span>
    </div>
    <div class="code-callout__body">
      <p>For each feature <code>j</code>, computes the partial residual by temporarily removing its contribution, then applies the Lasso soft-threshold rule: set to zero if the correlation <code>rho</code> is within ±alpha, otherwise shrink by alpha.</p>
    </div>
  </div>
</aside>
</div>

### 2. ADMM Implementation

ADMM (Alternating Direction Method of Multipliers) is like having two people work together to solve a problem, each focusing on their part while coordinating with the other.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def admm_lasso(X, y, alpha, rho=1.0, max_iter=1000):
    """ADMM algorithm for Lasso"""
    n_samples, n_features = X.shape

    beta = np.zeros(n_features)  # Primal variable
    z = np.zeros(n_features)     # Auxiliary variable
    u = np.zeros(n_features)     # Dual (scaled) variable

    # Pre-compute Cholesky factorisation for fast solves
    XtX = X.T @ X
    L = np.linalg.cholesky(XtX + rho * np.eye(n_features))

    for _ in range(max_iter):
        # Beta update: ridge-like least squares
        q = X.T @ y + rho * (z - u)
        beta = np.linalg.solve(L.T, np.linalg.solve(L, q))

        # z update: soft-threshold (Lasso proximal operator)
        z_old = z
        beta_hat = beta + u
        z = np.sign(beta_hat) * np.maximum(
            np.abs(beta_hat) - alpha/rho, 0
        )

        # Dual variable update
        u = u + beta - z

        if np.allclose(z, z_old):
            break

    return beta
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Variables and Precompute</span>
    </div>
    <div class="code-callout__body">
      <p>Three variable vectors (beta, z, u) initialise the ADMM state; the Cholesky factorisation of (X'X + ρI) is computed once outside the loop so each iteration only requires two triangular solves instead of a full matrix inversion.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-28" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">ADMM Update Steps</span>
    </div>
    <div class="code-callout__body">
      <p>Each iteration alternates: (1) update beta via a ridge solve, (2) update z with the Lasso soft-threshold proximal operator, (3) accumulate the dual residual u, convergence is declared when z stops changing.</p>
    </div>
  </div>
</aside>
</div>

## Advanced Cross-Validation

### 1. Stability Selection

Stability selection is like taking multiple tests to ensure you really understand the material, not just memorizing the answers. It helps identify features that are consistently important.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class StabilitySelection:
    """Stability selection for feature selection"""
    def __init__(self, estimator, n_bootstrap=100, threshold=0.5):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.selection_probabilities_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        feature_counts = np.zeros(n_features)

        for _ in range(self.n_bootstrap):
            # Sample half the data without replacement
            indices = np.random.choice(
                n_samples, size=n_samples//2, replace=False
            )
            X_boot = X[indices]
            y_boot = y[indices]

            # Fit and count non-zero coefficients
            self.estimator.fit(X_boot, y_boot)
            feature_counts += (
                self.estimator.coef_ != 0
            ).astype(int)

        self.selection_probabilities_ = (
            feature_counts / self.n_bootstrap
        )

        return self
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Stores any sparse estimator (e.g. Lasso), the number of bootstrap repetitions, a selection-frequency threshold, and a placeholder for the resulting selection probabilities.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Bootstrap Counting</span>
    </div>
    <div class="code-callout__body">
      <p>Each iteration draws a half-sample (without replacement), refits the estimator, and increments a counter for every feature with a non-zero coefficient; dividing by <code>n_bootstrap</code> gives the fraction of runs in which each feature was selected.</p>
    </div>
  </div>
</aside>
</div>

### 2. Randomized Lasso

Randomized Lasso is like having multiple teachers evaluate a student, each with slightly different criteria. This helps identify features that are robustly important.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class RandomizedLasso:
    """Randomized Lasso implementation"""
    def __init__(self, alpha=1.0, scaling=0.5, n_resampling=100):
        self.alpha = alpha
        self.scaling = scaling
        self.n_resampling = n_resampling

    def fit(self, X, y):
        n_samples, n_features = X.shape
        scores = np.zeros(n_features)

        for _ in range(self.n_resampling):
            # Randomly weaken each feature (scaling in [scaling, 1.0])
            scalings = np.random.uniform(
                self.scaling, 1.0, size=n_features
            )
            X_scaled = X * scalings

            # Fit Lasso and accumulate non-zero counts
            lasso = Lasso(alpha=self.alpha)
            lasso.fit(X_scaled, y)
            scores += (lasso.coef_ != 0).astype(int)

        self.scores_ = scores / self.n_resampling
        return self
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Init Parameters</span>
    </div>
    <div class="code-callout__body">
      <p>Stores the Lasso <code>alpha</code>, a <code>scaling</code> lower bound for random feature attenuation (e.g. 0.5 means each feature may be weakened to 50-100% of its original magnitude), and the number of repetitions.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Randomised Resampling</span>
    </div>
    <div class="code-callout__body">
      <p>Each run draws a random per-feature scaling factor, trains Lasso on the perturbed features, and counts which coefficients survive; dividing by <code>n_resampling</code> gives robustness scores, features that survive across many random perturbations are genuinely informative.</p>
    </div>
  </div>
</aside>
</div>

## Regularization for Neural Networks

### 1. Weight Decay Implementation

Weight decay in neural networks is like having rules that prevent the network from becoming too complex, similar to how regularization works in linear models.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import tensorflow as tf

def create_regularized_model(input_shape, l2_lambda=0.01):
    """Create neural network with L2 regularization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            128, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
            input_shape=input_shape
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            64, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    return model
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-2" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Import</span>
    </div>
    <div class="code-callout__body">
      <p>TensorFlow is imported; all layers, regularizers, and the Sequential API come from this single namespace.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="4-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Regularised Architecture</span>
    </div>
    <div class="code-callout__body">
      <p>Both hidden Dense layers carry an L2 weight penalty (<code>kernel_regularizer</code>) that adds squared weight magnitudes to the loss, while Dropout layers (30% and 20%) independently drop neurons during training, combining two complementary regularisation strategies.</p>
    </div>
  </div>
</aside>
</div>

## Common Mistakes to Avoid

1. Using too complex regularization when simpler methods would work
2. Not understanding the assumptions behind each method
3. Ignoring feature scaling in advanced methods
4. Not validating the stability of selected features
5. Overlooking the computational cost of advanced methods

## Next Steps

Now that you understand advanced regularization techniques, move on to [Applications](5-applications.md) to see how these methods are used in real-world scenarios!

## Gotchas

- **`AdaptiveLasso` divides by `ols.coef_`, which will crash if any OLS coefficient is exactly zero**: `1 / (np.abs(ols.coef_) ** gamma)` produces `inf` or `NaN` for zero coefficients; in practice, add a small epsilon (`1e-8`) to the denominator before computing adaptive weights to avoid numerical failures.
- **`GroupLasso.fit` uses a fixed step size of `0.01` with no line search**: gradient descent with a hard-coded step can diverge or converge very slowly depending on the scale of `X`; for data with large-magnitude features, normalise columns first or the 1 000-iteration limit will terminate before convergence.
- **`StabilitySelection` refits the estimator in place on each bootstrap, destroying its previous state**: Lasso stores `coef_` after each fit, so checking `coef_ != 0` after `estimator.fit(X_boot, y_boot)` gives the current bootstrap's result, not a cumulative one; the implementation is correct, but learners who pass a stateful custom estimator may see unexpected behaviour.
- **Combining L2 weight decay and Dropout in the neural network creates redundant regularisation at low noise**: both mechanisms reduce effective capacity; with clean, low-dimensional data they can interact to under-fit; start with one regulariser and add the second only if validation loss is still high.
- **`np.linalg.cholesky` in `admm_lasso` will raise `LinAlgError` if `X'X + ρI` is not positive definite**: this should not happen mathematically (ρI guarantees positive definiteness), but floating-point rounding on poorly conditioned X can cause it; scaling features and increasing ρ resolves the issue.
- **`RandomizedLasso` is deprecated and removed from scikit-learn**: the class was removed in sklearn 0.25; the custom implementation here recreates the concept manually, but learners who search for it in the sklearn docs will find it missing; use `StabilitySelection` with a Lasso estimator as the modern equivalent.

## Additional Resources

- [Advanced Regularization Techniques](https://towardsdatascience.com/advanced-regularization-techniques-1c4e6b5c5343)
- [Stability Selection in Practice](https://www.stat.berkeley.edu/~bickel/papers/2010_StabilitySelection.pdf)
- [ADMM for Machine Learning](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
