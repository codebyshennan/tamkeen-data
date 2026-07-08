---
reading_minutes: 20
objectives:
  - >-
    Combine adaptive regularization (Adaptive Lasso, group lasso) with
    stratified or grouped cross-validation when groups in the data should be
    respected.
  - >-
    Apply dropout and weight decay in neural networks, and explain why dropout's
    masking behaves like noisy ensembling.
  - >-
    Decide when to *combine* regularizers (e.g., L2 + dropout in a deep model)
    and when one is enough.
---

# Advanced Regularization Techniques

**After this lesson:** you can explain Advanced Regularization Techniques and try the examples in your own notebook.

## Overview

Dropout and other NN-centric regularizers vs classical penalties; when to combine approaches.

## Adaptive Regularization

Adaptive regularization is like having a smart teacher who adjusts their teaching style based on each student's needs. Instead of treating all features the same, it gives more attention to the important ones.

### 1. Adaptive Lasso

Class Init

Stores regularization strength (`alpha`), weight-adjustment power (`gamma`), and placeholders for the adaptive weights and the inner Lasso model.

Adaptive Weights and Fit

An OLS pass estimates initial coefficients; features with small coefficients get large weights (1/|β|^γ), making the subsequent Lasso penalise them more aggressively and thus driving them to zero.

Predict

Re-applies the same feature scaling learned during `fit` before passing new data to the inner Lasso, ensuring training and inference use identical transformations.

### 2. Group Lasso

Group Lasso is like having a team coach who manages groups of players together, rather than individual players. It's useful when you have related features that should be selected or dropped as a group.

Penalty Function

Computes the Group Lasso penalty as the sum over groups of √(group size) × L2 norm, larger groups are penalised more heavily in proportion to their size.

Proximal Gradient Fit

Each iteration takes a gradient step on the squared loss, then applies a proximal shrinkage operator per group: if the group's L2 norm is smaller than `alpha` it collapses the entire group to zero, otherwise it shrinks it uniformly.

## Advanced Optimization Techniques

### 1. Coordinate Descent

Coordinate descent is like solving a puzzle one piece at a time. Instead of trying to solve everything at once, it focuses on one feature at a time.

Setup

Initialises all coefficients to zero; the outer loop repeats until convergence or `max_iter` is reached, copying the previous coefficients to detect when updates become negligible.

Soft-threshold Update

For each feature `j`, computes the partial residual by temporarily removing its contribution, then applies the Lasso soft-threshold rule: set to zero if the correlation `rho` is within ±alpha, otherwise shrink by alpha.

### 2. ADMM Implementation

ADMM (Alternating Direction Method of Multipliers) is like having two people work together to solve a problem, each focusing on their part while coordinating with the other.

Variables and Precompute

Three variable vectors (beta, z, u) initialise the ADMM state; the Cholesky factorisation of (X'X + ρI) is computed once outside the loop so each iteration only requires two triangular solves instead of a full matrix inversion.

ADMM Update Steps

Each iteration alternates: (1) update beta via a ridge solve, (2) update z with the Lasso soft-threshold proximal operator, (3) accumulate the dual residual u, convergence is declared when z stops changing.

## Advanced Cross-Validation

### 1. Stability Selection

Stability selection is like taking multiple tests to ensure you really understand the material, not just memorizing the answers. It helps identify features that are consistently important.

Class Setup

Stores any sparse estimator (e.g. Lasso), the number of bootstrap repetitions, a selection-frequency threshold, and a placeholder for the resulting selection probabilities.

Bootstrap Counting

Each iteration draws a half-sample (without replacement), refits the estimator, and increments a counter for every feature with a non-zero coefficient; dividing by `n_bootstrap` gives the fraction of runs in which each feature was selected.

### 2. Randomized Lasso

Randomized Lasso is like having multiple teachers evaluate a student, each with slightly different criteria. This helps identify features that are robustly important.

Init Parameters

Stores the Lasso `alpha`, a `scaling` lower bound for random feature attenuation (e.g. 0.5 means each feature may be weakened to 50-100% of its original magnitude), and the number of repetitions.

Randomised Resampling

Each run draws a random per-feature scaling factor, trains Lasso on the perturbed features, and counts which coefficients survive; dividing by `n_resampling` gives robustness scores, features that survive across many random perturbations are genuinely informative.

## Regularization for Neural Networks

### 1. Weight Decay Implementation

Weight decay in neural networks is like having rules that prevent the network from becoming too complex, similar to how regularization works in linear models.

Import

TensorFlow is imported; all layers, regularizers, and the Sequential API come from this single namespace.

Regularised Architecture

Both hidden Dense layers carry an L2 weight penalty (`kernel_regularizer`) that adds squared weight magnitudes to the loss, while Dropout layers (30% and 20%) independently drop neurons during training, combining two complementary regularisation strategies.

## Common Mistakes to Avoid

1. Using too complex regularization when simpler methods would work
2. Not understanding the assumptions behind each method
3. Ignoring feature scaling in advanced methods
4. Not validating the stability of selected features
5. Overlooking the computational cost of advanced methods

## Next Steps

Now that you understand advanced regularization techniques, move on to [Applications](5-applications.md) to see how these methods are used in real-world scenarios!

## Gotchas

* **`AdaptiveLasso` divides by `ols.coef_`, which will crash if any OLS coefficient is exactly zero**: `1 / (np.abs(ols.coef_) ** gamma)` produces `inf` or `NaN` for zero coefficients; in practice, add a small epsilon (`1e-8`) to the denominator before computing adaptive weights to avoid numerical failures.
* **`GroupLasso.fit` uses a fixed step size of `0.01` with no line search**: gradient descent with a hard-coded step can diverge or converge very slowly depending on the scale of `X`; for data with large-magnitude features, normalise columns first or the 1 000-iteration limit will terminate before convergence.
* **`StabilitySelection` refits the estimator in place on each bootstrap, destroying its previous state**: Lasso stores `coef_` after each fit, so checking `coef_ != 0` after `estimator.fit(X_boot, y_boot)` gives the current bootstrap's result, not a cumulative one; the implementation is correct, but learners who pass a stateful custom estimator may see unexpected behaviour.
* **Combining L2 weight decay and Dropout in the neural network creates redundant regularisation at low noise**: both mechanisms reduce effective capacity; with clean, low-dimensional data they can interact to under-fit; start with one regulariser and add the second only if validation loss is still high.
* **`np.linalg.cholesky` in `admm_lasso` will raise `LinAlgError` if `X'X + ρI` is not positive definite**: this should not happen mathematically (ρI guarantees positive definiteness), but floating-point rounding on poorly conditioned X can cause it; scaling features and increasing ρ resolves the issue.
* **`RandomizedLasso` is deprecated and removed from scikit-learn**: the class was removed in sklearn 0.25; the custom implementation here recreates the concept manually, but learners who search for it in the sklearn docs will find it missing; use `StabilitySelection` with a Lasso estimator as the modern equivalent.

## Additional Resources

* [Advanced Regularization Techniques](https://towardsdatascience.com/advanced-regularization-techniques-1c4e6b5c5343)
* [Stability Selection in Practice](https://www.stat.berkeley.edu/~bickel/papers/2010_StabilitySelection.pdf)
* [ADMM for Machine Learning](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
