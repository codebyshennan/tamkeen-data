---
reading_minutes: 18
objectives:
  - Tell high bias (underfitting) from high variance (overfitting) using a train-vs-validation gap or a learning curve.
  - Pick the right fix for each side of the tradeoff — more/fewer features, more data, or regularization.
  - Explain regularization as a penalty on large coefficients, and tell L1 (Lasso, drops features) from L2 (Ridge, shrinks features).
  - Tune the regularization strength (`alpha`) with cross-validation instead of guessing.
---

# Bias, Variance, and Regularization — the Simple Version

**After this lesson:** you can look at a model's behaviour, name *why* it is wrong (too simple or too sensitive), and reach for the right fix — including the regularization dial that L1 and L2 turn.

## Overview

Every model can be wrong in two opposite ways:

- **Bias** — the model is **too simple** and misses the real pattern (this is called **underfitting**).
- **Variance** — the model is **too sensitive** and chases the noise in this particular dataset (this is called **overfitting**).

You cannot usually remove both at once — pushing one down tends to push the other up. That push-and-pull is the **bias–variance tradeoff**, and **regularization** is one of the cleanest ways to steer it.

**Prerequisites:** [What is ML?](what-is-ml.md) and the [workflow](ml-workflow.md) lesson. A deeper treatment lives in [5.5 Model evaluation](../5.5-model-eval/regularization.md).

## Why this matters

Almost every modelling decision — adding features, growing a deeper tree, turning regularization up or down — moves bias and variance in opposite directions. If you can *name* which one is hurting you, the fix is usually obvious. If you can't, you end up guessing.

## Helpful video

Crash Course AI: how supervised learning fits into ML workflows.

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qVRBYAdLAo" title="Supervised Learning: Crash Course AI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## The dartboard picture

Imagine throwing darts at a bullseye:

- **High bias** — your darts land tightly together, but in the *wrong* spot. Consistent, but consistently off. (The model is too rigid.)
- **High variance** — your darts scatter all over the board. Sometimes close, sometimes wild. (The model reacts too much to small changes.)
- **The goal** — a tight cluster *on* the bullseye: low bias **and** low variance.

{% include mermaid-diagram.html src="5-ml-fundamentals/5.1-intro-to-ml/diagrams/bias-variance-1.mmd" %}

## Seeing it in code

Let's make a tiny dataset where we *know* the true answer, then watch models of different complexity succeed and fail on it. Every code block below shows its real output, so you can see exactly what each change does.

First, the data: a smooth wave with some random noise sprinkled on top. In real life we only ever see the noisy dots — never the clean line.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# The TRUE pattern we want to learn (in real life we never see this directly)
def true_pattern(x):
    return np.sin(2 * np.pi * x)

# What we actually observe: the true pattern plus random noise
X = np.sort(np.random.rand(30))
y = true_pattern(X) + np.random.normal(0, 0.15, size=30)

Xc = X.reshape(-1, 1)          # sklearn wants a 2-D column
grid = np.linspace(0, 1, 200)  # smooth x-axis for drawing curves

plt.figure(figsize=(6, 4))
plt.scatter(X, y, color="#2563eb", label="Data we observe (noisy)")
plt.plot(grid, true_pattern(grid), "--", color="#6b7280", label="True pattern (hidden)")
plt.title("A smooth pattern hidden under noise")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.show()
```

### Too simple, just right, too complex

We now fit the same data three times, changing only **one knob**: the polynomial *degree* (how wiggly the model is allowed to be). Degree 1 is a straight line; degree 15 can bend almost anywhere.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

settings = [
    (1,  "Degree 1 - underfit (high bias)"),
    (4,  "Degree 4 - just right"),
    (15, "Degree 15 - overfit (high variance)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (degree, title) in zip(axes, settings):
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    model.fit(Xc, y)
    ax.scatter(X, y, color="#2563eb", s=20)
    ax.plot(grid, model.predict(grid.reshape(-1, 1)), color="#dc2626", lw=2, label="Model")
    ax.plot(grid, true_pattern(grid), "--", color="#6b7280", label="True pattern")
    ax.set_ylim(-1.6, 1.6); ax.set_title(title)
axes[0].legend(loc="lower center")
fig.suptitle("Same data, three levels of model complexity")
plt.show()
```

Read the three panels left to right:

- **Degree 1 (high bias):** a straight line can't bend to follow a wave, so it's wrong almost everywhere — but it would be wrong in the *same* way on any sample.
- **Degree 4 (just right):** flexible enough to trace the true wave, not so flexible that it chases every dot.
- **Degree 15 (high variance):** the curve twists through individual noisy points. It nails *this* data and would look completely different on a fresh sample.

### Putting numbers on it

Eyeballing curves is fine for one feature, but normally you can't plot the data. The reliable signal is the **gap between training error and validation error**. We measure error with RMSE (lower is better): how the model does on data it *trained on* vs data it has *never seen* (via cross-validation).

```python
from sklearn.model_selection import cross_val_score

print(f"{'degree':>6} | {'train RMSE':>10} | {'cross-val RMSE':>14}")
print("-" * 38)
for degree in [1, 4, 15]:
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    model.fit(Xc, y)
    train_rmse = np.sqrt(np.mean((model.predict(Xc) - y) ** 2))
    cv_rmse = -cross_val_score(model, Xc, y, cv=5,
                               scoring="neg_root_mean_squared_error").mean()
    print(f"{degree:>6} | {train_rmse:>10.3f} | {cv_rmse:>14.3f}")
```

The pattern in those numbers is the whole lesson:

| Symptom in the numbers | Diagnosis | What it means |
| --- | --- | --- |
| Train **and** validation error both high | **High bias** (underfit) | Model is too simple — it can't even fit the data it has seen. |
| Train error low, validation error **much** higher | **High variance** (overfit) | Model memorised the training noise and falls apart on new data. |
| Both low and **close together** | **Good fit** | This is what you want. |

> A small gap alone is **not** good news. Two *high* errors that are close together still means underfitting. Both errors must be low **and** close.

### Learning curves: does more data help?

A **learning curve** plots score against the amount of training data. It answers a practical question: *would collecting more data fix this?*

```python
from sklearn.model_selection import learning_curve

# Larger sample so the curves are smooth
np.random.seed(1)
X_big = np.sort(np.random.rand(200)).reshape(-1, 1)
y_big = true_pattern(X_big.ravel()) + np.random.normal(0, 0.15, size=200)

model = make_pipeline(PolynomialFeatures(15, include_bias=False), LinearRegression())
sizes, train_scores, val_scores = learning_curve(
    model, X_big, y_big, cv=5, train_sizes=np.linspace(0.1, 1.0, 8))

plt.figure(figsize=(6, 4))
plt.plot(sizes, train_scores.mean(axis=1), "o-", color="#2563eb", label="Training score")
plt.plot(sizes, val_scores.mean(axis=1), "o-", color="#dc2626", label="Validation score")
plt.ylim(0, 1.05)
plt.title("Learning curve: the gap shrinks as data grows")
plt.xlabel("Training examples"); plt.ylabel("R² score"); plt.legend()
plt.show()
```

- A **wide gap** that closes as you add data → **high variance**. More data helps.
- Both curves **flat and low**, close together → **high bias**. More data won't help; you need a better model or features.

## The fix menu

Once you've named the problem, the fix is short:

| If you have… | High bias (underfit) | High variance (overfit) |
| --- | --- | --- |
| Model complexity | **Increase** it (higher degree, deeper tree) | **Decrease** it (lower degree, prune the tree) |
| Features | **Add** useful ones | **Remove** irrelevant ones |
| More data | Won't help much | **Helps** |
| Regularization | **Less** of it | **More** of it |

That last row is what the rest of this page is about.

## Regularization: a dial that fights overfitting

Look again at the degree-15 curve — it had to swing violently up and down to thread every noisy point. To swing that hard, it needs **huge coefficients**. That's the tell: **overfit models tend to have large coefficients.**

**Regularization adds a penalty for large coefficients** to what the model is trying to minimise. Now the model has to balance two goals: *fit the data* **and** *keep coefficients small*. Made to choose, it gives up the wild swings and settles on a smoother curve — trading a tiny bit of training accuracy for much better generalization.

A single knob, **`alpha`** (sometimes written λ), controls how hard you push:

- `alpha = 0` → no penalty, back to the plain overfit model.
- small `alpha` → gentle nudge toward simplicity.
- large `alpha` → strong push; go too far and you *underfit*.

There are two common flavours, and the difference is exactly how they measure "large coefficients."

### L2 (Ridge) vs L1 (Lasso)

- **L2 — Ridge** penalises the **sum of squared** coefficients. Squaring punishes big coefficients hard but never *quite* reaches zero, so Ridge **shrinks every coefficient toward zero but keeps them all**. Use it when you think many features each contribute a little.
- **L1 — Lasso** penalises the **sum of absolute** coefficients. This shape lets it push weak coefficients **exactly to zero**, which effectively **deletes those features** — automatic feature selection. Use it when you think only a few features really matter.

Watch both calm the same wild curve:

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

def poly_model(estimator):
    # degree-15 features, standardized so the penalty hits each term fairly
    return make_pipeline(PolynomialFeatures(15, include_bias=False),
                         StandardScaler(), estimator)

fits = [
    ("No penalty (overfits)",   poly_model(LinearRegression())),
    ("Ridge - L2 (alpha=0.1)",  poly_model(Ridge(alpha=0.1))),
    ("Lasso - L1 (alpha=0.01)", poly_model(Lasso(alpha=0.01, max_iter=10000))),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, model) in zip(axes, fits):
    model.fit(Xc, y)
    ax.scatter(X, y, color="#2563eb", s=20)
    ax.plot(grid, model.predict(grid.reshape(-1, 1)), color="#dc2626", lw=2)
    ax.plot(grid, true_pattern(grid), "--", color="#6b7280")
    ax.set_ylim(-1.6, 1.6); ax.set_title(title)
fig.suptitle("Regularization calms the wild degree-15 curve")
plt.show()
```

The left panel is the same overfit mess as before. The middle and right panels — *same degree-15 model*, only a penalty added — recover a smooth, sensible curve.

Now the headline difference between L1 and L2. Both shrink coefficients; only L1 sets them to **exactly zero**:

```python
ridge = poly_model(Ridge(alpha=0.1)).fit(Xc, y)
lasso = poly_model(Lasso(alpha=0.01, max_iter=10000)).fit(Xc, y)

ridge_coefs = ridge.named_steps["ridge"].coef_
lasso_coefs = lasso.named_steps["lasso"].coef_

count_zero = lambda c: int(np.sum(np.abs(c) < 1e-4))
print(f"15 polynomial features in total")
print(f"Ridge (L2): {count_zero(ridge_coefs):2d} coefficients set to zero  -> keeps every feature, just smaller")
print(f"Lasso (L1): {count_zero(lasso_coefs):2d} coefficients set to zero  -> drops features automatically")
```

| | L2 — Ridge | L1 — Lasso |
| --- | --- | --- |
| Penalty | sum of **squared** coefficients | sum of **absolute** coefficients |
| Effect on coefficients | shrinks all toward zero | pushes weak ones to **exactly zero** |
| Feature selection? | No — keeps every feature | **Yes** — drops features for free |
| Best when | many features each matter a bit | only a few features matter |
| (Want both? | use **ElasticNet** — a blend of L1 and L2) | |

> **Always standardize first.** The penalty compares coefficients directly, so features must be on the same scale — otherwise a feature measured in small units gets unfairly punished. That's why the pipeline above includes `StandardScaler`.

### Choosing `alpha` without guessing

Don't hand-pick `alpha`. Let cross-validation try a range and keep the best — `RidgeCV` (and `LassoCV`) do exactly this:

```python
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-3, 2, 30)  # 0.001 ... 100
model = make_pipeline(PolynomialFeatures(15, include_bias=False),
                     StandardScaler(), RidgeCV(alphas=alphas))
model.fit(Xc, y)
print(f"Cross-validation tried 30 values and picked alpha = {model.named_steps['ridgecv'].alpha_:.3f}")
```

## Gotchas

- **Judging a model by its training score alone** — a perfect training score tells you *nothing* about variance. Always compare training and validation scores side by side before naming the problem.
- **Reading a small gap as "good fit"** — a small gap is necessary but not sufficient. Both scores must also be *good*. Two poor scores close together is high bias, not success.
- **Tuning on the test set** — every peek at the test set leaks information and inflates your estimate. Tune with cross-validation; touch the test set once, at the very end.
- **Forgetting to standardize before regularizing** — without it the penalty is uneven across features, and `alpha` means different things for different columns.
- **Setting `alpha` too high** — regularization fixes overfitting, but overdo it and you swing all the way to underfitting. Let cross-validation pick the value.

## Next steps

- Go deeper on the math and on ElasticNet in [5.5 Model evaluation → Regularization](../5.5-model-eval/regularization.md).
- Try the same degree-sweep on a dataset you care about, and keep a short log of how the train/validation gap moves as you change each knob.

Remember: name the failure mode first (bias or variance), *then* pick the fix. Half the battle is reading the symptom correctly.
