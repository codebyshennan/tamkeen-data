"""Illustrative good-vs-bad diagnostic plots for the "Four Key Questions to Ask
About Your Model" section of model-diagnostics.md. Each figure shows a healthy
diagnostic on the left and a problematic one on the right, so learners can see
what each check looks like before reading the code.
"""
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

np.random.seed(42)
BAD = "#c0392b"
GOOD = "#27ae60"


def residuals(x, y):
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    pred = model.predict(x.reshape(-1, 1))
    return pred, y - pred


# ----------------------------------------------------------------------
# Question 1: Is the relationship actually straight? (Linearity)
# Residuals vs fitted — good: random scatter; bad: clear curve.
# ----------------------------------------------------------------------
n = 120
x = np.linspace(1, 10, n)
y_good = 2.5 * x + np.random.normal(0, 1.2, n)
y_bad = 0.4 * x**2 + np.random.normal(0, 1.2, n)
fit_g, res_g = residuals(x, y_good)
fit_b, res_b = residuals(x, y_bad)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fit_g, res_g, alpha=0.7, color=GOOD)
axes[0].set_title("Healthy: residuals scatter randomly\naround the zero line")
axes[1].scatter(fit_b, res_b, alpha=0.7, color=BAD)
axes[1].set_title("Problem: a clear curve means the\nrelationship isn't straight")
for ax in axes:
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.grid(True, alpha=0.3)
fig.suptitle(
    "Question 1 — Linearity (residuals vs fitted)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    ASSETS_DIR / "model-diagnostics_question1.png", dpi=200, bbox_inches="tight"
)
plt.close()


# ----------------------------------------------------------------------
# Question 2: Are the observations independent? (Independence)
# Residuals in observation order — good: no pattern; bad: adjacent
# residuals track each other (positive autocorrelation, low Durbin-Watson).
# ----------------------------------------------------------------------
n = 120
order = np.arange(n)
res_indep = np.random.normal(0, 1, n)
# autocorrelated residuals: each point pulled toward the previous one
res_corr = np.zeros(n)
res_corr[0] = np.random.normal(0, 1)
for i in range(1, n):
    res_corr[i] = 0.85 * res_corr[i - 1] + np.random.normal(0, 0.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(order, res_indep, marker="o", markersize=3, linewidth=0.8, color=GOOD)
axes[0].set_title(
    "Healthy: no run pattern — adjacent\nresiduals are unrelated (DW ≈ 2)"
)
axes[1].plot(order, res_corr, marker="o", markersize=3, linewidth=0.8, color=BAD)
axes[1].set_title(
    "Problem: residuals drift in runs —\nadjacent points track each other (DW < 1)"
)
for ax in axes:
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Observation order")
    ax.set_ylabel("Residuals")
    ax.grid(True, alpha=0.3)
fig.suptitle(
    "Question 2 — Independence (residuals in order)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    ASSETS_DIR / "model-diagnostics_question2.png", dpi=200, bbox_inches="tight"
)
plt.close()


# ----------------------------------------------------------------------
# Question 3: Is the error spread consistent? (Homoscedasticity)
# |residuals| vs fitted — good: even band; bad: funnel.
# ----------------------------------------------------------------------
n = 150
x = np.linspace(1, 10, n)
y_good = 2.5 * x + np.random.normal(0, 1.5, n)
y_bad = 2.5 * x + np.random.normal(0, 1, n) * x  # spread grows with x
fit_g, res_g = residuals(x, y_good)
fit_b, res_b = residuals(x, y_bad)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fit_g, np.abs(res_g), alpha=0.7, color=GOOD)
axes[0].set_title("Healthy: error size stays\nconstant across fitted values")
axes[1].scatter(fit_b, np.abs(res_b), alpha=0.7, color=BAD)
axes[1].set_title("Problem: a funnel — errors grow\nas the fitted value grows")
for ax in axes:
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("|Residuals|")
    ax.grid(True, alpha=0.3)
fig.suptitle(
    "Question 3 — Homoscedasticity (error spread)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    ASSETS_DIR / "model-diagnostics_question3.png", dpi=200, bbox_inches="tight"
)
plt.close()


# ----------------------------------------------------------------------
# Question 4: Do the errors follow a bell curve? (Normality)
# Q-Q plot — good: points on the line; bad: points curve away at the ends.
# ----------------------------------------------------------------------
n = 200
res_normal = np.random.normal(0, 1, n)
res_skewed = np.random.exponential(1, n) - 1  # right-skewed, heavy tail

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
stats.probplot(res_normal, dist="norm", plot=axes[0])
axes[0].get_lines()[0].set_color(GOOD)
axes[0].get_lines()[0].set_markerfacecolor(GOOD)
axes[0].set_title("Healthy: points hug the diagonal —\nerrors are roughly normal")
stats.probplot(res_skewed, dist="norm", plot=axes[1])
axes[1].get_lines()[0].set_color(BAD)
axes[1].get_lines()[0].set_markerfacecolor(BAD)
axes[1].set_title("Problem: points curve away at the\nends — skew or heavy tails")
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle("Question 4 — Normality (Q-Q plot)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(
    ASSETS_DIR / "model-diagnostics_question4.png", dpi=200, bbox_inches="tight"
)
plt.close()

print("Generated four-questions good-vs-bad images:")
for i in range(1, 5):
    print(f"{i}. assets/model-diagnostics_question{i}.png")
