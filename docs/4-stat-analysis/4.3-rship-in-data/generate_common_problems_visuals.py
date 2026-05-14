"""Before/after diagnostic plots for the "Common Problems and How to Fix Them"
section of model-diagnostics.md. Each figure shows the erroneous diagnostic on
the left and the same diagnostic after the recommended fix on the right.
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
    return model.predict(x.reshape(-1, 1)), y - model.predict(x.reshape(-1, 1))


# ----------------------------------------------------------------------
# Problem 1: The relationship isn't actually straight (non-linearity)
# Fix: add a quadratic term so the model can follow the curve.
# ----------------------------------------------------------------------
n = 120
x = np.linspace(1, 10, n)
y = 0.4 * x**2 + np.random.normal(0, 1.5, n)

fitted_bad, resid_bad = residuals(x, y)

X2 = np.column_stack([x, x**2])
model2 = LinearRegression().fit(X2, y)
fitted_good = model2.predict(X2)
resid_good = y - fitted_good

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted_bad, resid_bad, alpha=0.7, color=BAD)
axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
axes[0].set_title("Problem: straight-line model\n(residuals show a clear curve)")
axes[1].scatter(fitted_good, resid_good, alpha=0.7, color=GOOD)
axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
axes[1].set_title("Fixed: added a quadratic term\n(residuals now scatter randomly)")
for ax in axes:
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.grid(True, alpha=0.3)
fig.suptitle("Problem 1 — Non-linear relationship", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "model-diagnostics_problem1.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------
# Problem 2: Inconsistent error spread (heteroscedasticity)
# Fix: model log(y) so the multiplicative noise becomes additive.
# ----------------------------------------------------------------------
n = 150
x = np.linspace(1, 10, n)
y = 2 * x * np.exp(np.random.normal(0, 0.35, n))  # spread grows with x

fitted_bad, resid_bad = residuals(x, y)

logy = np.log(y)
fitted_log, resid_good = residuals(x, logy)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted_bad, np.sqrt(np.abs(resid_bad)), alpha=0.7, color=BAD)
axes[0].set_title("Problem: raw y\n(funnel — spread grows with fitted value)")
axes[0].set_ylabel("√|Residuals|")
axes[1].scatter(fitted_log, np.sqrt(np.abs(resid_good)), alpha=0.7, color=GOOD)
axes[1].set_title("Fixed: modelled log(y)\n(spread is now roughly even)")
axes[1].set_ylabel("√|Residuals| (log scale model)")
for ax in axes:
    ax.set_xlabel("Fitted values")
    ax.grid(True, alpha=0.3)
fig.suptitle(
    "Problem 2 — Inconsistent error spread (heteroscedasticity)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "model-diagnostics_problem2.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------
# Problem 3: Errors don't follow a bell curve (non-normal residuals)
# Fix: log-transform a right-skewed y to pull in the long tail.
# ----------------------------------------------------------------------
n = 200
x = np.linspace(1, 10, n)
y = np.exp(0.3 * x + np.random.normal(0, 0.5, n))  # right-skewed outcome

_, resid_bad = residuals(x, y)
_, resid_good = residuals(x, np.log(y))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
stats.probplot(resid_bad, dist="norm", plot=axes[0])
axes[0].get_lines()[0].set_color(BAD)
axes[0].get_lines()[0].set_markerfacecolor(BAD)
axes[0].set_title("Problem: raw y\n(Q-Q points curve away from the line)")
stats.probplot(resid_good, dist="norm", plot=axes[1])
axes[1].get_lines()[0].set_color(GOOD)
axes[1].get_lines()[0].set_markerfacecolor(GOOD)
axes[1].set_title("Fixed: modelled log(y)\n(Q-Q points hug the line)")
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle("Problem 3 — Non-normal errors", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "model-diagnostics_problem3.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------
# Problem 4: Troublemaker points with too much influence
# Fix: investigate, then refit without the erroneous influential point.
# ----------------------------------------------------------------------
n = 60
x = np.random.uniform(2, 8, n)
y = 1.5 * x + np.random.normal(0, 1, n)
# inject one high-leverage, high-influence point (data-entry error)
x_bad = np.append(x, 20.0)
y_bad = np.append(y, 8.0)

grid = np.linspace(0, 21, 100)
m_all, b_all = np.polyfit(x_bad, y_bad, 1)
m_clean, b_clean = np.polyfit(x, y, 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(x, y, alpha=0.7, color="#34495e", label="Normal points")
axes[0].scatter(
    [20.0], [8.0], color=BAD, s=120, zorder=5, edgecolor="k", label="Influential point"
)
axes[0].plot(
    grid,
    m_all * grid + b_all,
    color=BAD,
    linestyle="--",
    label="Fitted line (all points)",
)
axes[0].set_title("Problem: one point drags the line\n(high leverage + high influence)")
axes[0].legend(fontsize=9)
axes[1].scatter(x, y, alpha=0.7, color="#34495e", label="Normal points")
axes[1].plot(
    grid, m_clean * grid + b_clean, color=GOOD, label="Fitted line (point removed)"
)
axes[1].set_title(
    "Fixed: confirmed data-entry error, refit\n(line follows the real pattern)"
)
axes[1].legend(fontsize=9)
for ax in axes:
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
fig.suptitle("Problem 4 — Influential points", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "model-diagnostics_problem4.png", dpi=200, bbox_inches="tight")
plt.close()

print("Generated common-problems before/after images:")
for i in range(1, 5):
    print(f"{i}. assets/model-diagnostics_problem{i}.png")
