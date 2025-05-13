import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from pathlib import Path
import statsmodels.api as sm

# Create assets directory if it doesn't exist
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data for student exam scores
n_samples = 100

# Create factors that might affect exam scores
study_hours = np.random.normal(0, 1, n_samples)
prev_gpa = np.random.normal(0, 1, n_samples)
sleep_hours = np.random.normal(0, 1, n_samples)

# Create exam scores based on these factors
exam_scores = 2*study_hours + 3*prev_gpa + 1.5*sleep_hours + np.random.normal(0, 1, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'study_hours': study_hours,
    'prev_gpa': prev_gpa,
    'sleep_hours': sleep_hours,
    'exam_score': exam_scores
})

# Fit model
X = data[['study_hours', 'prev_gpa', 'sleep_hours']]
y = data['exam_score']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 1. Create predicted vs actual plot
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Multiple Regression: Predicted vs Actual Exam Scores')
plt.grid(True, alpha=0.3)

# Add correlation coefficient
corr = np.corrcoef(y, y_pred)[0, 1]
plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig(ASSETS_DIR / "multiple-regression-prediction.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Create partial regression plots (using statsmodels)
# Add constant for statsmodels
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()

# Create partial regression plots (also known as component+residual plots)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot for each predictor
for i, var in enumerate(X.columns):
    sm.graphics.plot_partregress(var, 'exam_score', list(set(X.columns) - {var}), 
                                data=data, ax=axes[i], alpha=0.7)
    axes[i].set_title(f'Partial Regression Plot for {var}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ASSETS_DIR / "partial-regression-plots.png", dpi=300, bbox_inches='tight')
plt.close()

print("Generated multiple linear regression visualization images:")
print("1. Multiple regression prediction: assets/multiple-regression-prediction.png")
print("2. Partial regression plots: assets/partial-regression-plots.png")
