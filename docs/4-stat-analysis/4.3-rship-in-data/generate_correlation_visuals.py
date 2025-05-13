import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Create assets directory if it doesn't exist
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Correlation Scale
plt.figure(figsize=(12, 3))
scale = np.linspace(-1, 1, 100)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.scatter(scale, np.zeros_like(scale), c=scale, cmap='coolwarm', s=100, alpha=0.7)

# Add labels
plt.text(-1, 0.05, "Perfect\nNegative\n(-1)", ha='center', fontsize=12)
plt.text(-0.5, 0.05, "Strong\nNegative", ha='center', fontsize=12)
plt.text(0, 0.05, "No\nCorrelation\n(0)", ha='center', fontsize=12)
plt.text(0.5, 0.05, "Strong\nPositive", ha='center', fontsize=12)
plt.text(1, 0.05, "Perfect\nPositive\n(+1)", ha='center', fontsize=12)

# Remove axes ticks
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.title('Correlation Coefficient Scale', fontsize=14)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "correlation-scale.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation Examples at different strengths
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
titles = ["Perfect Negative\nr = -1.0", "Moderate Negative\nr = -0.5", 
          "No Correlation\nr = 0.0", "Moderate Positive\nr = 0.5", 
          "Perfect Positive\nr = 1.0"]
correlations = [-1.0, -0.5, 0.0, 0.5, 1.0]

for i, (ax, title, corr) in enumerate(zip(axes, titles, correlations)):
    # Generate correlated data
    x = np.random.normal(0, 1, 50)
    if corr == 0:
        y = np.random.normal(0, 1, 50)  # Completely random for zero correlation
    else:
        # Generate correlated data
        y = corr * x + np.random.normal(0, np.sqrt(1 - corr**2), 50)
    
    # Plot
    ax.scatter(x, y, alpha=0.7)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(ASSETS_DIR / "correlation-examples.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Outlier Effect
np.random.seed(123)
x = np.random.normal(0, 1, 20)
y = 0.7 * x + np.random.normal(0, 0.5, 20)

# Add one outlier
x_outlier = np.append(x, [3])
y_outlier = np.append(y, [-3])

# Calculate correlations
corr_no_outlier = np.corrcoef(x, y)[0,1]
corr_with_outlier = np.corrcoef(x_outlier, y_outlier)[0,1]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Without outlier
axes[0].scatter(x, y, alpha=0.7, s=50)
m, b = np.polyfit(x, y, 1)
axes[0].plot(x, m*x + b, color='red')
axes[0].set_title(f'Without Outlier\nCorrelation: {corr_no_outlier:.2f}')
axes[0].set_xlim(-3.5, 3.5)
axes[0].set_ylim(-3.5, 3.5)

# With outlier
axes[1].scatter(x_outlier, y_outlier, alpha=0.7, s=50)
axes[1].scatter([x_outlier[-1]], [y_outlier[-1]], color='red', s=100, label='Outlier')
m, b = np.polyfit(x_outlier, y_outlier, 1)
axes[1].plot(x_outlier, m*x_outlier + b, color='red')
axes[1].set_title(f'With Outlier\nCorrelation: {corr_with_outlier:.2f}')
axes[1].set_xlim(-3.5, 3.5)
axes[1].set_ylim(-3.5, 3.5)
axes[1].legend()

plt.tight_layout()
plt.savefig(ASSETS_DIR / "outlier-effect.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Non-linear Relationship
x = np.linspace(-3, 3, 100)
y = x**2 + np.random.normal(0, 0.5, 100)  # Quadratic relationship

# Calculate Pearson correlation
corr = np.corrcoef(x, y)[0, 1]

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7)
plt.title(f'Non-linear Relationship\nPearson Correlation: {corr:.2f}')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.text(0.05, 0.95, "Note: Low correlation coefficient\ndespite clear relationship", 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.tight_layout()
plt.savefig(ASSETS_DIR / "non-linear-relationship.png", dpi=300, bbox_inches='tight')
plt.close()

print("Generated correlation visualization images:")
print("1. Correlation scale: assets/correlation-scale.png")
print("2. Correlation examples: assets/correlation-examples.png")
print("3. Outlier effect: assets/outlier-effect.png")
print("4. Non-linear relationship: assets/non-linear-relationship.png")
