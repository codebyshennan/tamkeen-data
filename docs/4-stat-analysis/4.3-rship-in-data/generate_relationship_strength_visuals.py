import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create assets directory if it doesn't exist
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate a strong relationship example
n_samples = 100
x_strong = np.random.uniform(0, 10, n_samples)
# Strong relationship (correlation around 0.9)
y_strong = 2 * x_strong + np.random.normal(0, 1, n_samples)

# Create plot for strong relationship
plt.figure(figsize=(8, 6))
plt.scatter(x_strong, y_strong, alpha=0.7)
plt.title("Strong Relationship Example", fontsize=14)
plt.xlabel("X Variable", fontsize=12)
plt.ylabel("Y Variable", fontsize=12)

# Add correlation coefficient
corr_strong = np.corrcoef(x_strong, y_strong)[0, 1]
plt.text(0.05, 0.95, f"Correlation: {corr_strong:.2f}", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# Add trend line
m, b = np.polyfit(x_strong, y_strong, 1)
plt.plot(x_strong, m*x_strong + b, 'r--', alpha=0.7)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "understanding-relationships_strong.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Generate a weak relationship example
x_weak = np.random.uniform(0, 10, n_samples)
# Weak relationship (correlation around 0.3)
y_weak = 0.5 * x_weak + np.random.normal(0, 3, n_samples)

# Create plot for weak relationship
plt.figure(figsize=(8, 6))
plt.scatter(x_weak, y_weak, alpha=0.7)
plt.title("Weak Relationship Example", fontsize=14)
plt.xlabel("X Variable", fontsize=12)
plt.ylabel("Y Variable", fontsize=12)

# Add correlation coefficient
corr_weak = np.corrcoef(x_weak, y_weak)[0, 1]
plt.text(0.05, 0.95, f"Correlation: {corr_weak:.2f}", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# Add trend line
m, b = np.polyfit(x_weak, y_weak, 1)
plt.plot(x_weak, m*x_weak + b, 'r--', alpha=0.7)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "understanding-relationships_weak.png", dpi=300, bbox_inches='tight')
plt.close()

print("Generated relationship strength visualization images:")
print("1. Strong relationship: assets/understanding-relationships_strong.png")
print("2. Weak relationship: assets/understanding-relationships_weak.png")
