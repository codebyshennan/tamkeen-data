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

# Generate strong relationship data
# Height and weight example - strong relationship
heights = np.random.normal(170, 10, 100)  # Heights in cm
weights = heights * 0.5 + np.random.normal(10, 5, 100)  # Weights with some noise

# Generate weak relationship data
# Sleep and test scores example - weak relationship
sleep_hours = np.random.normal(7, 1.5, 100)  # Sleep hours
test_scores = sleep_hours * 2 + np.random.normal(70, 15, 100)  # Test scores with lots of noise

# Create strong relationship plot
plt.figure(figsize=(10, 6))
plt.scatter(heights, weights, alpha=0.7, color='blue')
plt.title('Strong Relationship: Height and Weight', fontsize=14)
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Weight (kg)', fontsize=12)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "understanding-relationships_strong.png", dpi=300, bbox_inches='tight')
plt.close()

# Create weak relationship plot
plt.figure(figsize=(10, 6))
plt.scatter(sleep_hours, test_scores, alpha=0.7, color='green')
plt.title('Weak Relationship: Sleep Hours and Test Scores', fontsize=14)
plt.xlabel('Sleep Hours', fontsize=12)
plt.ylabel('Test Score', fontsize=12)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "understanding-relationships_weak.png", dpi=300, bbox_inches='tight')
plt.close()

print("Generated relationship example images:")
print("1. Strong relationship: assets/understanding-relationships_strong.png")
print("2. Weak relationship: assets/understanding-relationships_weak.png")
