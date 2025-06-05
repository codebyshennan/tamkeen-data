# Building Your First Decision Tree

## Getting Started with Scikit-learn

Scikit-learn is like a toolbox for machine learning. It provides ready-to-use implementations of many algorithms, including decision trees. Let's learn how to use it!

### Installation

First, make sure you have scikit-learn installed:

```bash
pip install scikit-learn
```

## Your First Decision Tree: Disease Diagnosis

Let's build a simple system that helps diagnose whether someone might be sick based on their symptoms.

### Step 1: Prepare the Data

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create sample data
# Each row represents a patient
# Columns: [temperature, cough, fatigue]
# Values: 0 = No, 1 = Yes
X = np.array([
    [101, 1, 1],  # Patient 1: High temp, cough, fatigue
    [99, 0, 0],   # Patient 2: Normal temp, no cough, no fatigue
    [102, 1, 1],  # Patient 3: High temp, cough, fatigue
    [98, 0, 1],   # Patient 4: Normal temp, no cough, fatigue
    [100, 1, 0]   # Patient 5: Slightly high temp, cough, no fatigue
])

# Labels: 'sick' or 'healthy'
y = ['sick', 'healthy', 'sick', 'healthy', 'healthy']
```

This code sets up our sample patient data with three features: body temperature, presence of cough, and fatigue level. We also create corresponding labels indicating whether each patient is sick or healthy.

### Step 2: Create and Train the Model

```python
# Create the model with specific settings
clf = DecisionTreeClassifier(
    max_depth=3,          # Don't let the tree get too deep
    min_samples_split=2,  # Need at least 2 samples to split
    min_samples_leaf=1    # Each leaf needs at least 1 sample
)

# Train the model on our data
clf.fit(X, y)

# Visualize the tree to understand how it makes decisions
plt.figure(figsize=(15, 10))
plot_tree(
    clf,
    feature_names=['temperature', 'cough', 'fatigue'],
    class_names=['healthy', 'sick'],
    filled=True,    # Color the nodes
    rounded=True    # Make it look nice
)
plt.title('Disease Diagnosis Decision Tree')
plt.show()
```

In this step, we create a decision tree classifier with specific settings to control its complexity. We then train the model using our patient data and visualize the resulting tree to understand how it makes decisions. The visualization shows which features (temperature, cough, fatigue) the tree uses to classify patients.

### Step 3: Make Predictions

```python
# New patient data
new_patient = np.array([[100, 1, 1]])  # Temperature: 100, Cough: Yes, Fatigue: Yes

# Make prediction
prediction = clf.predict(new_patient)
print(f"Diagnosis: {prediction[0]}")

# Get prediction probabilities
probabilities = clf.predict_proba(new_patient)
print(f"Confidence: {max(probabilities[0]) * 100:.1f}%")
```

Here we use our trained model to diagnose a new patient. We input their symptoms (temperature, cough, fatigue) and the model returns a prediction. We also calculate the confidence level of this prediction.

## Understanding the Tree Visualization

The tree visualization shows:

1. **Questions** at each node (e.g., "temperature <= 100.5")
2. **Gini impurity** (how mixed the groups are)
3. **Samples** in each node (how many patients)
4. **Class distribution** (how many healthy vs sick)

This visual representation helps us understand exactly how the model makes decisions based on the input features.

## Iris Flower Classification Example

Let's try another example with the famous Iris dataset, which is built into scikit-learn:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train the model
iris_clf = DecisionTreeClassifier(max_depth=3)
iris_clf.fit(X_train, y_train)

# Evaluate the model
accuracy = iris_clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.1f}%")

# Visualize the tree
plt.figure(figsize=(15, 10))
plot_tree(
    iris_clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True
)
plt.title('Iris Classification Tree')
plt.show()
```

This example demonstrates how to work with a real dataset. We:
1. Load the built-in Iris dataset with measurements of different Iris flowers
2. Split the data into training and testing sets
3. Train a decision tree classifier on the training data
4. Evaluate its accuracy on the test data
5. Visualize the resulting decision tree

## House Price Prediction Example

Now let's try a regression problem - predicting house prices:

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# Sample house data
# Each row: [size (sqft), bedrooms, age (years)]
X_houses = np.array([
    [1400, 3, 10],  # House 1
    [1600, 3, 8],   # House 2
    [1700, 4, 15],  # House 3
    [1875, 4, 5],   # House 4
    [1100, 2, 20],  # House 5
    [2000, 4, 2],   # House 6
    [1800, 3, 1],   # House 7
    [1250, 2, 12],  # House 8
    [1350, 3, 3],   # House 9
    [1500, 3, 7]    # House 10
])

# Prices in thousands of dollars
y_prices = np.array([250, 280, 300, 350, 200, 380, 340, 220, 260, 270])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_houses, y_prices, test_size=0.3, random_state=42
)

# Create and train the model
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X_train, y_train)

# Evaluate the model
train_score = regressor.score(X_train, y_train)
test_score = regressor.score(X_test, y_test)
print(f"Training R² Score: {train_score:.3f}")
print(f"Testing R² Score: {test_score:.3f}")

# Make a prediction for a new house
new_house = np.array([[1500, 3, 12]])  # 1500 sqft, 3 bedrooms, 12 years old
predicted_price = regressor.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:.2f}k")

# Visualize feature importance
feature_importance = regressor.feature_importances_
features = ['Size (sqft)', 'Bedrooms', 'Age (years)']

plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.title('Feature Importance for House Price Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

This example shows:
1. How to use decision trees for regression (predicting numeric values)
2. How to create and train a DecisionTreeRegressor
3. How to evaluate regression models using R² score
4. How to identify which features are most important for making predictions

## Visualizing Decision Boundaries

For a better understanding, let's create a simple 2D visualization of how decision trees create boundaries:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

# Create a simple dataset with two features
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple rule: x + y > 1

# Add some noise
noise = np.random.randint(0, 10, size=len(y))
y = np.where(noise == 0, 1 - y, y)  # Flip about 10% of labels

# Create and train the model
tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X, y)

# Create meshgrid for plotting decision boundary
h = 0.02  # Step size
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Make predictions on the meshgrid
Z = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.title('Decision Tree Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

This visualization shows:
1. How the decision tree divides the feature space into regions
2. How these regions form a "decision boundary" between different classes
3. The rectangular nature of decision tree boundaries (unlike curved boundaries in other algorithms)

## Common Mistakes to Avoid

### 1. Overfitting

```python
# Bad: Tree too deep - will memorize training data
deep_tree = DecisionTreeClassifier(max_depth=None)
deep_tree.fit(X_train, y_train)
print(f"Training score: {deep_tree.score(X_train, y_train):.3f}")
print(f"Testing score: {deep_tree.score(X_test, y_test):.3f}")

# Good: Reasonable depth - will generalize better
good_tree = DecisionTreeClassifier(max_depth=3)
good_tree.fit(X_train, y_train)
print(f"Training score: {good_tree.score(X_train, y_train):.3f}")
print(f"Testing score: {good_tree.score(X_test, y_test):.3f}")
```

Overfitting happens when your tree becomes too complex and starts memorizing the training data instead of learning general patterns. This is why we limit the tree depth and use other parameters to control complexity.

### 2. Ignoring Feature Scaling

Decision trees don't require feature scaling, which is a benefit compared to many other algorithms:

```python
from sklearn.preprocessing import StandardScaler

# Decision trees work fine without scaling
tree_no_scaling = DecisionTreeClassifier(max_depth=3)
tree_no_scaling.fit(X_train, y_train)
print(f"Without scaling: {tree_no_scaling.score(X_test, y_test):.3f}")

# Scaling doesn't hurt, but isn't necessary
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tree_with_scaling = DecisionTreeClassifier(max_depth=3)
tree_with_scaling.fit(X_train_scaled, y_train)
print(f"With scaling: {tree_with_scaling.score(X_test_scaled, y_test):.3f}")
```

This is a key advantage of decision trees - they don't require feature scaling because they make decisions based on greater than/less than comparisons, not distances between points.

## Practice Exercise

Try building your own decision tree:

1. Choose a dataset (Iris or Titanic are good starters)
2. Split the data into training and testing sets
3. Create and train a decision tree
4. Make predictions and evaluate the model
5. Visualize the tree and feature importance

## Next Steps

Ready to learn more? Check out:

1. [Advanced techniques](4-advanced.md) for improving your trees
2. [Real-world applications](5-applications.md) of decision trees
3. How to combine multiple trees into powerful ensembles
