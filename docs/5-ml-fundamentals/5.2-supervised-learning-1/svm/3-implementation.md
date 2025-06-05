# Implementing SVM with Scikit-learn

## Learning Objectives

By the end of this section, you will be able to:

- Implement SVM for classification and regression
- Preprocess data for SVM
- Tune SVM parameters
- Handle common challenges in SVM implementation

## Getting Started with SVM

### Basic Setup

First, let's import the necessary libraries:

```python
# Essential imports
from sklearn.svm import SVC, SVR  # For classification and regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
```

### Why These Libraries?

- `sklearn.svm`: Provides SVM implementations (SVC for classification, SVR for regression)
- `sklearn.preprocessing`: For data scaling (essential for SVM)
- `sklearn.model_selection`: For data splitting and validation
- `numpy`: For numerical operations and array handling
- `matplotlib`: For visualization of data and decision boundaries

## Basic Classification Example

Let's implement a complete binary classification example:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a simple binary classification dataset
X = np.array([
    [1, 2], [2, 3], [3, 4], [2, 1], [1, 3], [2, 2],  # Class 0
    [5, 6], [6, 7], [7, 8], [6, 5], [5, 7], [7, 6]   # Class 1
])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25,  # 25% for testing
    random_state=42  # For reproducibility
)

# Step 1: Scale the features (important for SVM)
scaler = StandardScaler()
scaler.fit(X_train)  # Compute mean and std on training data
X_train_scaled = scaler.transform(X_train)  # Apply to training data
X_test_scaled = scaler.transform(X_test)    # Apply to test data

# Step 2: Create and train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Step 3: Make predictions
predictions = svm_model.predict(X_test_scaled)

# Step 4: Evaluate accuracy
accuracy = svm_model.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Visualize the decision boundary
def plot_decision_boundary(X, y, model, scaler):
    # Create mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Scale mesh grid points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    
    # Get predictions
    Z = model.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=70, edgecolors='k')
    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Uncomment the line below to visualize the decision boundary
# plot_decision_boundary(X, y, svm_model, scaler)
```

**Explanation:**
1. **Data Creation**: We create a simple 2D dataset with two clearly separated classes
2. **Data Splitting**: We divide the data into training (75%) and testing (25%) sets
3. **Feature Scaling**: This is crucial for SVM as it's sensitive to the scale of input features
4. **Model Training**: We use the SVC classifier with an RBF kernel, which works well for many problems
5. **Evaluation**: We check model accuracy on the test set
6. **Visualization**: The included function can visualize the decision boundary to help understand how SVM separates the classes

Note that scaling is performed separately on the training and testing data to prevent data leakage (the test set shouldn't influence the scaling parameters).

## Multiclass Classification

SVM naturally extends to multiple classes. Let's implement a complete example using the Iris dataset:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25,
    random_state=42
)

# Scale features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the multiclass SVM model
svm_model = SVC(
    kernel='rbf',
    decision_function_shape='ovo',  # One-vs-one strategy for multiclass
    probability=True,  # Enable probability estimates
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Evaluate model
print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_names
))

# Visualize the results (in 2D using two of the features)
def plot_iris_decision_boundary(X, y, model, scaler, feature_idx=(0, 1)):
    # Select two features for visualization
    X_2d = X[:, feature_idx]
    
    # Create mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # For prediction, we need all 4 features, so use zeros for the non-visualized features
    if feature_idx == (0, 1):
        mesh_points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape), np.zeros(xx.ravel().shape)]
    elif feature_idx == (2, 3):
        mesh_points = np.c_[np.zeros(xx.ravel().shape), np.zeros(xx.ravel().shape), xx.ravel(), yy.ravel()]
    else:  # Mixed features
        zeros_array = np.zeros(xx.ravel().shape)
        mesh_points = np.zeros((xx.ravel().shape[0], 4))
        mesh_points[:, feature_idx[0]] = xx.ravel()
        mesh_points[:, feature_idx[1]] = yy.ravel()
    
    # Scale mesh points
    mesh_points_scaled = scaler.transform(mesh_points)
    
    # Get predictions
    Z = model.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=70, edgecolors='k')
    plt.title(f"SVM Decision Boundary using {feature_names[feature_idx[0]]} and {feature_names[feature_idx[1]]}")
    plt.xlabel(feature_names[feature_idx[0]])
    plt.ylabel(feature_names[feature_idx[1]])
    plt.show()

# Uncomment the line below to visualize the decision boundary using two features
# plot_iris_decision_boundary(X, y, svm_model, scaler, feature_idx=(0, 1))
```

**Explanation:**
1. **Dataset**: We use the famous Iris dataset which has 3 classes (setosa, versicolor, virginica) and 4 features
2. **Scaling**: Again, we scale the features which is particularly important for SVM
3. **Multiclass Strategy**: We use 'ovo' (one-vs-one) which builds a binary classifier for each pair of classes
4. **Evaluation**: The classification report shows precision, recall, and F1-score for each class
5. **Visualization**: The included function can visualize the decision boundaries, but since the data has 4 dimensions, we select 2 dimensions to visualize

This example demonstrates how SVM naturally handles multiclass problems, despite being fundamentally a binary classifier.

## Regression with SVM

SVM can also be used for regression tasks using Support Vector Regression (SVR):

```python
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a housing price dataset
# Features: [square_footage, bedrooms, age_years]
X = np.array([
    [1400, 3, 10],
    [1600, 3, 8],
    [1700, 4, 15],
    [1875, 4, 5],
    [1100, 2, 20],
    [1550, 3, 12],
    [2100, 4, 7],
    [1320, 3, 15],
    [1800, 4, 10],
    [1950, 3, 5],
    [1650, 3, 9],
    [1300, 2, 12]
])

# House prices (in thousands of dollars)
y = np.array([250, 280, 300, 350, 200, 270, 380, 230, 320, 345, 290, 220])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVR model
svr_model = SVR(
    kernel='rbf',
    C=100.0,  # Regularization parameter (higher value: less regularization)
    epsilon=10,  # Controls the width of the epsilon-tube (tolerance for errors)
    gamma='scale'  # Kernel coefficient
)
svr_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = svr_model.predict(X_train_scaled)
y_test_pred = svr_model.predict(X_test_scaled)

# Evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict price for a new house
new_house = np.array([[1500, 3, 12]])
new_house_scaled = scaler.transform(new_house)
prediction = svr_model.predict(new_house_scaled)
print(f"Predicted price for new house: ${prediction[0]:.2f}k")

# Visualize the results (simplifying to 1D for visualization)
def plot_svr_results():
    # Use just square footage (first feature) for visualization
    X_1d = X[:, 0].reshape(-1, 1)
    X_1d_scaled = scaler.fit_transform(X_1d)
    
    # Create SVR model for 1D data
    svr_1d = SVR(kernel='rbf', C=100, epsilon=10)
    svr_1d.fit(X_1d_scaled, y)
    
    # Create mesh for visualization
    X_grid = np.arange(X_1d.min(), X_1d.max(), 10).reshape(-1, 1)
    X_grid_scaled = scaler.transform(X_grid)
    y_grid = svr_1d.predict(X_grid_scaled)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_1d, y, color='darkorange', label='Data points')
    plt.plot(X_grid, y_grid, color='navy', label='SVR prediction')
    plt.xlabel('House Square Footage')
    plt.ylabel('Price ($k)')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

# Uncomment the line below to visualize the SVR results
# plot_svr_results()
```

**Explanation:**
1. **Data**: We create a dataset of house features (square footage, bedrooms, age) and their prices
2. **Scaling**: As with classification, feature scaling is crucial for SVR
3. **SVR Parameters**:
   - **C**: Controls the trade-off between model complexity and allowing errors
   - **epsilon**: Defines the width of the tube where errors are ignored
   - **gamma**: Defines the influence radius of each training example
4. **Evaluation**: We use Mean Squared Error (MSE) and R² to evaluate the regression quality
5. **Prediction**: We demonstrate how to predict the price of a new house
6. **Visualization**: The function shows how SVR creates a regression line (simplified to 1D)

SVR works by finding a function that deviates from the observed targets by at most epsilon while being as flat as possible.

## Parameter Tuning

Finding the optimal parameters is crucial for SVM performance. Here's how to use Grid Search:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scale features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid to search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['rbf', 'linear']
}

# Create the base model
base_model = SVC(random_state=42)

# Perform grid search
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on the test set
test_accuracy = best_model.score(X_test_scaled, y_test)
print("Test accuracy with best model: {:.3f}".format(test_accuracy))

# Compare with default model
default_model = SVC(random_state=42)
default_model.fit(X_train_scaled, y_train)
default_accuracy = default_model.score(X_test_scaled, y_test)
print("Test accuracy with default model: {:.3f}".format(default_accuracy))

# Visualize the results of different parameter combinations
def plot_param_performance():
    # Extract the results from grid search
    results = grid_search.cv_results_
    C_values = [0.1, 1, 10, 100]
    
    # Filter for RBF kernel results
    rbf_scores = [results['mean_test_score'][i] 
                 for i in range(len(results['params'])) 
                 if results['params'][i]['kernel'] == 'rbf']
    
    # Reshape to get a matrix of C vs gamma
    score_matrix = np.array(rbf_scores).reshape(4, 4)
    
    # Plot as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(score_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.colorbar(label='Accuracy')
    plt.xticks(np.arange(4), ['scale', 'auto', '0.1', '1'])
    plt.yticks(np.arange(4), ['0.1', '1', '10', '100'])
    plt.title('Grid Search Results: RBF Kernel')
    
    # Add the scores in the cells
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{score_matrix[i, j]:.3f}", 
                     ha="center", va="center", 
                     color="white" if score_matrix[i, j] > 0.8 else "black")
    
    plt.tight_layout()
    plt.show()

# Uncomment the line below to visualize parameter performance
# plot_param_performance()
```

**Explanation:**
1. **Grid Search**:
   - We create a grid of parameter combinations to try systematically
   - Each combination is evaluated using cross-validation
   - The best parameters are those that achieve the highest cross-validation score
   - We compare the optimized model against the default model to see the improvement

Cross-validation helps prevent overfitting by evaluating model performance on multiple data splits:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the SVM model
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    svm_model, X_scaled, y,
    cv=cv,
    scoring='accuracy'
)

# Print results
print("Cross-validation scores:", scores)
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Compare different C values using cross-validation
def compare_c_values():
    C_values = [0.01, 0.1, 1, 10, 100]
    mean_scores = []
    std_scores = []
    
    for C in C_values:
        model = SVC(kernel='rbf', C=C, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(C_values, mean_scores, yerr=std_scores, fmt='o-', capsize=5)
    plt.xscale('log')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('Cross-validation accuracy')
    plt.title('SVM Performance vs Regularization Strength')
    plt.grid(True)
    plt.show()

# Uncomment the line below to compare different C values
# compare_c_values()
```

**Explanation:**
1. **Cross-Validation**:
   - Instead of a single train-test split, we use multiple splits (folds)
   - This gives a more robust estimate of model performance
   - We can see the variation in performance across different data subsets
   - The visualization shows how the C parameter affects model performance

These techniques help prevent overfitting and ensure your model will generalize well to new data.

## Handling Common Challenges

### 1. Feature Scaling

Feature scaling is essential for SVM performance:

```python
from sklearn.preprocessing import StandardScaler

# Example usage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Explanation:**
- SVM is sensitive to the scale of input features
- Standardization transforms features to have zero mean and unit variance
- Always fit the scaler on training data only, then apply to test data

### 2. Imbalanced Data

When dealing with imbalanced classes, use class weights or SMOTE:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Create an imbalanced dataset
np.random.seed(42)
X_majority = np.random.randn(100, 2)
X_minority = np.random.randn(20, 2) + [2, 2]
X_imbalanced = np.vstack([X_majority, X_minority])
y_imbalanced = np.hstack([np.zeros(100), np.ones(20)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_imbalanced, y_imbalanced, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Using class_weight
model_weighted = SVC(class_weight='balanced', random_state=42)
model_weighted.fit(X_train_scaled, y_train)

# Method 2: Using SMOTE for resampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

model_smote = SVC(random_state=42)
model_smote.fit(X_resampled, y_resampled)

# Evaluate both models
print("Using class_weight='balanced':")
print(classification_report(y_test, model_weighted.predict(X_test_scaled)))

print("\nUsing SMOTE resampling:")
print(classification_report(y_test, model_smote.predict(X_test_scaled)))
```

**Explanation:**
1. **Class Weights**: Automatically adjusts weights inversely proportional to class frequencies
2. **SMOTE**: Creates synthetic examples of the minority class to balance the dataset
3. **Evaluation**: Classification report helps assess performance on imbalanced data by showing per-class metrics

### 3. Text Classification

For text data, combine SVM with TF-IDF vectorization:

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample text data
texts = [
    "I love this movie, it was great",
    "Terrible film, I hated it",
    "Amazing story and acting",
    "Worst movie ever, complete waste of time",
    "Brilliant director and excellent plot",
    "So boring and predictable"
]
labels = [1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.5, random_state=42
)

# Step 1: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1000,  # Limit vocabulary size
    min_df=2,           # Minimum document frequency
    stop_words='english'  # Remove common words
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 2: Train SVM model (linear kernel works best for text)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = svm_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Extract important features
def show_important_features(model, vectorizer, n=10):
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Get top positive and negative features
    top_positive_idx = np.argsort(coefficients)[-n:]
    top_negative_idx = np.argsort(coefficients)[:n]
    
    print("Top positive features:")
    for idx in reversed(top_positive_idx):
        print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
    
    print("\nTop negative features:")
    for idx in top_negative_idx:
        print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")

# Uncomment to see important features
# show_important_features(svm_model, vectorizer)
```

**Explanation:**
1. **TF-IDF Vectorization**: Converts text to numerical features by considering term frequency and inverse document frequency
2. **Linear Kernel**: Best for high-dimensional sparse data like text
3. **Feature Importance**: Coefficients of the linear SVM indicate the importance of each word for classification

## Common Mistakes to Avoid

1. **Forgetting to Scale Features**

   ```python
   # Wrong
   model = SVC()
   model.fit(X_train, y_train)
   
   # Right
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   model = SVC()
   model.fit(X_train_scaled, y_train)
   ```

2. **Ignoring Class Imbalance**

   ```python
   # Wrong
   model = SVC()
   
   # Right
   model = SVC(class_weight='balanced')
   ```

3. **Using Wrong Kernel**

   ```python
   # Wrong for text data
   model = SVC(kernel='rbf')
   
   # Right for text data
   model = SVC(kernel='linear')
   ```

## Next Steps

1. [Advanced Techniques](4-advanced.md) - Learn optimization techniques
2. [Applications](5-applications.md) - See real-world examples

Remember: Start with simple implementations and gradually add complexity!

## Handling Imbalanced Data

When dealing with imbalanced datasets, using class weights can significantly improve model performance:

![Class Weights Comparison](assets/class_weights_comparison.png)

*Figure: Effect of class weights on decision boundary. Notice how balanced weights help prevent bias towards the majority class.*
