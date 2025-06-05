# Advanced SVM Techniques

## Learning Objectives

By the end of this section, you will be able to:

- Implement advanced optimization techniques for SVM
- Create and use custom kernels
- Optimize SVM performance
- Handle large-scale SVM problems

## Advanced Optimization Techniques

### Sequential Minimal Optimization (SMO)

SMO is like breaking a big problem into smaller, manageable pieces. Here's why it's useful:

1. **Faster Training**
   - Works on small subsets of data at a time
   - More efficient than traditional methods
   - Better for large datasets

2. **Memory Efficient**
   - Doesn't need to store entire dataset
   - Works well with limited memory
   - Good for big data applications

### Regularization Parameter (C)

The C parameter controls the trade-off between having a wide margin and correctly classifying training points:

![C Parameter Comparison](assets/C_parameter_comparison.png)

*Figure: Effect of C parameter on decision boundary. Left: Low C (more regularization), Middle: Balanced C, Right: High C (less regularization).*

Here's a complete example showing the impact of different C values:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a synthetic dataset with some noise
np.random.seed(42)
X = np.random.randn(120, 2)
y = np.zeros(120)
# Make the first 100 points clustered around (0,0)
# Make the last 20 points form a line through the first cluster
X[100:, 0] = np.linspace(-2, 2, 20)
X[100:, 1] = np.linspace(-2, 2, 20) + 0.5
y[100:] = 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a function to plot decision boundaries for different C values
def plot_different_c_values(X, y, scaler):
    C_values = [0.1, 1.0, 100.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, C in enumerate(C_values):
        # Train SVM with current C value
        svm = SVC(kernel='rbf', C=C, gamma='scale')
        svm.fit(X_train_scaled, y_train)
        
        # Create mesh grid for plotting
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Scale the mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)
        
        # Get predictions and reshape
        Z = svm.predict(mesh_points_scaled)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        axes[i].contourf(xx, yy, Z, alpha=0.3)
        axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        axes[i].set_title(f'C = {C}')
        
        # Highlight support vectors
        axes[i].scatter(
            X[svm.support_], y[svm.support_],
            s=100, linewidth=1, facecolors='none',
            edgecolors='r', label='Support Vectors'
        )
        
        # Display accuracy
        accuracy = svm.score(X_test_scaled, y_test)
        axes[i].text(x_min + 0.5, y_min + 0.5, f'Accuracy: {accuracy:.2f}')
    
    plt.tight_layout()
    plt.suptitle('Effect of C Parameter on Decision Boundary', y=1.05, fontsize=16)
    plt.show()

# Uncomment to visualize the effect of C parameter
# plot_different_c_values(X, y, scaler)

# Implement early stopping for SVM training
def train_svm_with_early_stopping(X, y, max_iter=100, tolerance=1e-3):
    """
    Train SVM with early stopping based on convergence.
    
    Parameters:
    - X: Training features
    - y: Training labels
    - max_iter: Maximum number of iterations
    - tolerance: Convergence threshold
    
    Returns:
    - Trained SVM model
    - Number of iterations needed
    """
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    prev_score = 0
    iterations_needed = max_iter
    
    for i in range(1, max_iter + 1):
        # Create model with current max_iter
        model = SVC(
            kernel='rbf',
            cache_size=1000,  # Increase cache for faster training
            max_iter=i,
            random_state=42
        )
        
        # Train the model
        model.fit(X_scaled, y)
        
        # Calculate current score
        score = model.score(X_scaled, y)
        
        # Check for convergence
        if abs(score - prev_score) < tolerance and i > 5:
            print(f"Converged after {i} iterations with score {score:.4f}")
            iterations_needed = i
            break
            
        prev_score = score
    
    # Final model with optimal iterations
    final_model = SVC(
        kernel='rbf',
        cache_size=1000,
        max_iter=iterations_needed,
        random_state=42
    )
    final_model.fit(X_scaled, y)
    
    return final_model, iterations_needed

# Example of using early stopping
# model, iters = train_svm_with_early_stopping(X, y)
# print(f"Final accuracy: {model.score(scaler.transform(X_test), y_test):.4f}")
```

**Explanation:**
- This example demonstrates how different C values affect the decision boundary
- A low C value creates a smoother boundary but may misclassify some points
- A high C value tries to correctly classify all training points, which can lead to overfitting
- The early stopping implementation monitors model convergence to avoid unnecessary iterations
- We track the model's score and stop training when changes become smaller than a tolerance threshold

## Advanced Kernel Techniques

### Custom Kernel Implementation

Sometimes you need a special kernel for your specific problem. Here's a complete example with a custom kernel:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a more complex dataset where custom kernels can be useful
np.random.seed(42)
X1 = np.random.randn(100, 2)
X2 = np.random.randn(100, 2) * 0.3
X2[:, 0] = X2[:, 0] * np.cos(X2[:, 1] * 5) + 2
X2[:, 1] = X2[:, 1] * np.sin(X2[:, 0] * 5) + 2
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(100), np.ones(100)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class CustomKernelSVM:
    def __init__(self, C=1.0):
        """
        Initialize SVM with custom kernel.
        
        Parameters:
        - C: Regularization parameter
        """
        self.C = C
        self.model = SVC(kernel='precomputed', C=C)
        
    def hybrid_kernel(self, X, Y=None):
        """
        Create a custom kernel combining RBF and polynomial.
        
        Parameters:
        - X: First set of points
        - Y: Second set of points (optional)
        
        Returns:
        - Kernel matrix
        """
        if Y is None:
            Y = X
            
        # RBF component
        gamma = 0.1
        rbf = np.exp(-gamma * pairwise_kernels(X, Y, metric='euclidean')**2)
        
        # Polynomial component
        degree = 2
        poly = (np.dot(X, Y.T) + 1) ** degree
        
        # Combine kernels (weighted sum)
        return 0.7 * rbf + 0.3 * poly
        
    def fit(self, X, y):
        """Train model with custom kernel"""
        self.X_train = X.copy()  # Store training data
        K = self.hybrid_kernel(X)  # Compute kernel matrix
        self.model.fit(K, y)  # Train the model
        return self
        
    def predict(self, X):
        """Make predictions using custom kernel"""
        K = self.hybrid_kernel(X, self.X_train)  # Kernel between test and train
        return self.model.predict(K)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        return np.mean(self.predict(X) == y)

# Compare standard kernels with custom kernel
def compare_kernels():
    # Standard kernels
    rbf_svm = SVC(kernel='rbf', gamma=0.1)
    poly_svm = SVC(kernel='poly', degree=2, coef0=1)
    
    # Custom kernel
    custom_svm = CustomKernelSVM()
    
    # Train all models
    rbf_svm.fit(X_train_scaled, y_train)
    poly_svm.fit(X_train_scaled, y_train)
    custom_svm.fit(X_train_scaled, y_train)
    
    # Calculate scores
    rbf_score = rbf_svm.score(X_test_scaled, y_test)
    poly_score = poly_svm.score(X_test_scaled, y_test)
    custom_score = custom_svm.score(X_test_scaled, y_test)
    
    print(f"RBF Kernel Accuracy: {rbf_score:.4f}")
    print(f"Polynomial Kernel Accuracy: {poly_score:.4f}")
    print(f"Custom Hybrid Kernel Accuracy: {custom_score:.4f}")
    
    # Visualize decision boundaries
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['RBF Kernel', 'Polynomial Kernel', 'Custom Hybrid Kernel']
    models = [rbf_svm, poly_svm, custom_svm]
    
    for i, (title, model) in enumerate(zip(titles, models)):
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Get predictions
        if i < 2:  # Standard kernels
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_scaled = scaler.transform(mesh_points)
            Z = model.predict(mesh_scaled)
        else:  # Custom kernel
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_scaled = scaler.transform(mesh_points)
            Z = model.predict(mesh_scaled)
            
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        axes[i].contourf(xx, yy, Z, alpha=0.3)
        axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        axes[i].set_title(f'{title}\nAccuracy: {model.score(X_test_scaled, y_test):.4f}')
    
    plt.tight_layout()
    plt.show()

# Uncomment to compare different kernels
# compare_kernels()
```

**Explanation:**
- We implement a custom kernel that combines the strengths of RBF and polynomial kernels
- The hybrid kernel is a weighted sum: 70% RBF + 30% polynomial
- Custom kernels are useful when standard kernels don't capture the unique patterns in your data
- The SVC model with kernel='precomputed' allows us to provide a pre-computed kernel matrix
- We store the training data to compute the kernel between test and training data during prediction
- The visualization shows how different kernels create different decision boundaries

## Advanced Visualization

### Decision Boundary and Support Vectors Visualization

Visualizing decision boundaries helps understand how SVM works:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Create a more interesting dataset for visualization
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM model
svm_model = SVC(kernel='rbf', gamma=10, C=1.0)
svm_model.fit(X_scaled, y)

def visualize_svm_details(X, y, model, scaler):
    """
    Create a detailed visualization of SVM decision boundary,
    margins, and support vectors.
    
    Parameters:
    - X: Feature data
    - y: Labels
    - model: Trained SVM model
    - scaler: Fitted scaler for the data
    """
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    # Scale mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_scaled = scaler.transform(mesh_points)
    
    # Get predictions and decision function values
    Z = model.predict(mesh_scaled)
    Z = Z.reshape(xx.shape)
    
    # Get decision function values (distance from hyperplane)
    decision_values = model.decision_function(mesh_scaled)
    decision_values = decision_values.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.figure(figsize=(12, 8))
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, decision_values, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='w')
    
    # Highlight support vectors
    plt.scatter(
        X[model.support_, 0],
        X[model.support_, 1],
        s=200, linewidth=1, facecolors='none',
        edgecolors='r', label='Support Vectors'
    )
    
    # Add information about the model
    plt.title(f'SVM Decision Boundary (kernel={model.kernel}, C={model.C}, gamma={model._gamma})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Display number of support vectors
    support_vector_count = len(model.support_)
    total_points = len(X)
    sv_percentage = support_vector_count / total_points * 100
    
    plt.text(x_min + 0.5, y_min + 0.3, 
             f'Support Vectors: {support_vector_count}/{total_points} ({sv_percentage:.1f}%)',
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

# Uncomment to visualize SVM details
# visualize_svm_details(X, y, svm_model, scaler)
```

**Explanation:**
- This visualization shows not just the decision boundary but also the margins
- The solid line is the decision boundary (where decision function = 0)
- The dashed lines are the margins (where decision function = ±1)
- Support vectors are highlighted with red circles
- We display the percentage of points that are support vectors, which indicates model complexity
- A high percentage of support vectors can suggest the model is complex and might overfit

## Performance Optimization

### Memory-Efficient Implementation

For large datasets, memory efficiency is crucial:

```python
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Create a synthetic large dataset
X, y = make_classification(
    n_samples=10000,  # 10,000 samples
    n_features=20,    # 20 features
    n_informative=10, # 10 informative features
    random_state=42
)

def memory_efficient_svm(X, y, chunk_size=1000):
    """
    Train SVM in memory-efficient way by processing data in chunks.
    
    Parameters:
    - X: Training features
    - y: Training labels
    - chunk_size: Size of data chunks to process
    
    Returns:
    - Trained model and scaler
    """
    print(f"Dataset shape: {X.shape}")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Process data in chunks to fit the scaler
    print("Scaling data in chunks...")
    for i in range(0, len(X), chunk_size):
        end = min(i + chunk_size, len(X))
        print(f"  Processing chunk {i//chunk_size + 1}: samples {i} to {end-1}")
        chunk = X[i:end]
        # Update scaler incrementally
        if i == 0:
            scaler.fit(chunk)  # First fit
        else:
            # For demonstration - in practice, you'd use partial_fit
            # We'll approximate by re-fitting on each chunk
            scaler.fit(chunk)
    
    # Transform all data (in a real scenario with huge data,
    # you might transform chunks as needed)
    X_scaled = scaler.transform(X)
    
    # Train model with memory-efficient configuration
    print("Training memory-efficient LinearSVC...")
    model = LinearSVC(
        dual=False,  # More memory efficient for n_samples > n_features
        max_iter=1000,
        tol=1e-4
    )
    model.fit(X_scaled, y)
    
    # Report results
    accuracy = model.score(X_scaled, y)
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Number of iterations: {model.n_iter_}")
    
    return model, scaler

# Example usage
# model, scaler = memory_efficient_svm(X, y)

# Make predictions on new data
def predict_efficiently(model, scaler, new_data):
    """Make predictions on new data using trained model"""
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)

# Example prediction on a small sample of new data
# new_samples = X[9000:9010]  # Just for demonstration
# predictions = predict_efficiently(model, scaler, new_samples)
# print("Predictions:", predictions)
```

**Explanation:**
- This implementation processes data in chunks to reduce memory usage
- For very large datasets, we can scale features incrementally without loading everything at once
- The LinearSVC is used with dual=False which is more memory-efficient when n_samples > n_features
- In real applications with truly huge datasets, you'd implement the transform step in chunks too
- This approach can handle datasets too large to fit in memory all at once

### Parallel Processing for Parameter Tuning

Speed up training with parallel processing:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from joblib import Parallel, delayed
from itertools import product
import matplotlib.pyplot as plt
import time

# Create a dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    random_state=42
)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def parallel_parameter_search(X, y, n_jobs=-1):
    """
    Perform parallel parameter search for SVM.
    
    Parameters:
    - X: Training features
    - y: Training labels
    - n_jobs: Number of parallel jobs (-1 for all cores)
    
    Returns:
    - Best parameters and their score
    """
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    # Function to evaluate a single parameter combination
    def evaluate_params(params):
        """Evaluate single parameter set using cross-validation"""
        start_time = time.time()
        model = SVC(**params)
        scores = cross_val_score(
            model, X, y,
            cv=5, n_jobs=1,  # Use 1 job here as we parallelize at a higher level
            scoring='accuracy'
        )
        mean_score = scores.mean()
        duration = time.time() - start_time
        return params, mean_score, duration
    
    # Generate all parameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), values))
        for values in product(*param_grid.values())
    ]
    
    print(f"Evaluating {len(param_combinations)} parameter combinations in parallel...")
    
    # Run parameter evaluation in parallel
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_params)(params)
        for params in param_combinations
    )
    total_time = time.time() - start_time
    
    # Find best parameters
    best_idx = np.argmax([score for _, score, _ in results])
    best_params, best_score, _ = results[best_idx]
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    print(f"Total search time: {total_time:.2f} seconds")
    
    # Visualize results
    def plot_results():
        # Extract data for visualization
        scores = np.array([score for _, score, _ in results])
        times = np.array([time for _, _, time in results])
        
        # Create parameter description strings
        param_strings = [
            f"C={p['C']}, gamma={p['gamma']}, kernel={p['kernel']}" 
            for p, _, _ in results
        ]
        
        # Plot scores for each parameter combination
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.barh(range(len(scores)), scores)
        plt.yticks(range(len(scores)), [s[:20] for s in param_strings], fontsize=8)
        plt.xlabel('Cross-validation Score')
        plt.title('Parameter Performance Comparison')
        plt.grid(axis='x')
        
        # Plot evaluation time
        plt.subplot(1, 2, 2)
        plt.barh(range(len(times)), times)
        plt.yticks(range(len(times)), [s[:20] for s in param_strings], fontsize=8)
        plt.xlabel('Evaluation Time (seconds)')
        plt.title('Parameter Evaluation Time')
        plt.grid(axis='x')
        
        plt.tight_layout()
        plt.show()
    
    # Uncomment to plot results
    # plot_results()
    
    return best_params, best_score

# Example usage
# best_params, best_score = parallel_parameter_search(X_scaled, y)

# Train final model with best parameters
def train_final_model(X, y, best_params):
    """Train final model with best parameters"""
    model = SVC(**best_params)
    model.fit(X, y)
    return model

# Example of training final model
# final_model = train_final_model(X_scaled, y, best_params)
# print(f"Final model accuracy: {final_model.score(X_scaled, y):.4f}")
```

**Explanation:**
- This implementation uses Parallel and delayed from joblib to run parameter evaluation in parallel
- Each parameter combination is evaluated independently using cross-validation
- The approach is much faster than sequential parameter search, especially with many combinations
- We keep track of evaluation time to identify which parameter combinations are more computationally expensive
- The visualization helps understand the trade-off between parameter performance and computational cost

## Advanced Feature Engineering

### Feature Selection with SVM

Select the most important features with SVM-based feature selection:

```python
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Load breast cancer dataset (30 features)
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"Original dataset shape: {X.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def select_features_with_svm(X_train, y_train, X_test, threshold='mean'):
    """
    Select important features using SVM weights.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - threshold: Feature importance threshold
    
    Returns:
    - Selected features for train and test
    - Feature selector
    - Feature importances
    """
    # Train linear SVM with L1 penalty
    lsvc = LinearSVC(
        C=0.01,               # Stronger regularization
        penalty='l1',         # L1 regularization for sparsity
        dual=False,           # L1 only works with dual=False
        max_iter=10000,
        tol=1e-4
    )
    lsvc.fit(X_train, y_train)
    
    # Get feature importances (absolute coefficient values)
    importances = np.abs(lsvc.coef_[0])
    
    # Create feature selector
    selector = SelectFromModel(
        lsvc,
        prefit=True,
        threshold=threshold
    )
    
    # Transform data
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Number of features selected: {X_train_selected.shape[1]} out of {X_train.shape[1]}")
    
    return X_train_selected, X_test_selected, selector, importances

# Select features
X_train_selected, X_test_selected, selector, importances = select_features_with_svm(
    X_train_scaled, y_train, X_test_scaled
)

def evaluate_feature_selection():
    """Evaluate the impact of feature selection"""
    # Train SVM on full feature set
    full_model = LinearSVC(max_iter=10000)
    full_model.fit(X_train_scaled, y_train)
    full_accuracy = full_model.score(X_test_scaled, y_test)
    
    # Train SVM on selected features
    selected_model = LinearSVC(max_iter=10000)
    selected_model.fit(X_train_selected, y_train)
    selected_accuracy = selected_model.score(X_test_selected, y_test)
    
    print(f"Accuracy with all features: {full_accuracy:.4f}")
    print(f"Accuracy with selected features: {selected_accuracy:.4f}")
    
    # Visualize feature importances
    plt.figure(figsize=(12, 6))
    
    # Sort features by importance
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=8)
    plt.xlabel('Feature Importance (absolute coefficient value)')
    plt.title('SVM Feature Importance')
    
    # Highlight selected features
    mask = selector.get_support()
    selected_indices = [i for i, selected in enumerate(mask) if selected]
    
    for i, idx in enumerate(indices):
        if idx in selected_indices:
            plt.barh(i, importances[idx], color='red')
    
    plt.tight_layout()
    plt.show()

# Uncomment to evaluate feature selection
# evaluate_feature_selection()
```

**Explanation:**
- We use LinearSVC with L1 regularization to encourage sparsity (many coefficients become zero)
- The SelectFromModel transformer keeps only features with importance above a threshold
- By default, the 'mean' threshold keeps features with importance above the mean importance
