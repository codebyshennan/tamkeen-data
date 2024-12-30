# Support Vector Machines (SVM) ⚔️

Support Vector Machines are powerful classifiers that find the optimal hyperplane to separate classes with the maximum margin. Think of it as finding the widest possible street between two neighborhoods - the wider the street, the more confident we are in our classification! 

## Mathematical Foundation 📐

### The Optimization Problem

For a binary classification problem, SVM solves:

$$\min_{w,b} \frac{1}{2}||w||^2$$
subject to:
$$y_i(w^Tx_i + b) \geq 1, \forall i$$

Where:
- $w$ is the normal vector to the hyperplane
- $b$ is the bias term
- $x_i$ are the training examples
- $y_i \in \{-1,1\}$ are the class labels

### The Margin
The margin width is given by:

$$\text{margin} = \frac{2}{||w||}$$

### Soft Margin SVM
For non-perfectly separable data:

$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^n \xi_i$$
subject to:
$$y_i(w^Tx_i + b) \geq 1 - \xi_i, \xi_i \geq 0, \forall i$$

Where:
- $\xi_i$ are slack variables
- $C$ is the regularization parameter

## The Kernel Trick 🎩

### Kernel Function
Maps data to higher dimensions:
$$K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$$

Common kernels:

1. Linear: $$K(x_i, x_j) = x_i^Tx_j$$

2. RBF (Gaussian): $$K(x_i, x_j) = \exp\left(-\gamma ||x_i - x_j||^2\right)$$

3. Polynomial: $$K(x_i, x_j) = (x_i^Tx_j + r)^d$$

## Implementation Examples 💻

### Linear SVM Classifier
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVMVisualizer:
    def __init__(self, kernel='linear'):
        self.scaler = StandardScaler()
        self.svm = SVC(kernel=kernel, random_state=42)
        
    def fit_and_visualize(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit SVM
        self.svm.fit(X_scaled, y)
        
        # Create mesh grid
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Get predictions for mesh grid
        Z = self.svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, alpha=0.8)
        
        # Plot support vectors
        plt.scatter(self.svm.support_vectors_[:, 0],
                   self.svm.support_vectors_[:, 1],
                   s=200, linewidth=1, facecolors='none',
                   edgecolors='k', label='Support Vectors')
        
        plt.title(f'SVM with {self.svm.kernel} kernel')
        plt.legend()
        plt.show()
        
        return self.svm

# Example usage
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

visualizer = SVMVisualizer()
svm_model = visualizer.fit_and_visualize(X, y)
```

### Non-linear Classification with RBF Kernel
```python
class NonLinearSVMExample:
    def __init__(self):
        self.scaler = StandardScaler()
        self.svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        
    def generate_nonlinear_data(self, n_samples=300):
        """Generate circular data"""
        np.random.seed(42)
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0]**2 + X[:, 1]**2 < 1.5).astype(int)
        return X, y
        
    def train_and_evaluate(self):
        # Generate and prepare data
        X, y = self.generate_nonlinear_data()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.svm.fit(X_scaled, y)
        
        # Visualize results
        self.plot_decision_boundary(X_scaled, y)
        
    def plot_decision_boundary(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = self.svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title('Non-linear SVM Classification')
        plt.show()

# Run example
nonlinear_example = NonLinearSVMExample()
nonlinear_example.train_and_evaluate()
```

## Real-World Applications 🌍

### 1. Text Classification
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class TextClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )),
            ('svm', SVC(kernel='linear', C=1.0))
        ])
        
    def train(self, texts, labels):
        """Train the text classifier"""
        self.pipeline.fit(texts, labels)
        
    def predict(self, texts):
        """Predict text categories"""
        return self.pipeline.predict(texts)
        
    def evaluate(self, texts, true_labels):
        """Evaluate classifier performance"""
        from sklearn.metrics import classification_report
        predictions = self.predict(texts)
        print(classification_report(true_labels, predictions))

# Example usage
texts = [
    "great product amazing service",
    "terrible experience bad quality",
    "excellent purchase recommend highly"
]
labels = [1, 0, 1]  # 1=positive, 0=negative

classifier = TextClassifier()
classifier.train(texts, labels)
```

### 2. Image Classification
```python
class ImageSVMClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.svm = SVC(kernel='rbf', C=10, gamma='scale')
        
    def preprocess_image(self, image):
        """Preprocess image for SVM"""
        # Flatten 2D image to 1D
        flattened = image.reshape(1, -1)
        # Scale pixel values
        return self.scaler.transform(flattened)
        
    def train(self, images, labels):
        """Train on image dataset"""
        # Flatten all images
        X = images.reshape(images.shape[0], -1)
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # Train SVM
        self.svm.fit(X_scaled, labels)
        
    def predict(self, image):
        """Predict image class"""
        processed = self.preprocess_image(image)
        return self.svm.predict(processed)
```

## Hyperparameter Optimization 🎛️

### Grid Search with Cross-Validation
```python
from sklearn.model_selection import GridSearchCV

class SVMOptimizer:
    def __init__(self):
        self.svm = SVC()
        self.param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly'],
            'degree': [2, 3, 4]  # for poly kernel
        }
        
    def optimize(self, X, y, cv=5):
        """Find optimal parameters"""
        grid_search = GridSearchCV(
            self.svm,
            self.param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y)
        
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)
        
        # Plot parameter comparison
        self.plot_parameter_comparison(grid_search)
        
        return grid_search.best_estimator_
        
    def plot_parameter_comparison(self, grid_search):
        """Visualize parameter effects"""
        results = pd.DataFrame(grid_search.cv_results_)
        
        plt.figure(figsize=(15, 5))
        
        # Plot C parameter effect
        plt.subplot(131)
        C_scores = results[results.param_kernel == 'rbf'].groupby('param_C').mean()
        plt.semilogx(C_scores.index, C_scores.mean_test_score)
        plt.xlabel('C parameter')
        plt.ylabel('Score')
        
        # Plot gamma parameter effect
        plt.subplot(132)
        gamma_scores = results[
            (results.param_kernel == 'rbf') & 
            (results.param_gamma != 'scale') & 
            (results.param_gamma != 'auto')
        ].groupby('param_gamma').mean()
        plt.semilogx(gamma_scores.index, gamma_scores.mean_test_score)
        plt.xlabel('gamma parameter')
        
        plt.tight_layout()
        plt.show()
```

## Best Practices and Optimization 💡

### 1. Feature Scaling
```python
def scale_features(X_train, X_test):
    """Scale features properly"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
```

### 2. Handling Imbalanced Data
```python
def handle_imbalance(X, y):
    """Handle imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    
    # Create weighted SVM
    svm = SVC(class_weight='balanced')
    
    return svm
```

### 3. Memory Optimization
```python
def optimize_memory_usage(X, y):
    """Handle large datasets"""
    from sklearn.svm import LinearSVC
    
    # Use LinearSVC for better memory efficiency
    svm = LinearSVC(dual=False)
    
    # Or use SGDClassifier for very large datasets
    from sklearn.linear_model import SGDClassifier
    svm_sgd = SGDClassifier(loss='hinge')  # hinge loss = linear SVM
    
    return svm, svm_sgd
```

## Common Pitfalls and Solutions ⚠️

1. **High Dimensionality**
   - Use dimensionality reduction
   - Feature selection
   - Linear kernel for high dimensions

2. **Large Datasets**
   - Use LinearSVC instead of SVC
   - Consider SGDClassifier
   - Sample data for parameter tuning

3. **Poor Performance**
   - Try different kernels
   - Scale features properly
   - Tune hyperparameters systematically

## Next Steps 📚

Now that you understand Support Vector Machines, let's explore [Decision Trees](./decision-trees.md) to learn about hierarchical decision making!
