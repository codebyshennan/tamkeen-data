# Module 5: Machine Learning Fundamentals Assignment - Answer Key

## Part A: Concept Check (Multiple Choice)

### Section 1: Machine Learning Fundamentals

1. Answer: a. Algorithms that learn from data
   **Explanation**: Machine learning enables systems to learn patterns from data without explicit programming. Key aspects:
   - Data-driven approach vs traditional rule-based programming
   - Pattern recognition and generalization
   - Automatic improvement with experience
   - Minimal human intervention in the learning process

2. Answer: a. Creating new features from existing data
   **Explanation**: Feature engineering transforms raw data into meaningful features:
   - Uses domain knowledge to create new features
   - Improves model performance
   - Examples: polynomial features, binning, encoding
   - Critical for model success

3. Answer: a. Balance between model complexity and generalization
   **Explanation**: The bias-variance tradeoff represents:
   - Bias: Error from model assumptions (underfitting)
   - Variance: Sensitivity to training data (overfitting)
   - Trade-off between simple and complex models
   - Goal: Find optimal complexity point

4. Answer: a. Assessing model performance on unseen data
   **Explanation**: Cross-validation:
   - Provides robust performance estimates
   - Helps detect overfitting
   - Uses all data efficiently
   - Standard practice in model evaluation

5. Answer: a. Supervised uses labeled data, unsupervised doesn't
   **Explanation**: Key differences:
   - Supervised: Has target variable, makes predictions
   - Unsupervised: No labels, finds patterns
   - Different applications and evaluation methods
   - Complementary approaches in ML

### Section 2: Supervised Learning

6. Answer: a. Bayes' theorem with independence assumption
   **Explanation**: Naive Bayes:
   - Uses Bayes' theorem P(y|X) ∝ P(X|y)P(y)
   - Assumes feature independence
   - Efficient for high-dimensional data
   - Popular in text classification

7. Answer: a. Similar instances belong to same class
   **Explanation**: k-Nearest Neighbors:
   - Instance-based learning
   - Uses distance metrics
   - Non-parametric method
   - Local decision making

8. Answer: a. Maximum margin hyperplane
   **Explanation**: Support Vector Machines:
   - Finds optimal separating hyperplane
   - Maximizes margin between classes
   - Uses support vectors
   - Kernel trick for non-linearity

9. Answer: a. Ensemble of decision trees
   **Explanation**: Random Forest:
   - Combines multiple decision trees
   - Uses bootstrap sampling
   - Random feature selection
   - Reduces overfitting

10. Answer: a. Regularization technique
    **Explanation**: Dropout:
    - Randomly deactivates neurons
    - Prevents co-adaptation
    - Reduces overfitting
    - Acts as model averaging

### Section 3: Unsupervised Learning

11. Answer: a. Dimensionality reduction
    **Explanation**: Principal Component Analysis:
    - Linear dimensionality reduction
    - Preserves maximum variance
    - Creates orthogonal components
    - Useful for visualization

12. Answer: a. Visualization of high-dimensional data
    **Explanation**: t-SNE:
    - Non-linear dimensionality reduction
    - Preserves local structure
    - Better for visualization
    - Handles complex patterns

13. Answer: a. Partitioning data into k groups
    **Explanation**: k-means clustering:
    - Iterative algorithm
    - Minimizes within-cluster variance
    - Requires number of clusters (k)
    - Centroid-based method

14. Answer: a. Creating tree of nested clusters
    **Explanation**: Hierarchical clustering:
    - Creates cluster hierarchy
    - No predefined cluster number
    - Dendrogram visualization
    - Multiple clustering levels

15. Answer: a. Density-based clustering
    **Explanation**: DBSCAN:
    - Finds dense regions
    - Handles noise points
    - No predefined clusters
    - Shape-flexible clustering

## Part B: Implementation Tasks

### Task 1: Text Classification with Naive Bayes

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Smoothing parameter
        
    def fit(self, X, y):
        # Convert text to word count matrix
        self.vectorizer = CountVectorizer()
        X_counts = self.vectorizer.fit_transform(X)
        
        # Calculate prior probabilities
        self.classes = np.unique(y)
        self.class_priors = {}
        for c in self.classes:
            self.class_priors[c] = np.mean(y == c)
        
        # Calculate word probabilities for each class
        self.word_probs = {}
        for c in self.classes:
            # Get documents for this class
            docs_in_class = X_counts[y == c]
            # Calculate word frequencies with smoothing
            word_counts = np.array(docs_in_class.sum(axis=0))[0] + self.alpha
            total_words = word_counts.sum()
            self.word_probs[c] = word_counts / total_words
            
    def predict(self, X):
        X_counts = self.vectorizer.transform(X)
        predictions = []
        
        for doc in X_counts:
            # Calculate log probabilities for each class
            log_probs = {}
            for c in self.classes:
                # Start with log prior
                log_prob = np.log(self.class_priors[c])
                # Add log likelihood for each word
                word_counts = np.array(doc.todense())[0]
                log_prob += np.sum(word_counts * np.log(self.word_probs[c]))
                log_probs[c] = log_prob
            
            # Select class with highest probability
            predictions.append(max(log_probs.items(), key=lambda x: x[1])[0])
            
        return np.array(predictions)

# Example usage
texts = ["spam email buy now", "hello how are you", "buy discount now", "meeting tomorrow morning"]
labels = ["spam", "ham", "spam", "ham"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Train classifier
nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)

# Make predictions
predictions = nb.predict(X_test)
```

### Task 2: Customer Segmentation

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

def customer_segmentation(data):
    # Preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # PCA
    pca = PCA(n_components=0.95)  # Keep 95% variance
    pca_data = pca.fit_transform(scaled_data)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(pca_data)
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(pca_data)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                         c=clusters, cmap='viridis')
    plt.title('Customer Segments Visualization')
    plt.colorbar(scatter)
    plt.show()
    
    # Hierarchical clustering comparison
    hc = AgglomerativeClustering(n_clusters=4)
    hc_clusters = hc.fit_predict(pca_data)
    
    return clusters, hc_clusters, pca, tsne_data

# Analysis of segments
def analyze_segments(data, clusters):
    df = pd.DataFrame(data)
    df['Cluster'] = clusters
    
    # Segment statistics
    segment_stats = df.groupby('Cluster').agg(['mean', 'std'])
    
    # Visualization
    for col in df.columns[:-1]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=col, data=df)
        plt.title(f'{col} Distribution by Segment')
        plt.show()
    
    return segment_stats
```

### Task 3: Model Evaluation Framework

```python
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
    def plot_roc_curve(self, X_test, y_test):
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def plot_learning_curve(self, cv=5):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X, self.y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
```

### Task 4: Hyperparameter Optimization

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import optuna
import numpy as np
import matplotlib.pyplot as plt

class HyperparameterOptimizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def grid_search(self, param_grid):
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(self.X, self.y)
        
        return grid_search.best_params_, grid_search.best_score_
        
    def random_search(self, param_distributions, n_iter=100):
        rf = RandomForestClassifier()
        random_search = RandomizedSearchCV(rf, param_distributions,
                                         n_iter=n_iter, cv=5, n_jobs=-1)
        random_search.fit(self.X, self.y)
        
        return random_search.best_params_, random_search.best_score_
        
    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        
        rf = RandomForestClassifier(**params)
        return np.mean(cross_val_score(rf, self.X, self.y, cv=5))
        
    def bayesian_optimization(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
        
    def compare_methods(self):
        # Run all optimization methods
        grid_params, grid_score = self.grid_search(param_grid)
        random_params, random_score = self.random_search(param_dist)
        bayes_params, bayes_score = self.bayesian_optimization()
        
        # Plot comparison
        methods = ['Grid Search', 'Random Search', 'Bayesian Opt']
        scores = [grid_score, random_score, bayes_score]
        
        plt.figure(figsize=(10, 6))
        plt.bar(methods, scores)
        plt.title('Optimization Method Comparison')
        plt.ylabel('Best Score')
        plt.show()
        
        return {
            'grid_search': (grid_params, grid_score),
            'random_search': (random_params, random_score),
            'bayesian_opt': (bayes_params, bayes_score)
        }
```

## Implementation Notes

1. The Naive Bayes implementation includes:
   - Laplace smoothing for zero probabilities
   - Log probabilities for numerical stability
   - Vectorized operations for efficiency

2. Customer segmentation pipeline:
   - Standardization for equal feature scaling
   - PCA for dimension reduction
   - Multiple clustering techniques
   - Visualization for interpretation

3. Model evaluation framework provides:
   - Comprehensive performance metrics
   - Visual analysis tools
   - Cross-validation support
   - Interpretability features

4. Hyperparameter optimization:
   - Multiple optimization strategies
   - Efficient parameter space exploration
   - Comparative analysis
   - Visualization of results

## Common Pitfalls and Best Practices

1. Data Preprocessing:
   - Always scale features for distance-based algorithms
   - Handle missing values appropriately
   - Check for data leakage
   - Validate data quality

2. Model Selection:
   - Consider problem requirements
   - Balance complexity vs performance
   - Validate assumptions
   - Use appropriate metrics

3. Evaluation:
   - Use cross-validation
   - Consider statistical significance
   - Check for overfitting
   - Validate on holdout set

4. Implementation:
   - Optimize for efficiency
   - Include error handling
   - Document assumptions
   - Test edge cases
