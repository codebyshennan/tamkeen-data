import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Create assets directory if it doesn't exist
import os
os.makedirs('assets', exist_ok=True)

print("Generating Model Selection Visualizations and Outputs...")

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Linear Model
print("\n=== Linear Model ===")
linear_model = LogisticRegression(random_state=42)
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
linear_accuracy = accuracy_score(y_test, y_pred_linear)
print(f"Linear Model Accuracy: {linear_accuracy:.3f}")

# Visualize decision boundary
def plot_decision_boundary(model, X, y, filename):
    # Reduce to 2D for visualization
    X_2d = X[:, :2]
    model_2d = LogisticRegression(random_state=42)
    model_2d.fit(X_2d, y)
    
    # Create mesh grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Predict on mesh grid
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.8, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear Model Decision Boundary')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'assets/{filename}', bbox_inches='tight')
    plt.close()

plot_decision_boundary(linear_model, X, y, 'linear_decision_boundary.png')

# 2. Tree-Based Model
print("\n=== Tree-Based Model ===")
tree_model = RandomForestClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)
print(f"Tree Model Accuracy: {tree_accuracy:.3f}")

# Visualize feature importance
def plot_feature_importance(model, feature_names, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    bars = plt.bar(range(len(importances)), importances[indices], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xticks(range(len(importances)), 
               [f'Feature {i+1}' for i in indices], 
               rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'assets/{filename}', bbox_inches='tight')
    plt.close()

plot_feature_importance(tree_model, [f'Feature {i+1}' for i in range(X.shape[1])], 'feature_importance.png')

# 3. Neural Network
print("\n=== Neural Network ===")
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, y_pred_nn)
print(f"Neural Network Accuracy: {nn_accuracy:.3f}")

# Visualize learning curve
def plot_learning_curve(model, X, y, filename):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score', color='blue', linewidth=2)
    plt.plot(train_sizes, val_mean, 'o-', label='Cross-validation score', color='red', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve - Neural Network')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'assets/{filename}', bbox_inches='tight')
    plt.close()

plot_learning_curve(nn_model, X, y, 'learning_curve.png')

# 4. Model Comparison
print("\n=== Model Comparison ===")
def compare_models(models, X_train, X_test, y_train, y_test, filename):
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {results[name]:.3f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(list(results.keys()), list(results.values()), 
                   color=['lightblue', 'lightgreen', 'lightcoral'],
                   edgecolor='black', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f'assets/{filename}', bbox_inches='tight')
    plt.close()
    
    return results

# Compare models
models = {
    'Linear': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
}

results = compare_models(models, X_train, X_test, y_train, y_test, 'model_comparison.png')

# 5. Credit Risk Prediction Example
print("\n=== Credit Risk Prediction ===")
# Create credit risk dataset
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.normal(35, 10, n_samples)
income = np.random.exponential(50000, n_samples)
credit_score = np.random.normal(700, 100, n_samples)

X_credit = np.column_stack([age, income, credit_score])
y_credit = (credit_score + income/1000 + age > 800).astype(int)  # Binary target

# Split credit data
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42
)

# Create pipelines
pipelines = {
    'Linear': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000))
    ])
}

# Compare pipelines
credit_results = compare_models(pipelines, X_credit_train, X_credit_test, 
                               y_credit_train, y_credit_test, 'credit_risk_model_comparison.png')

# 6. Model Selection Process Visualization
print("\n=== Model Selection Process ===")
def model_selection_process(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define models
    models = {
        'Linear': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
    }
    
    # Compare models
    results = compare_models(models, X_train, X_test, y_train, y_test, 'model_selection_process.png')
    
    # Plot learning curves for best model
    best_model_name = max(results.keys(), key=lambda k: results[k])
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]:.3f}")
    
    # Create comprehensive learning curves for all models
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[idx].plot(train_sizes, train_mean, 'o-', label='Training score', color='blue')
        axes[idx].plot(train_sizes, val_mean, 'o-', label='Cross-validation score', color='red')
        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
        axes[idx].set_xlabel('Training Examples')
        axes[idx].set_ylabel('Score')
        axes[idx].set_title(f'Learning Curve - {name}')
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/comprehensive_learning_curves.png', bbox_inches='tight')
    plt.close()
    
    return results

final_results = model_selection_process(X, y)

print("\n=== Summary ===")
print("Generated visualizations:")
print("- linear_decision_boundary.png")
print("- feature_importance.png") 
print("- learning_curve.png")
print("- model_comparison.png")
print("- credit_risk_model_comparison.png")
print("- model_selection_process.png")
print("- comprehensive_learning_curves.png")

print(f"\nFinal Model Accuracies:")
for model, accuracy in final_results.items():
    print(f"- {model}: {accuracy:.3f}")

print("\nAll visualizations have been generated successfully!")
