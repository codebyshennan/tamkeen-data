# Machine Learning Workflow 🔄

Building a machine learning solution is like conducting a scientific experiment - it requires careful planning, systematic execution, and rigorous evaluation. Let's break down this process into manageable steps! 

## 1. Problem Definition 🎯

### Understanding the Business Context
Before diving into code, clearly define the problem:

$$\text{Business Problem} \xrightarrow{\text{Translation}} \text{ML Problem} \xrightarrow{\text{Success Metrics}}$$

```python
"""
Example Problem Statement:
Goal: Predict house prices
Type: Regression (continuous output)
Success Metric: Mean Absolute Error < $50,000
Input Features: House characteristics (size, location, etc.)
Output: Predicted price in dollars
Business Impact: Support real estate valuation
"""
```

### Key Questions Checklist
1. Problem Type:
   - Classification? $$P(y|X) \text{ where } y \in \{1,2,...,K\}$$
   - Regression? $$f(X) = y \text{ where } y \in \mathbb{R}$$
   - Clustering? $$\text{argmin}_C \sum_{i=1}^k \sum_{x \in C_i} ||x - \mu_i||^2$$
2. Success Metrics:
   - Technical (MAE, RMSE, F1-score)
   - Business (ROI, Cost Savings)
3. Data Requirements:
   - Volume needed
   - Features required
   - Quality standards

## 2. Data Collection and Exploration 📊

### Initial Data Assessment
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and examine data
df = pd.read_csv('house_data.csv')

# Dataset overview
print(f"Dataset Shape: {df.shape}")
print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
print("\nFeature Types:\n", df.dtypes)

# Basic statistics with interpretations
stats = df.describe()
print("\nKey Statistics:")
for column in stats.columns:
    print(f"\n{column}:")
    print(f"- Range: {stats[column]['min']:.2f} to {stats[column]['max']:.2f}")
    print(f"- Central Tendency: mean={stats[column]['mean']:.2f}, median={stats[column]['50%']:.2f}")
    print(f"- Spread: std={stats[column]['std']:.2f}")
```

### Exploratory Data Analysis (EDA)
```python
def plot_distribution(df, column):
    plt.figure(figsize=(10, 6))
    
    # Distribution plot
    sns.histplot(data=df, x=column, kde=True)
    
    # Add mean and median lines
    plt.axvline(df[column].mean(), color='red', linestyle='--', 
                label=f'Mean: {df[column].mean():.2f}')
    plt.axvline(df[column].median(), color='green', linestyle='--', 
                label=f'Median: {df[column].median():.2f}')
    
    plt.title(f'Distribution of {column}')
    plt.legend()
    plt.show()

# Analyze key features
for column in ['price', 'sqft_living', 'bedrooms']:
    plot_distribution(df, column)

# Correlation matrix with insights
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.show()

# Print strong correlations
strong_corr = np.where(np.abs(corr_matrix) > 0.5)
for i, j in zip(*strong_corr):
    if i != j:
        print(f"{corr_matrix.index[i]} → {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
```

## 3. Data Preparation 🧹

### Data Cleaning
```python
class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        
    def handle_missing_values(self):
        """Handle missing values with appropriate strategies"""
        # Numerical: mean for normal distributions, median for skewed
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].skew() > 1:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
                
        # Categorical: mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
    def remove_outliers(self, columns, n_std=3):
        """Remove outliers using z-score method"""
        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            self.df = self.df[
                (self.df[col] <= mean + (n_std * std)) & 
                (self.df[col] >= mean - (n_std * std))
            ]
            
    def clean(self):
        """Execute full cleaning pipeline"""
        self.handle_missing_values()
        self.remove_outliers(['price', 'sqft_living'])
        self.df = self.df.drop_duplicates()
        return self.df

# Clean the data
cleaner = DataCleaner(df)
df_cleaned = cleaner.clean()
```

### Feature Engineering
```python
class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        
    def create_numerical_features(self):
        """Create new numerical features"""
        self.df['price_per_sqft'] = self.df['price'] / self.df['sqft_living']
        self.df['total_rooms'] = self.df['bedrooms'] + self.df['bathrooms']
        self.df['age'] = 2023 - self.df['yr_built']
        
    def create_categorical_features(self):
        """Create new categorical features"""
        self.df['is_renovated'] = (self.df['yr_renovated'] > 0).astype(int)
        self.df['price_category'] = pd.qcut(self.df['price'], q=5, 
                                          labels=['very_low', 'low', 'medium', 
                                                 'high', 'very_high'])
                                                 
    def encode_categorical(self):
        """Encode categorical variables"""
        # One-hot encoding for nominal variables
        nominal_cols = ['view', 'condition']
        self.df = pd.get_dummies(self.df, columns=nominal_cols)
        
        # Label encoding for ordinal variables
        from sklearn.preprocessing import LabelEncoder
        ordinal_cols = ['price_category']
        le = LabelEncoder()
        for col in ordinal_cols:
            self.df[col] = le.fit_transform(self.df[col])
            
    def scale_features(self):
        """Scale numerical features"""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_cols = ['sqft_living', 'bedrooms', 'bathrooms', 'age']
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return scaler  # Return for use in predictions
        
    def engineer(self):
        """Execute full feature engineering pipeline"""
        self.create_numerical_features()
        self.create_categorical_features()
        self.encode_categorical()
        scaler = self.scale_features()
        return self.df, scaler

# Engineer features
engineer = FeatureEngineer(df_cleaned)
df_engineered, scaler = engineer.engineer()
```

## 4. Model Selection and Training 🤖

### Data Splitting
The training process follows this pattern:

$$
\text{Data} \xrightarrow{\text{Split}} \begin{cases} 
\text{Training Set (60%)} \\
\text{Validation Set (20%)} \\
\text{Test Set (20%)}
\end{cases}
$$

```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X = df_engineered.drop('price', axis=1)
y = df_engineered['price']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)
```

### Model Training and Selection
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor()
        }
        self.results = {}
        
    def train_evaluate(self, X_train, X_val, y_train, y_val):
        """Train and evaluate all models"""
        for name, model in self.models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Evaluate
            self.results[name] = {
                'train_mae': mean_absolute_error(y_train, train_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'train_r2': r2_score(y_train, train_pred),
                'val_r2': r2_score(y_val, val_pred),
                'model': model
            }
            
    def display_results(self):
        """Display results in a formatted table"""
        print("\nModel Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'Train MAE':>12} {'Val MAE':>12} "
              f"{'Train R²':>12} {'Val R²':>12}")
        print("-" * 80)
        
        for name, metrics in self.results.items():
            print(f"{name:<20} "
                  f"${metrics['train_mae']:>11,.0f} "
                  f"${metrics['val_mae']:>11,.0f} "
                  f"{metrics['train_r2']:>11.3f} "
                  f"{metrics['val_r2']:>11.3f}")
                  
    def get_best_model(self, metric='val_mae'):
        """Get the best performing model"""
        if metric.startswith('val_mae'):
            best_model = min(self.results.items(), 
                           key=lambda x: x[1][metric])
        else:
            best_model = max(self.results.items(), 
                           key=lambda x: x[1][metric])
        return best_model[0], best_model[1]['model']

# Train and evaluate models
trainer = ModelTrainer()
trainer.train_evaluate(X_train, X_val, y_train, y_val)
trainer.display_results()

# Get best model
best_name, best_model = trainer.get_best_model()
print(f"\nBest Model: {best_name}")
```

## 5. Model Evaluation 📈

### Comprehensive Evaluation
```python
class ModelEvaluator:
    def __init__(self, model, X_train, X_val, X_test, 
                 y_train, y_val, y_test):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
    def calculate_metrics(self, y_true, y_pred):
        """Calculate various regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
    def plot_residuals(self, y_true, y_pred, title):
        """Plot residual analysis"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 4))
        
        # Residual scatter plot
        plt.subplot(121)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - {title}')
        
        # Residual distribution
        plt.subplot(122)
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title(f'Residual Distribution - {title}')
        
        plt.tight_layout()
        plt.show()
        
    def plot_prediction_scatter(self, y_true, y_pred, title):
        """Plot predicted vs actual values"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual - {title}')
        plt.legend()
        plt.show()
        
    def evaluate(self):
        """Perform comprehensive evaluation"""
        # Get predictions
        train_pred = self.model.predict(self.X_train)
        val_pred = self.model.predict(self.X_val)
        test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(self.y_train, train_pred)
        val_metrics = self.calculate_metrics(self.y_val, val_pred)
        test_metrics = self.calculate_metrics(self.y_test, test_pred)
        
        # Display metrics
        print("\nModel Performance Metrics:")
        print("-" * 50)
        metrics = ['MAE', 'RMSE', 'R2']
        print(f"{'Metric':<10} {'Train':>12} {'Validation':>12} {'Test':>12}")
        print("-" * 50)
        
        for metric in metrics:
            print(f"{metric:<10} "
                  f"{train_metrics[metric]:>12,.2f} "
                  f"{val_metrics[metric]:>12,.2f} "
                  f"{test_metrics[metric]:>12,.2f}")
        
        # Plot visualizations
        self.plot_residuals(self.y_test, test_pred, 'Test Set')
        self.plot_prediction_scatter(self.y_test, test_pred, 'Test Set')

# Evaluate best model
evaluator = ModelEvaluator(best_model, X_train, X_val, X_test, 
                          y_train, y_val, y_test)
evaluator.evaluate()
```

## 6. Model Deployment and Monitoring 🚀

### Model Persistence
```python
import joblib

class ModelDeployer:
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
    def save_model(self, path='model/'):
        """Save model and associated objects"""
        import os
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.model, f'{path}model.joblib')
        joblib.dump(self.scaler, f'{path}scaler.joblib')
        joblib.dump(self.feature_names, f'{path}features.joblib')
        
    @staticmethod
    def load_model(path='model/'):
        """Load model and associated objects"""
        model = joblib.load(f'{path}model.joblib')
        scaler = joblib.load(f'{path}scaler.joblib')
        features = joblib.load(f'{path}features.joblib')
        
        return ModelDeployer(model, scaler, features)
        
    def predict(self, features_df):
        """Make prediction with proper preprocessing"""
        # Ensure correct features
        features_df = features_df[self.feature_names]
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(scaled_features)
        
        return prediction[0]

# Save model
deployer = ModelDeployer(best_model, scaler, X_train.columns)
deployer.save_model()

# Example prediction
new_house = pd.DataFrame({
    'sqft_living': [2000],
    'bedrooms': [3],
    'bathrooms': [2],
    'age': [15]
    # Add other required features
})

# Load model and predict
loaded_deployer = ModelDeployer.load_model()
predicted_price = loaded_deployer.predict(new_house)
print(f"\nPredicted House Price: ${predicted_price:,.2f}")
```

## Best Practices and Guidelines 📋

### 1. Documentation 📝
Maintain comprehensive documentation:
```python
"""
Model Documentation Template:

1. Problem Definition
   - Business objective
   - Success metrics
   - Constraints

2. Data Description
   - Source
   - Schema
   - Quality metrics
   - Update frequency

3. Feature Engineering
   - Preprocessing steps
   - Feature creation logic
   - Scaling/encoding methods

4. Model Details
   - Algorithm choice rationale
   - Hyperparameters
   - Performance metrics
   - Validation strategy

5. Deployment Information
   - Environment requirements
   - API endpoints
   - Monitoring setup
"""
```

### 2. Version Control 🔄
```bash
# Directory structure
model/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
│   ├── v1/
│   └── v2/
├── notebooks/
├── src/
└── tests/
```

### 3. Monitoring Strategy 📊
```python
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
    def log_prediction(self, prediction, actual=None):
        """Log each prediction and actual value"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(pd.Timestamp.now())
        
    def check_drift(self, window_size=100):
        """Check for data drift"""
        if len(self.predictions) < window_size:
            return
            
        recent_preds = self.predictions[-window_size:]
        
        # Basic drift detection
        mean_shift = np.mean(recent_preds) - np.mean(self.predictions)
        std_shift = np.std(recent_preds) - np.std(self.predictions)
        
        if abs(mean_shift) > 0.5 or abs(std_shift) > 0.5:
            print("Warning: Potential data drift detected!")
```

### 4. Error Analysis 🔍
```python
def analyze_errors(y_true, y_pred, features_df):
    """Analyze prediction errors"""
    errors = np.abs(y_true - y_pred)
    error_df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': errors
    })
    
    # Join with features
    error_analysis = pd.concat([error_df, features_df], axis=1)
    
    # Find worst predictions
    worst_predictions = error_analysis.nlargest(10, 'error')
    
    # Feature correlation with error
    error_correlations = error_analysis.corr()['error'].sort_values()
    
    return worst_predictions, error_correlations
```

## Common Pitfalls and Solutions ⚠️

### 1. Data Leakage Prevention
```python
def prevent_leakage(df, time_column):
    """Ensure temporal data splitting"""
    df = df.sort_values(time_column)
    train_idx = int(len(df) * 0.6)
    val_idx = int(len(df) * 0.8)
    
    train = df[:train_idx]
    val = df[train_idx:val_idx]
    test = df[val_idx:]
    
    return train, val, test
```

### 2. Overfitting Prevention
```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y):
    """Plot learning curves to detect overfitting"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 
             label='Training Score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 
             label='Validation Score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()
```

### 3. Business Context Integration
```python
def calculate_business_impact(y_true, y_pred, cost_per_error=100):
    """Calculate business impact of predictions"""
    errors = np.abs(y_true - y_pred)
    total_cost = np.sum(errors) * cost_per_error
    
    return {
        'total_cost': total_cost,
        'avg_cost_per_prediction': total_cost / len(y_true),
        'worst_case_cost': np.max(errors) * cost_per_error
    }
```

## Next Steps 📚

Now that you understand the machine learning workflow, let's dive into [Feature Engineering](./feature-engineering.md) to learn how to create better input features for your models!
