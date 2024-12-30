# Feature Engineering 🛠️

Feature engineering is the art and science of transforming raw data into meaningful features that help machine learning models learn better patterns. It's like preparing ingredients before cooking - the quality of your preparation directly impacts the final result! 

## Understanding Features 🎯

Features are the individual properties or characteristics that your model uses to make predictions. They can be represented mathematically as:

$$X = \{x_1, x_2, ..., x_n\} \text{ where each } x_i \text{ is a feature}$$

For example, in a house price prediction model:

```python
# Raw features
raw_features = {
    'size': 2000,          # Square footage (x₁)
    'bedrooms': 3,         # Number of bedrooms (x₂)
    'year_built': 1985,    # Year built (x₃)
    'location': 'suburb'   # Location type (x₄)
}
```

## Types of Feature Engineering 🔄

### 1. Numerical Transformations

#### Scaling
Standardization: $$z = \frac{x - \mu}{\sigma}$$
Normalization: $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data
df = pd.DataFrame({
    'price': [100000, 200000, 150000, 300000],
    'size': [1500, 2000, 1800, 2500],
    'age': [20, 5, 15, 1]
})

# Standardization
scaler = StandardScaler()
df['size_scaled'] = scaler.fit_transform(df[['size']])

# Log transformation (for skewed data)
df['price_log'] = np.log1p(df['price'])

# Binning with statistical reasoning
df['age_group'] = pd.qcut(df['age'], q=3, 
                         labels=['new', 'medium', 'old'])

print("Transformed Features:")
print(df)
```

### 2. Categorical Transformations

#### One-Hot Encoding
For a categorical variable with k categories:
$$x_{cat} \rightarrow [x_1, x_2, ..., x_k] \text{ where } x_i \in \{0,1\}$$

#### Label Encoding
$$x_{cat} \rightarrow x_{numeric} \text{ where } x_{numeric} \in \{0,1,...,k-1\}$$

```python
# Sample categorical data
df = pd.DataFrame({
    'color': ['red', 'blue', 'red', 'green'],
    'size': ['S', 'M', 'L', 'M'],
    'brand': ['A', 'A', 'B', 'C']
})

# One-Hot Encoding with proper naming
one_hot = pd.get_dummies(df['color'], prefix='color')
print("\nOne-Hot Encoding:")
print(one_hot)

# Label Encoding for ordinal data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['brand_encoded'] = le.fit_transform(df['brand'])

# Ordinal Encoding with domain knowledge
size_map = {'S': 1, 'M': 2, 'L': 3}
df['size_ordinal'] = df['size'].map(size_map)
```

### 3. Feature Creation

#### Mathematical Combinations
Area: $$A = length \times width$$
Volume: $$V = length \times width \times height$$
Density: $$\rho = \frac{mass}{volume}$$

```python
# Creating meaningful combinations
df = pd.DataFrame({
    'length': [10, 20, 15, 25],
    'width': [5, 10, 8, 12],
    'height': [3, 6, 4, 8],
    'age_years': [2, 5, 3, 7],
    'maintenance_count': [1, 3, 2, 4]
})

# Geometric features
df['area'] = df['length'] * df['width']
df['volume'] = df['area'] * df['height']
df['aspect_ratio'] = df['length'] / df['width']

# Rate features
df['maintenance_rate'] = df['maintenance_count'] / df['age_years']

# Statistical aggregations
df['size_mean'] = df[['length', 'width', 'height']].mean(axis=1)
df['size_std'] = df[['length', 'width', 'height']].std(axis=1)
```

### 4. Time-Based Features

#### Cyclical Encoding
For periodic features like hour (h), day (d), or month (m):

$$sin_{h} = \sin(\frac{2\pi h}{24}), cos_{h} = \cos(\frac{2\pi h}{24})$$
$$sin_{d} = \sin(\frac{2\pi d}{7}), cos_{d} = \cos(\frac{2\pi d}{7})$$
$$sin_{m} = \sin(\frac{2\pi m}{12}), cos_{m} = \cos(\frac{2\pi m}{12})$$

```python
# Working with datetime features
df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=4)
})

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek

# Cyclical encoding for month
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# Business logic features
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
```

## Feature Selection Techniques 📊

### 1. Correlation-based Selection
Pearson correlation coefficient:
$$r_{xy} = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sqrt{\sum(x - \bar{x})^2\sum(y - \bar{y})^2}}$$

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix visualization
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()

# Remove highly correlated features
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns 
               if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)
```

### 2. Feature Importance Analysis

#### Random Forest Importance
$$Importance(x_i) = \frac{\sum \Delta Impurity(x_i)}{\sum \Delta Impurity(all)}$$

```python
from sklearn.ensemble import RandomForestRegressor

# Calculate feature importance
X = df.drop('target', axis=1)
y = df['target']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Visualize importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance')
plt.show()
```

## Best Practices 🌟

### 1. Handle Missing Values Strategically
```python
class MissingValueHandler:
    def __init__(self, df):
        self.df = df.copy()
        
    def analyze_missingness(self):
        """Analyze missing value patterns"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        return pd.DataFrame({
            'missing_count': missing,
            'missing_pct': missing_pct
        }).query('missing_count > 0')
        
    def impute_numeric(self, strategy='mean'):
        """Impute numeric values"""
        numeric_cols = self.df.select_dtypes(
            include=[np.number]).columns
        for col in numeric_cols:
            if strategy == 'mean':
                value = self.df[col].mean()
            elif strategy == 'median':
                value = self.df[col].median()
            self.df[col] = self.df[col].fillna(value)
            
    def impute_categorical(self, strategy='mode'):
        """Impute categorical values"""
        cat_cols = self.df.select_dtypes(
            include=['object']).columns
        for col in cat_cols:
            if strategy == 'mode':
                value = self.df[col].mode()[0]
            elif strategy == 'unknown':
                value = 'unknown'
            self.df[col] = self.df[col].fillna(value)
```

### 2. Scale at the Right Time
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create a proper pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit pipeline on training data only
pipeline.fit(X_train, y_train)
```

### 3. Document Transformations
```python
class FeatureTransformationLogger:
    def __init__(self):
        self.transformations = {
            'numeric_features': {},
            'categorical_features': {},
            'created_features': {},
            'dropped_features': []
        }
        
    def log_transformation(self, feature, category, transform):
        """Log a transformation"""
        self.transformations[category][feature] = transform
        
    def log_dropped_feature(self, feature, reason):
        """Log dropped features"""
        self.transformations['dropped_features'].append({
            'feature': feature,
            'reason': reason
        })
        
    def get_report(self):
        """Generate transformation report"""
        return pd.DataFrame(self.transformations)
```

## Common Pitfalls and Solutions ⚠️

### 1. Data Leakage Prevention
```python
class LeakageChecker:
    @staticmethod
    def check_temporal_leakage(df, time_col, feature_cols):
        """Check for temporal leakage"""
        sorted_df = df.sort_values(time_col)
        for col in feature_cols:
            if sorted_df[col].shift(1).corr(sorted_df[col]) > 0.95:
                print(f"Warning: Potential leakage in {col}")
                
    @staticmethod
    def check_target_leakage(feature_cols, target_col):
        """Check for target leakage"""
        suspicious_words = [target_col, 'future', 'next']
        for col in feature_cols:
            if any(word in col.lower() for word in suspicious_words):
                print(f"Warning: Potential target leakage in {col}")
```

### 2. Feature Complexity Management
```python
def evaluate_feature_impact(X, y, feature_sets):
    """Evaluate impact of feature complexity"""
    results = {}
    for name, features in feature_sets.items():
        model = RandomForestRegressor()
        scores = cross_val_score(model, X[features], y, cv=5)
        results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'n_features': len(features)
        }
    return pd.DataFrame(results).T
```

## Feature Engineering Checklist ✅

1. **Data Understanding**
   - [ ] Analyze data types and distributions
   - [ ] Identify missing values and patterns
   - [ ] Check for outliers and anomalies

2. **Basic Transformations**
   - [ ] Handle missing values appropriately
   - [ ] Encode categorical variables
   - [ ] Scale numerical features
   - [ ] Handle outliers

3. **Feature Creation**
   - [ ] Create domain-specific features
   - [ ] Generate interaction terms
   - [ ] Apply mathematical transformations
   - [ ] Extract temporal components

4. **Feature Selection**
   - [ ] Remove redundant features
   - [ ] Select important features
   - [ ] Validate feature impact
   - [ ] Document selection criteria

5. **Validation**
   - [ ] Check for data leakage
   - [ ] Validate transformations
   - [ ] Test feature importance
   - [ ] Measure performance impact

## Next Steps 📚

Now that you understand feature engineering, let's explore the [Bias-Variance Tradeoff](./bias-variance.md) to learn how to balance model complexity and performance!
