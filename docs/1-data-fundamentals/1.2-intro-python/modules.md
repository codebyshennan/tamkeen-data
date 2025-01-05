# Python Modules in Data Science

## Understanding Modules in Data Analysis

{% stepper %}
{% step %}
### Modules in Data Science
Think of modules as reusable components in your data analysis workflow:
- Data preprocessing utilities
- Feature engineering functions
- Model evaluation tools
- Visualization helpers

```python
# Example data science module structure
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Union

# Data preprocessing utilities
def clean_numeric_data(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """Clean numeric columns in DataFrame"""
    df = df.copy()
    for col in columns:
        # Replace infinite values
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Fill missing values with median
        df[col] = df[col].fillna(df[col].median())
    return df

# Feature engineering functions
def create_date_features(
    df: pd.DataFrame,
    date_column: str
) -> pd.DataFrame:
    """Create features from date column"""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    return df

# Model evaluation tools
def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate regression metrics"""
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score
    )
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
```
{% endstep %}

{% step %}
### Why Use Modules in Data Science?
Modules help you:
1. **Create reproducible analysis pipelines**
2. **Share code between team members**
3. **Maintain consistent preprocessing steps**
4. **Organize complex data projects**

Example without modules:
```python
# Without modules (repetitive and error-prone)
# Preprocessing Dataset 1
df1['date'] = pd.to_datetime(df1['date'])
df1['year'] = df1['date'].dt.year
df1['month'] = df1['date'].dt.month
df1.dropna(inplace=True)
df1['amount'] = df1['amount'].clip(lower=0)

# Preprocessing Dataset 2 (repeating same steps)
df2['date'] = pd.to_datetime(df2['date'])
df2['year'] = df2['date'].dt.year
df2['month'] = df2['date'].dt.month
df2.dropna(inplace=True)
df2['amount'] = df2['amount'].clip(lower=0)
```

Example with modules:
```python
# data_preprocessing.py
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standard preprocessing pipeline"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df.dropna(inplace=True)
    df['amount'] = df['amount'].clip(lower=0)
    return df

# Using the module
from data_preprocessing import preprocess_dataset

df1_processed = preprocess_dataset(df1)
df2_processed = preprocess_dataset(df2)
```
{% endstep %}
{% endstepper %}

## Essential Data Science Modules

{% stepper %}
{% step %}
### Core Data Analysis Modules
Common modules for data analysis:

```python
# NumPy: Numerical computations
import numpy as np

# Create array and perform operations
data = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Mean: {data.mean()}")
print(f"Standard deviation: {data.std()}")
print(f"Matrix multiplication: \n{data @ data.T}")

# Pandas: Data manipulation
import pandas as pd

# Read and process data
df = pd.read_csv('data.csv')
summary = df.describe()
grouped = df.groupby('category')['value'].mean()
pivoted = df.pivot_table(
    values='amount',
    index='date',
    columns='category',
    aggfunc='sum'
)

# Scikit-learn: Machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare and train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Matplotlib & Seaborn: Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Create visualizations
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y', hue='category')
plt.title('Data Distribution')
plt.show()
```
{% endstep %}

{% step %}
### Advanced Data Science Modules
Specialized modules for specific tasks:

```python
# Scipy: Scientific computing
from scipy import stats
from scipy.optimize import minimize

# Statistical tests
t_stat, p_value = stats.ttest_ind(group1, group2)
correlation = stats.pearsonr(x, y)

# Optimization
result = minimize(
    lambda x: (x[0] - 1)**2 + (x[1] - 2)**2,
    x0=[0, 0]
)

# Statsmodels: Statistical modeling
import statsmodels.api as sm

# Linear regression with statistics
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# XGBoost: Gradient boosting
import xgboost as xgb

# Train boosting model
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic'
}
model = xgb.train(params, dtrain, num_boost_round=100)

# Plotly: Interactive visualization
import plotly.express as px
import plotly.graph_objects as go

# Create interactive plots
fig = px.scatter(
    df,
    x='x',
    y='y',
    color='category',
    size='value',
    hover_data=['id']
)
fig.show()
```
{% endstep %}
{% endstepper %}

## Creating Data Science Modules

{% stepper %}
{% step %}
### Module Organization
Example of a well-organized data science module:

```python
"""
Feature Engineering Module

This module provides utilities for feature engineering in data science projects.
It includes functions for creating features from different data types:
- Numeric features
- Categorical features
- Date features
- Text features

Author: Your Name
Version: 1.0.0
"""

# Standard imports
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin

# Constants
NUMERIC_FEATURES = ['amount', 'quantity', 'price']
CATEGORICAL_FEATURES = ['category', 'region', 'product']
DATE_FEATURES = ['order_date', 'shipping_date']
TEXT_FEATURES = ['description', 'comments']

class FeatureEngineer:
    """Feature engineering for different data types"""
    
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        date_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None
    ):
        """
        Initialize feature engineer
        
        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            date_features: List of date column names
            text_features: List of text column names
        """
        self.numeric_features = numeric_features or NUMERIC_FEATURES
        self.categorical_features = (
            categorical_features or CATEGORICAL_FEATURES
        )
        self.date_features = date_features or DATE_FEATURES
        self.text_features = text_features or TEXT_FEATURES
        
        # Initialize transformers
        self.numeric_transformer = NumericTransformer()
        self.categorical_transformer = CategoricalTransformer()
        self.date_transformer = DateTransformer()
        self.text_transformer = TextTransformer()
    
    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create features from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Transform each feature type
        if self.numeric_features:
            df = self.numeric_transformer.fit_transform(
                df[self.numeric_features]
            )
        
        if self.categorical_features:
            df = self.categorical_transformer.fit_transform(
                df[self.categorical_features]
            )
        
        if self.date_features:
            df = self.date_transformer.fit_transform(
                df[self.date_features]
            )
        
        if self.text_features:
            df = self.text_transformer.fit_transform(
                df[self.text_features]
            )
        
        return df

class NumericTransformer(BaseEstimator, TransformerMixin):
    """Transform numeric features"""
    
    def __init__(self):
        self.stats = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'NumericTransformer':
        """Calculate statistics for transformations"""
        for col in X.columns:
            self.stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'median': X[col].median(),
                'min': X[col].min(),
                'max': X[col].max()
            }
        return self
    
    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform numeric features"""
        X = X.copy()
        
        for col in X.columns:
            stats = self.stats[col]
            
            # Create new features
            X[f'{col}_zscore'] = (
                (X[col] - stats['mean']) / stats['std']
            )
            X[f'{col}_normalized'] = (
                (X[col] - stats['min']) /
                (stats['max'] - stats['min'])
            )
            X[f'{col}_to_median'] = X[col] / stats['median']
        
        return X

# Similar implementations for other transformers...

def main():
    """Example usage"""
    # Create sample data
    df = pd.DataFrame({
        'amount': [100, 200, 300],
        'category': ['A', 'B', 'A'],
        'order_date': ['2023-01-01', '2023-01-02', '2023-01-03']
    })
    
    # Create and use feature engineer
    engineer = FeatureEngineer()
    features = engineer.fit_transform(df)
    print("Engineered features shape:", features.shape)

if __name__ == "__main__":
    main()
```
{% endstep %}

{% step %}
### Project Structure
Example of a data science project structure:

```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   └── data_utils.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py
├── notebooks/
│   ├── 1.0-data-exploration.ipynb
│   └── 2.0-modeling.ipynb
├── tests/
│   └── test_features.py
├── requirements.txt
└── setup.py
```

Example `setup.py`:
```python
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Data science project',
    author='Your Name',
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.2',
        'seaborn>=0.11.0'
    ],
    python_requires='>=3.8'
)
```
{% endstep %}
{% endstepper %}

## Package Management for Data Science

{% stepper %}
{% step %}
### Managing Dependencies
Common data science package management:

```bash
# Create virtual environment with conda
conda create -n ds_env python=3.8

# Activate environment
conda activate ds_env

# Install data science packages
conda install numpy pandas scikit-learn
conda install -c conda-forge xgboost lightgbm

# Create environment file
conda env export > environment.yml

# Create environment from file
conda env create -f environment.yml

# Install additional packages with pip
pip install category_encoders
pip install optuna

# Save pip requirements
pip freeze > requirements.txt
```

Example `environment.yml`:
```yaml
name: ds_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - numpy=1.19.2
  - pandas=1.2.0
  - scikit-learn=0.24.0
  - matplotlib=3.3.2
  - seaborn=0.11.0
  - jupyter=1.0.0
  - pip:
    - category_encoders==2.2.2
    - optuna==2.10.0
```
{% endstep %}

{% step %}
### Development Tools
Essential tools for data science development:

```bash
# Install development tools
conda install -c conda-forge jupyterlab
conda install -c conda-forge black flake8 mypy
conda install pytest pytest-cov

# Format code
black src/

# Check code style
flake8 src/

# Run type checking
mypy src/

# Run tests with coverage
pytest --cov=src tests/
```

Example test file:
```python
# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from src.features.build_features import FeatureEngineer

def test_numeric_features():
    """Test numeric feature engineering"""
    # Create test data
    df = pd.DataFrame({
        'amount': [100, 200, np.nan, 400]
    })
    
    # Create feature engineer
    engineer = FeatureEngineer(
        numeric_features=['amount']
    )
    
    # Transform data
    result = engineer.fit_transform(df)
    
    # Check results
    assert 'amount_zscore' in result.columns
    assert 'amount_normalized' in result.columns
    assert not result.isnull().any().any()
```
{% endstep %}
{% endstepper %}

## Practice Exercises for Data Science 🎯

Try these advanced exercises:

1. **Create a Feature Engineering Package**
   ```python
   # Build modules for:
   # - Numeric feature engineering
   # - Categorical encoding
   # - Text feature extraction
   # - Time series features
   ```

2. **Build a Model Evaluation Package**
   ```python
   # Create modules for:
   # - Cross-validation
   # - Performance metrics
   # - Model comparison
   # - Results visualization
   ```

3. **Develop a Data Pipeline Package**
   ```python
   # Implement modules for:
   # - Data loading and saving
   # - Data cleaning and validation
   # - Feature transformation
   # - Model training and prediction
   ```

Remember:
- Use type hints
- Write comprehensive docstrings
- Include unit tests
- Follow PEP 8 style guide
- Create clear documentation

Happy coding! 🚀
