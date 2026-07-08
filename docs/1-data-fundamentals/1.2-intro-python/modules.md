# Python Modules in Data Science

**After this lesson:** you can explain Python Modules in Data Science and try the examples in your own notebook.

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/sugvnHA7ElY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*Corey Schafer, How Python runs modules and the `if __name__ == "__main__"` guard*

> **AI Learning:** Ask "Explain Python modules using a library analogy"

> **Modern Tools:** Learn to use virtual environments with `uv` or `conda`

## Understanding modules in data analysis

A **module** is a `.py` file (or package) that holds related code. In real projects you rarely put an entire pipeline in one notebook cell: you **import** functions and classes from modules so notebooks stay readable and tests can target one file at a time.

### What lives in "data science" modules?

Typical building blocks you might split out:

- **Preprocessing**: Cleaning, type fixes, winsorizing outliers (used on every dataset refresh).
- **Feature helpers**: Date parts, rolling windows, encodings shared across models.
- **Evaluation**: Metrics and plots so train and validation use the same definitions.
- **Plotting**: Brand-consistent chart defaults so reports look uniform.

Together these pieces form a **library** your team imports instead of copy-pasting cells.

{% include mermaid-diagram.html src="1-data-fundamentals/1.2-intro-python/diagrams/modules-1.mmd" %}

*Each `.py` module does one job. The notebook stays readable because all the boilerplate is imported, not copy-pasted.*

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Module Imports</span>
    </div>
    <div class="code-callout__body">
      <p>Standard data science imports at the top of a module so all functions share the same dependencies.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Numeric Cleaner</span>
    </div>
    <div class="code-callout__body">
      <p>Replaces infinite values with NaN then fills NaN with the column median, a safe default for numeric pipelines.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-33" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Date Feature Builder</span>
    </div>
    <div class="code-callout__body">
      <p>Parses a date column and extracts year, month, day, and day-of-week as numeric features models can use directly.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="35-50" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Regression Metrics</span>
    </div>
    <div class="code-callout__body">
      <p>Lazy-imports sklearn metrics inside the function, then returns MSE, MAE, and R² in a dict for consistent evaluation across models.</p>
    </div>
  </div>
</aside>
</div>

---

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

## Essential Data Science Modules

---

### Core Data Analysis Modules
Common modules for data analysis:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">NumPy Basics</span>
    </div>
    <div class="code-callout__body">
      <p>Creates a 2D array and computes mean, standard deviation, and matrix multiplication, NumPy's core numeric operations.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pandas Wrangling</span>
    </div>
    <div class="code-callout__body">
      <p>Reads a CSV, generates summary statistics, groups by category, and builds a pivot table, the typical EDA workflow.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-36" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sklearn Pipeline</span>
    </div>
    <div class="code-callout__body">
      <p>Splits data, scales features, then fits a RandomForest, the standard train/scale/fit pattern for classification.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="38-45" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Visualization</span>
    </div>
    <div class="code-callout__body">
      <p>Creates a scatter plot coloured by category using Seaborn on top of Matplotlib, the most common plotting combo.</p>
    </div>
  </div>
</aside>
</div>

---

### Advanced Data Science Modules
Specialized modules for specific tasks:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
X = sm.add_constant(X) # Add intercept
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-13" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scipy Stats</span>
    </div>
    <div class="code-callout__body">
      <p>Runs a t-test and Pearson correlation for hypothesis testing, plus numerical optimisation with a simple quadratic objective.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="15-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Statsmodels OLS</span>
    </div>
    <div class="code-callout__body">
      <p>Fits an Ordinary Least Squares regression with a constant term and prints a full statistical summary including p-values and confidence intervals.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-33" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">XGBoost Training</span>
    </div>
    <div class="code-callout__body">
      <p>Wraps data in a DMatrix, sets depth and learning rate parameters, then trains a gradient boosting classifier for 100 rounds.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="35-46" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Plotly Chart</span>
    </div>
    <div class="code-callout__body">
      <p>Creates an interactive scatter plot where colour encodes category and point size encodes a numeric value, hover reveals the ID.</p>
    </div>
  </div>
</aside>
</div>

## Creating Data Science Modules

---

### Module Organization
Example of a well-organized data science module:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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

   def fit(self, X: pd.DataFrame, y=None):
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

   def transform(self, X: pd.DataFrame) -> pd.DataFrame:
       """Transform numeric features"""
       X = X.copy()
       for col in X.columns:
           stats = self.stats[col]
           X[f'{col}_zscore'] = (X[col] - stats['mean']) / stats['std']
           X[f'{col}_normalized'] = (
               (X[col] - stats['min']) / (stats['max'] - stats['min'])
           )
           X[f'{col}_to_median'] = X[col] / stats['median']
       return X

# Similar implementations for other transformers...

def main():
   """Example usage"""
   df = pd.DataFrame({
       'amount': [100, 200, 300],
       'category': ['A', 'B', 'A'],
       'order_date': ['2023-01-01', '2023-01-02', '2023-01-03']
   })
   engineer = FeatureEngineer()
   features = engineer.fit_transform(df)
   print("Engineered features shape:", features.shape)

if __name__ == "__main__":
   main()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-25" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Module Header</span>
    </div>
    <div class="code-callout__body">
      <p>A module docstring, standard imports, and constants at the top establish shared feature-name lists for all functions below.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="27-50" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">FeatureEngineer Init</span>
    </div>
    <div class="code-callout__body">
      <p>Accepts optional feature-name lists defaulting to the constants above, then instantiates one specialised transformer per feature type.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="52-75" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">fit_transform Dispatch</span>
    </div>
    <div class="code-callout__body">
      <p>Calls each transformer only when the corresponding feature list is non-empty, chaining transformations on a copy of the DataFrame.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="77-102" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">NumericTransformer</span>
    </div>
    <div class="code-callout__body">
      <p>Follows the sklearn fit/transform pattern: <code>fit</code> stores column statistics, <code>transform</code> creates z-score, min-max, and median-ratio features.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="104-115" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Entry Point</span>
    </div>
    <div class="code-callout__body">
      <p>The <code>if __name__ == "__main__"</code> guard lets the module be imported without running the demo, a standard Python best practice.</p>
    </div>
  </div>
</aside>
</div>

---

### Project Structure
Example of a data science project structure:

```
project/
├── data/
│  ├── raw/
│  ├── processed/
│  └── external/
├── src/
│  ├── __init__.py
│  ├── data/
│  │  ├── __init__.py
│  │  ├── make_dataset.py
│  │  └── data_utils.py
│  ├── features/
│  │  ├── __init__.py
│  │  └── build_features.py
│  ├── models/
│  │  ├── __init__.py
│  │  ├── train_model.py
│  │  └── predict_model.py
│  └── visualization/
│    ├── __init__.py
│    └── visualize.py
├── notebooks/
│  ├── 1.0-data-exploration.ipynb
│  └── 2.0-modeling.ipynb
├── tests/
│  └── test_features.py
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

## Package Management for Data Science

---

### Managing Dependencies
Common data science package management:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight bash %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Create Environment</span>
    </div>
    <div class="code-callout__body">
      <p>Creates a named conda environment pinned to Python 3.8 and activates it so subsequent installs go into that isolated environment.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-10" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Install Packages</span>
    </div>
    <div class="code-callout__body">
      <p>Installs core data science libraries from the default channel and conda-forge for packages like XGBoost that need it.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-15" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Export Environment</span>
    </div>
    <div class="code-callout__body">
      <p>Exports the environment to a YAML file so teammates can recreate the exact same setup with <code>conda env create</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-21" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pip Extras</span>
    </div>
    <div class="code-callout__body">
      <p>Installs packages not available on conda via pip, then freezes all installed versions to a requirements file for reproducibility.</p>
    </div>
  </div>
</aside>
</div>

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

---

### Development Tools
Essential tools for data science development:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight bash %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Install Dev Tools</span>
    </div>
    <div class="code-callout__body">
      <p>Installs JupyterLab for notebooks, Black/Flake8/mypy for code quality, and pytest with coverage reporting.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Lint and Type Check</span>
    </div>
    <div class="code-callout__body">
      <p>Black auto-formats code, Flake8 checks PEP 8 style violations, and mypy catches type errors before runtime.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-16" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Run Tests</span>
    </div>
    <div class="code-callout__body">
      <p>Runs the full test suite with coverage so you can see which lines of your module are not yet exercised by tests.</p>
    </div>
  </div>
</aside>
</div>

Example test file:
<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Test Imports</span>
    </div>
    <div class="code-callout__body">
      <p>Imports pytest alongside pandas and numpy and the module under test so each test function has everything it needs.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Arrange Test Data</span>
    </div>
    <div class="code-callout__body">
      <p>Creates a minimal DataFrame with a NaN value to verify the transformer handles missing data, then instantiates FeatureEngineer for that column only.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-25" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Assert Output</span>
    </div>
    <div class="code-callout__body">
      <p>Checks that z-score and normalised columns were created and that no nulls remain, covering both shape and correctness of the transformation.</p>
    </div>
  </div>
</aside>
</div>

## Practice Exercises for Data Science
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
- **Use AI to generate docstrings and tests**
- **Check code quality with automated tools**

> **AI for Modules:**
> - "Generate a Python module structure for [your project]"
> - "Create unit tests for this module: [paste code]"
> - "Review my module organization and suggest improvements"

> **Learn More:** Check [Video Resources](./video-resources.md) - Modules section

## Common pitfalls

- **Circular imports**: Two modules importing each other at load time causes errors; move shared code to a third module or defer imports.
- **Name clashes**: **from m import *** pollutes your namespace; prefer **import m** or explicit names.
- **Wrong working directory**: Relative file paths depend on where you run the script; use **pathlib** or pass paths explicitly.

## Next steps

Continue to [Introduction to Statistics](../1.3-intro-statistics/README.md), starting with [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) (or follow your instructor's order within submodule 1.3).

Happy coding!
