# modules

## Python Modules in Data Science

**After this lesson:** you can explain Python Modules in Data Science and try the examples in your own notebook.

#### Video

_Corey Schafer, How Python runs modules and the `if __name__ == "__main__"` guard_

> **AI Learning:** Ask "Explain Python modules using a library analogy"

> **Modern Tools:** Learn to use virtual environments with `uv` or `conda`

### Understanding modules in data analysis

A **module** is a `.py` file (or package) that holds related code. In real projects you rarely put an entire pipeline in one notebook cell: you **import** functions and classes from modules so notebooks stay readable and tests can target one file at a time.

#### What lives in "data science" modules?

Typical building blocks you might split out:

* **Preprocessing**: Cleaning, type fixes, winsorizing outliers (used on every dataset refresh).
* **Feature helpers**: Date parts, rolling windows, encodings shared across models.
* **Evaluation**: Metrics and plots so train and validation use the same definitions.
* **Plotting**: Brand-consistent chart defaults so reports look uniform.

Together these pieces form a **library** your team imports instead of copy-pasting cells.

_Each `.py` module does one job. The notebook stays readable because all the boilerplate is imported, not copy-pasted._

Module Imports

Standard data science imports at the top of a module so all functions share the same dependencies.

Numeric Cleaner

Replaces infinite values with NaN then fills NaN with the column median, a safe default for numeric pipelines.

Date Feature Builder

Parses a date column and extracts year, month, day, and day-of-week as numeric features models can use directly.

Regression Metrics

Lazy-imports sklearn metrics inside the function, then returns MSE, MAE, and R² in a dict for consistent evaluation across models.

***

#### Why Use Modules in Data Science?

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

### Essential Data Science Modules

***

#### Core Data Analysis Modules

Common modules for data analysis:

NumPy Basics

Creates a 2D array and computes mean, standard deviation, and matrix multiplication, NumPy's core numeric operations.

Pandas Wrangling

Reads a CSV, generates summary statistics, groups by category, and builds a pivot table, the typical EDA workflow.

Sklearn Pipeline

Splits data, scales features, then fits a RandomForest, the standard train/scale/fit pattern for classification.

Visualization

Creates a scatter plot coloured by category using Seaborn on top of Matplotlib, the most common plotting combo.

***

#### Advanced Data Science Modules

Specialized modules for specific tasks:

Scipy Stats

Runs a t-test and Pearson correlation for hypothesis testing, plus numerical optimisation with a simple quadratic objective.

Statsmodels OLS

Fits an Ordinary Least Squares regression with a constant term and prints a full statistical summary including p-values and confidence intervals.

XGBoost Training

Wraps data in a DMatrix, sets depth and learning rate parameters, then trains a gradient boosting classifier for 100 rounds.

Plotly Chart

Creates an interactive scatter plot where colour encodes category and point size encodes a numeric value, hover reveals the ID.

### Creating Data Science Modules

***

#### Module Organization

Example of a well-organized data science module:

Module Header

A module docstring, standard imports, and constants at the top establish shared feature-name lists for all functions below.

FeatureEngineer Init

Accepts optional feature-name lists defaulting to the constants above, then instantiates one specialised transformer per feature type.

fit\_transform Dispatch

Calls each transformer only when the corresponding feature list is non-empty, chaining transformations on a copy of the DataFrame.

NumericTransformer

Follows the sklearn fit/transform pattern: `fit` stores column statistics, `transform` creates z-score, min-max, and median-ratio features.

Entry Point

The `if __name__ == "__main__"` guard lets the module be imported without running the demo, a standard Python best practice.

***

#### Project Structure

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

### Package Management for Data Science

***

#### Managing Dependencies

Common data science package management:

Create Environment

Creates a named conda environment pinned to Python 3.8 and activates it so subsequent installs go into that isolated environment.

Install Packages

Installs core data science libraries from the default channel and conda-forge for packages like XGBoost that need it.

Export Environment

Exports the environment to a YAML file so teammates can recreate the exact same setup with `conda env create`.

Pip Extras

Installs packages not available on conda via pip, then freezes all installed versions to a requirements file for reproducibility.

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

***

#### Development Tools

Essential tools for data science development:

Install Dev Tools

Installs JupyterLab for notebooks, Black/Flake8/mypy for code quality, and pytest with coverage reporting.

Lint and Type Check

Black auto-formats code, Flake8 checks PEP 8 style violations, and mypy catches type errors before runtime.

Run Tests

Runs the full test suite with coverage so you can see which lines of your module are not yet exercised by tests.

Example test file:

Test Imports

Imports pytest alongside pandas and numpy and the module under test so each test function has everything it needs.

Arrange Test Data

Creates a minimal DataFrame with a NaN value to verify the transformer handles missing data, then instantiates FeatureEngineer for that column only.

Assert Output

Checks that z-score and normalised columns were created and that no nulls remain, covering both shape and correctness of the transformation.

### Practice Exercises for Data Science

Try these advanced exercises:

1.  **Create a Feature Engineering Package**

    ```python
    ```

## Build modules for:

## - Numeric feature engineering

## - Categorical encoding

## - Text feature extraction

## - Time series features

````

2. **Build a Model Evaluation Package**
```python
# Create modules for:
# - Cross-validation
# - Performance metrics
# - Model comparison
# - Results visualization
````

3.  **Develop a Data Pipeline Package**

    ```python
    ```

## Implement modules for:

## - Data loading and saving

## - Data cleaning and validation

## - Feature transformation

## - Model training and prediction

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
```
