---
reading_minutes: 35
objectives:
  - >-
    Translate a business question into a problem statement that names goal,
    type, metric, data needs, and success criteria.
  - >-
    Run a data audit (shape, dtypes, missingness, target distribution,
    correlation heatmap) before any modeling.
  - >-
    Apply a reusable cleaning pattern (median/mode imputation, z-score outlier
    filtering) without leaking test statistics.
  - >-
    Compare baseline regressors with a single MAE/R² helper on a held-out
    validation fold.
  - >-
    Persist the model, scaler, and feature list together so inference
    reconstructs the training pipeline exactly.
---

# Machine Learning Workflow: A Step-by-Step Guide

**After this lesson:** you can explain Machine Learning Workflow: A Step-by-Step Guide and try the examples in your own notebook.

## Overview

A **workflow** is the repeatable path from a business or research question through data, modeling, evaluation, and (when appropriate) deployment. This page walks that path with house-price-style examples so you see how each stage connects to the next. Read [What is Machine Learning?](what-is-ml.md) first if the problem types are still new.

## Why this matters

Skipping steps, especially clear problem definition, honest splits, and evaluation, produces models that look good in a notebook and fail in production. A shared workflow also keeps teams aligned on what "done" means and what to document.

Welcome to our guide on the machine learning workflow! This guide will walk you through each step of building a machine learning solution, with practical examples and clear explanations.

## What is a Machine Learning Workflow?

A machine learning workflow is a systematic process that helps us build effective ML solutions. Think of it as a recipe for creating machine learning models. Just like a recipe has specific steps to follow, the ML workflow has clear stages that help us build better models.

## Why is a Workflow Important?

Structured stages reduce rework: you discover data issues before training, pick metrics that match the problem before tuning, and keep a validation set untouched until you are ready to estimate generalization. In short, the workflow is how you turn ad hoc experiments into something you can ship and maintain.

Following a structured workflow helps us:

1. Stay organized and systematic
2. Avoid common mistakes
3. Build better models
4. Save time and resources
5. Make our work reproducible

## The Machine Learning Workflow Steps

The workflow consists of six main steps:

1. Problem Definition
2. Data Collection and Exploration
3. Data Preparation
4. Model Selection and Training
5. Model Evaluation
6. Model Deployment

we will look at each step in detail.

## 1. Problem Definition

### Understanding the Problem

Before writing any code, we need to clearly understand what we're trying to solve. This is like planning a journey - we need to know our destination before we start.

Key questions to ask:

* What problem are we trying to solve?
* What are our success metrics?
* What data do we need?
* How will the solution be used?

### Types of Machine Learning Problems

There are three main types of ML problems:

1. **Regression**: Predicting continuous values
   * Example: House prices, temperature forecasting
   * Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R²
   * Formula: $$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$
2. **Classification**: Predicting categories
   * Example: Spam detection, image recognition
   * Metrics: Accuracy, Precision, Recall, F1-score
   * Formula: $$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}$$
3. **Clustering**: Finding natural groups
   * Example: Customer segmentation
   * Metrics: Silhouette Score, Davies-Bouldin Index
   * Formula: $$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

### Example Problem Statement

look at a concrete example:

#### Document the problem before writing modeling code

```python
"""
Problem Statement Example:
Goal: Predict house prices
Type: Regression problem
Success Metric: Predictions within $50,000 of actual price
Required Data: House features (size, location, etc.)
Business Impact: Help real estate agents price houses accurately
"""
```

```

Problem Statement Example:
Goal: Predict house prices
Type: Regression problem
Success Metric: Predictions within $50,000 of actual price
Required Data: House features (size, location, etc.)
Business Impact: Help real estate agents price houses accurately
```

## 2. Data Collection and Exploration

### Understanding Your Data

Before we can build a model, we need to understand our data. This is like getting to know the ingredients before cooking.

### Initial Data Assessment

Start by loading and examining our data:

#### Load CSV and inspect shape, dtypes, and missing values

Imports and Load

Standard ML imports, pandas for the dataframe, numpy for numerics, matplotlib and seaborn for plots; `read_csv` loads the raw house dataset.

Shape and Missingness

Print row/column counts, column dtypes, and per-column null counts before any modeling; `describe()` adds numeric range summaries to spot outliers early.

### Exploratory Data Analysis (EDA)

EDA helps us understand patterns and relationships in our data:

#### Visualize the target and feature correlations

Price Distribution

`histplot` shows whether house prices are skewed or multimodal, skew suggests a log transform may help the model; multimodal peaks can indicate distinct market segments.

Correlation Heatmap

`df.corr()` computes pairwise Pearson correlations; the heatmap with annotations reveals which features move together, high inter-feature correlation flags potential multicollinearity.

The histogram shows whether the target is skewed or multimodal (which can affect metrics and transforms). The correlation matrix is a first pass at **linear** relationships only; strong nonlinear links may not appear here.

## 3. Data Preparation

### Why Prepare Data?

Data preparation is like preparing ingredients for cooking. We need to clean and transform our data to make it suitable for modeling.

### Data Cleaning

Create a helper class to clean our data:

#### Encapsulate imputation and outlier clipping

Class Setup

The class stores a copy of the dataframe so the original is never modified in place.

Fill Missing Values

Numeric columns get median imputation (reliable to skew); categorical columns use the most frequent value (mode).

Remove Outliers

Rows whose value on the chosen column falls beyond `n_std` standard deviations from the mean are dropped via z-score filtering.

Usage Example

Instantiate the cleaner, run both methods in sequence to get a clean dataframe ready for feature engineering.

### Feature Engineering

Feature engineering is about creating new features that might help our model:

#### Derive ratios, totals, and one-hot encodings

Numeric Features

Three derived numeric columns, price per sqft, total rooms, and a renovation flag, encode domain knowledge that raw columns alone cannot express for a linear model.

Categorical Encoding

`pd.get_dummies` one-hot encodes `view` and `condition` into binary columns; `create_features(df)` returns the expanded dataframe ready for splitting and modeling.

## 4. Model Selection and Training

### Why Split Data?

We need to split our data to evaluate our model properly. Think of it as having a practice test and a real test.

### Splitting the Data

#### Train / validation / test split for honest evaluation

```python
from sklearn.model_selection import train_test_split

# Split features and target
X = df.drop('price', axis=1)
y = df['price']

# Create train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

### Training Multiple Models

Try different models to find the best one:

#### Compare baselines with the same MAE / R² reporting

Imports

Three regressor classes plus MAE and R² metric functions are imported for the comparison loop.

Train and Evaluate

The helper fits any sklearn estimator, predicts on both splits, and returns a dict of four metrics to detect train/val divergence.

Model Comparison Loop

Three models are defined in a dict and evaluated with the same helper; `results` collects each model's metrics for side-by-side comparison.

## 5. Model Evaluation

### Why Evaluate Models?

Evaluation helps us understand how well our model performs and where it needs improvement.

### Comprehensive Evaluation

#### Metrics plus an actual-vs-predicted scatter on the test set

Compute Metrics

MAE, RMSE (via `sqrt(mse)`), and R² are computed then printed with dollar-formatted strings for interpretability.

Actual vs Predicted Plot

The scatter plot shows each test sample; the red dashed diagonal is perfect calibration, points above or below reveal systematic over- or under-prediction.

Run on Best Model

Pull the Random Forest from the `models` dict and pass it with the held-out test set to get final, unbiased evaluation numbers.

### Learning Curves Analysis

Learning curves help us understand if our model is learning well:

#### Plot learning curves to spot under- vs overfitting

Compute Curves

`learning_curve` re-fits the model at 10 linearly-spaced training sizes using 5-fold CV, returning raw scores for both the train and validation sets.

Plot Both Curves

Mean train and CV scores are plotted against sample count; a persistent gap indicates overfitting, while both curves being low indicates underfitting.

## 6. Model Deployment

### Why Deploy Models?

Deployment makes our model available for real-world use. It's like opening a restaurant after perfecting a recipe.

### Saving the Model

#### Persist estimator and preprocessing artifacts

Directory Setup

`os.makedirs(path, exist_ok=True)` creates the output directory if it doesn't exist, ensuring `joblib.dump` never fails on a missing path.

Serialize All Artifacts

Model, scaler, and feature names are saved as separate `.joblib` files, storing all three together ensures inference code can reconstruct the exact preprocessing + scoring pipeline.

### Making Predictions

#### Load artifacts and score new rows

Load Artifacts

All three saved objects, model, scaler, and feature list, are loaded from disk so inference is fully self-contained.

Align and Transform

Columns are reordered to match training, then the same scaler is applied so new data arrives in the same numeric range the model saw at fit time.

Predict New House

A single-row DataFrame with four features is passed to the function and the formatted price is printed, this is the minimal inference path.

## Gotchas

* **Calling `fit_transform` on test data instead of `transform`**: `DataCleaner` and scalers compute statistics (mean, std, min, max) during `fit`; applying `fit_transform` to `X_test` re-computes those statistics from the test set, leaking test distribution information into preprocessing and invalidating the evaluation.
* **Cleaning data before the train/test split**: imputing missing values or removing outliers across the full dataset uses information from test rows to influence training-set statistics; always split first, then fit your cleaning pipeline on `X_train` only.
* **Accepting a high validation R² or MAE without examining the actual-vs-predicted scatter**: a good scalar metric can hide systematic over-prediction at high values or heteroscedastic errors that matter in production; the scatter plot against the diagonal is a mandatory sanity check.
* **Selecting the best model by comparing only training scores**: `results[name]['train_mae']` says how well the model memorised the training data, not how well it generalises; always rank models by their `val_mae` or cross-validation score and treat `train_mae` only as a diagnostic for overfitting.
* **Saving only the model file and not the scaler or feature list**: `predict_house_price` reloads all three artifacts; deploying just `model.joblib` means the scoring function cannot apply the same column order and scaling the training pipeline used, silently producing wrong predictions.
* **Using `df.corr()` as the only feature-selection signal**: the correlation heatmap detects _linear_ relationships; features with near-zero Pearson correlation can still carry strong non-linear predictive signal that a tree-based model or polynomial feature would capture.

## Best Practices and Tips

These habits support the same workflow: reproducibility (version control), trust (documentation), and sustainability (monitoring and error analysis).

### 1. Version Control

* Keep track of your code changes
* Document model versions and performance
* Save model artifacts systematically

### 2. Documentation

* Document your assumptions
* Keep track of preprocessing steps
* Record model performance metrics

### 3. Monitoring

* Monitor model performance over time
* Watch for data drift
* Set up alerts for performance degradation

### 4. Error Analysis

* Analyze where your model makes mistakes
* Look for patterns in errors
* Use insights to improve the model

## Next Steps

Now that you understand the ML workflow:

1. Practice with different datasets
2. Try various algorithms
3. Experiment with feature engineering
4. Build end-to-end projects

Remember: The key to success in machine learning is iteration and experimentation. Don't expect perfect results on your first try!
