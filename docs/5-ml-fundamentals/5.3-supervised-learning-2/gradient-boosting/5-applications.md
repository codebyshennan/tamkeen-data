---
reading_minutes: 20
objectives:
  - >-
    Build end-to-end gradient-boosting pipelines for structured-data tasks like
    churn prediction, credit risk, and demand forecasting.
  - >-
    Combine threshold tuning, calibrated probabilities, and class weighting to
    handle imbalanced binary classification.
  - >-
    Estimate inference latency and model size to decide whether a boosted
    ensemble is shippable in your serving environment.
---

# Real-World Applications of Gradient Boosting

**After this lesson:** you can explain Real-World Applications of Gradient Boosting and try the examples in your own notebook.

## Overview

Competition-style tabular problems, ranking, and deployment considerations (latency, model size).

## 1. Financial Applications: Making Smart Money Decisions

### Credit Risk Assessment: Who Gets a Loan?

Imagine you're a bank manager deciding who to give loans to. Gradient Boosting can help make these decisions smarter and fairer.

Imports and Sample Data

Seven applicant features are generated with realistic distributions; the default label is derived from a rule combining high debt ratio, low credit score, and prior defaults.

Train Credit Model

Features are standard-scaled before fitting an XGBClassifier; `scale_pos_weight` compensates for class imbalance by weighting the minority (default) class proportionally.

Risk Scoring Function

Scores a new applicant by transforming their features, extracting the default probability, inverting it to a 0-100 credit score, ranking feature importances, and appending targeted recommendations when risk exceeds 0.3.

### Stock Market Prediction: Finding Patterns in Market Data

Build a system that can help predict stock movements. Think of this as having a smart assistant for stock trading.

Feature Engineering

Computes SMA-20/50, RSI, volume ratio, daily return, and 20-day rolling volatility; the target is the next-day return, shifted by one row so the model predicts one step ahead.

Walk-forward Training

Downloads two years of history, then slides a 252-day (one trading year) window forward: each iteration retrains LGBMRegressor on the preceding year and predicts the next day, simulating realistic out-of-sample evaluation.

## 2. Healthcare Applications: Predicting Health Risks

### Disease Risk Prediction: Early Warning System

Imagine you're a doctor trying to predict which patients might develop certain conditions. Gradient Boosting can help identify at-risk patients early.

Train with Cross-validation

Seven clinical features feed an XGBClassifier with depth=3 (intentionally shallow to prevent overfitting on medical data); stratified 5-fold cross-validation measures ROC-AUC before the final model is fitted on all data.

Risk Assessment

Maps the predicted probability to Low/Moderate/High risk tiers, ranks features by importance, then appends condition-specific recommendations (smoking, BMI, blood pressure) when risk exceeds 0.3.

## 3. Marketing Applications: Understanding Customers

### Customer Churn Prediction: Keeping Customers Happy

Build a system that can predict which customers might leave a service. This is like having a crystal ball for customer retention.

Feature Lists

Nine customer features are enumerated; six of them are explicitly marked as categorical so CatBoost encodes them natively without manual one-hot encoding.

CatBoost Training

CatBoostClassifier is configured with 200 trees, depth=6, and Logloss; passing `cat_features` to `fit` tells the library which columns to handle with ordered target statistics instead of label encoding.

## Common Mistakes to Avoid

1. **Ignoring Data Quality**
   * Like cooking with spoiled ingredients
   * Can lead to poor predictions
   * Solution: Clean and validate data first
2. **Overfitting to Specific Cases**
   * Like memorizing recipes instead of learning to cook
   * Won't work well on new data
   * Solution: Use cross-validation
3. **Not Considering Business Context**
   * Like cooking without knowing who you're cooking for
   * Can lead to impractical solutions
   * Solution: Understand the real-world problem

## Next Steps

Ready to try these applications? Start with the credit risk example and gradually move to more complex projects. Remember, the key is to understand both the technical aspects and the real-world context!

## Gotchas

* **`StandardScaler` is fit on the full dataset, then applied to test data**: In `train_credit_model`, `scaler.fit_transform(X_train)` is correct but easy to accidentally replace with `scaler.fit_transform(X_test)` during debugging. Fitting the scaler on test data leaks test-set statistics into preprocessing and inflates reported performance.
* **The walk-forward stock predictor retrains a fresh model each step**: `model.fit(X_train, y_train)` inside the loop discards the previous model entirely and trains from scratch on each sliding window. This is correct for a strict walk-forward evaluation, but the loop is very slow on large datasets; warm-starting or caching model state would speed it up.
* **`cross_val_score` with a final `model.fit(X, y)` trains on the full data after cross-validation**: In `train_disease_predictor`, the CV scores evaluate generalization, but the returned model is re-fitted on all labels, including the test fold it was evaluated against. This is standard practice, but reporting the CV score _and_ the final model's training accuracy side by side will show inflated in-sample performance.
* **Operator precedence in the churn label rule produces surprising results**: `(data['tenure'] < 12) & (data['monthly_charges'] > 80) | (data['contract_type'] == 'Month-to-month')` is evaluated as `((A & B) | C)` due to Python's operator precedence. The intended logic may be `A & (B | C)`. Always add explicit parentheses when mixing `&` and `|` in pandas boolean conditions.
* **`yfinance` data may have NaNs from rolling windows**: `create_stock_features` calls `.dropna()` after computing rolling indicators like SMA-50. If the downloaded history is shorter than 50 days (e.g., newly listed stocks), `dropna` removes most of the data silently. Always check `len(df)` after the drop.
* **`pd.cut` risk buckets are sensitive to the chosen thresholds**: The `[0, 0.3, 0.6, 1]` bins in the churn prediction example are arbitrary. Changing the threshold from 0.3 to 0.4 can flip a large segment of customers between "Low" and "Medium" risk. Thresholds should be calibrated on actual business outcomes, not chosen by convention.

## Additional Resources

For more learning:

* [XGBoost Applications](https://xgboost.readthedocs.io/en/latest/tutorials/index.html)
* [LightGBM Use Cases](https://lightgbm.readthedocs.io/en/latest/Examples.html)
* [CatBoost Applications](https://catboost.ai/docs/concepts/use-cases)
* [Kaggle Competitions](https://www.kaggle.com/competitions)
