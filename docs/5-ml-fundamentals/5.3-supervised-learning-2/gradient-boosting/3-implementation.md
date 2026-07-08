---
reading_minutes: 25
objectives:
  - "Train an XGBoost classifier with `DMatrix`, `xgb.train`, and `early_stopping_rounds`, and read the eval log to spot the best round."
  - "Fit sklearn's `GradientBoostingRegressor` / `GradientBoostingClassifier` with sensible defaults (`learning_rate`, `n_estimators`, `max_depth`) and tune via `GridSearchCV`."
  - "Plot training vs validation loss curves to confirm early stopping triggered before overfitting."
---

# Implementing Gradient Boosting

**After this lesson:** you can explain Implementing Gradient Boosting and try the examples in your own notebook.

## Overview

Libraries and APIs (e.g. **hist** gradient boosting in sklearn, XGBoost/LightGBM/CatBoost at overview level): key parameters and fit patterns.


## Getting Started: Basic Implementation with XGBoost

### Classification Example: Spam Detection

Build a spam detector as our first project. This is like creating a filter that can tell if an email is spam or not.

Before we dive into the code, get clear on what we're trying to achieve:

1. We want to classify emails as spam or not spam
2. We'll use features like word counts and sender information
3. We'll train a model to learn patterns from these features

Here's how we'll do it step by step:

![Learning Curve](assets/learning_curve.png)

#### XGBoost: `DMatrix`, `train`, early stopping

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# First, import the tools we need
# Think of these as our kitchen utensils
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Now, create some sample email data
# This is like preparing our ingredients
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000,        # Number of emails
    n_features=20,         # Number of features per email
    n_informative=15,      # Number of useful features
    n_redundant=5,         # Number of redundant features
    random_state=42        # For reproducibility
)

# Split the data into training and testing sets
# This is like dividing our ingredients for practice and final cooking
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% for testing
    random_state=42
)

# Prepare the data for XGBoost
# This is like organizing our ingredients before cooking
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the model parameters
# These are like the settings on our cooking equipment
params = {
    'max_depth': 3,        # How deep each tree can grow
    'eta': 0.1,            # Learning rate (how fast it learns)
    'objective': 'binary:logistic',  # We're doing binary classification
    'eval_metric': 'logloss',        # How we measure success
    'nthread': 4           # Use 4 CPU cores
}

# Time to train our model!
# This is like cooking our dish
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,   # Number of trees to build
    evals=[(dtrain, 'train'), (dtest, 'test')],  # Track progress
    early_stopping_rounds=10,  # Stop if no improvement
    verbose_eval=False
)

# Test the model on new emails
# This is like tasting our dish
y_pred = model.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to 0/1

# Finally, look at how well we did
# This is like getting feedback on our cooking
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="11-18" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Informative vs redundant features</span>
    </div>
    <div class="code-callout__body">
      <p><code>make_classification</code> creates 15 features that genuinely predict the label and 5 redundant noise features, mimics real data where not all columns are useful. Gradient boosting handles this well via feature selection inside each tree.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-31" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">DMatrix: XGBoost's data format</span>
    </div>
    <div class="code-callout__body">
      <p><code>xgb.DMatrix</code> wraps numpy arrays in XGBoost's optimized internal format, attaching labels alongside features. Required before calling <code>xgb.train</code>, unlike sklearn's <code>.fit(X, y)</code>, XGBoost separates data preparation from training.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="33-41" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Boosting parameters</span>
    </div>
    <div class="code-callout__body">
      <p><code>eta</code> (learning rate) shrinks each new tree's contribution, lower values need more rounds but generalize better. <code>objective: binary:logistic</code> produces probabilities for a two-class problem. <code>eval_metric: logloss</code> measures calibration of those probabilities.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="43-52" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train with eval tracking</span>
    </div>
    <div class="code-callout__body">
      <p><code>evals</code> logs loss on both train and test each round. <code>early_stopping_rounds=10</code> halts boosting when test loss stops improving for 10 rounds, XGBoost automatically uses the best round's weights, avoiding over-boosting.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="54-57" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Probability → class label</span>
    </div>
    <div class="code-callout__body">
      <p><code>model.predict</code> returns a float probability per sample. <code>(y_pred > 0.5)</code> applies the default decision threshold, lower it (e.g. 0.3) to catch more positives at the cost of more false alarms.</p>
    </div>
  </div>
</aside>
</div>

Great job! You've just built your first spam detector. break down what we did:

1. We prepared our data (emails and their features)
2. We split the data for training and testing
3. We set up our model with appropriate parameters
4. We trained the model and evaluated its performance

Now that you understand the basics, try something a bit more complex: predicting house prices!

## LightGBM Implementation: House Price Prediction

Predicting house prices is like helping a real estate agent estimate property values. We'll use LightGBM, which is particularly good at handling large datasets efficiently.

Before coding, think through what you need:

1. Features like square footage, number of bedrooms, location, etc.
2. A way to measure how accurate our predictions are
3. A model that can learn from these features

Implement this step by step:

![Feature Importance](assets/feature_importance.png)

#### LightGBM regression: `Dataset`, RMSE, \\(R^2\\)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(
    n_samples=1000,
    n_features=20,
    noise=0.1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(10)]
)

y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-18" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and LightGBM Datasets</span>
    </div>
    <div class="code-callout__body">
      <p>1000 synthetic regression samples with 20 features are split 80/20; <code>lgb.Dataset</code> wraps the train and test arrays, the <code>reference=train_data</code> argument aligns the test set's feature histogram with the training set.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-37" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Params and Train</span>
    </div>
    <div class="code-callout__body">
      <p>A parameter dict sets the regression objective and RMSE metric; <code>feature_fraction=0.9</code> adds column subsampling for regularisation; <code>early_stopping(10)</code> halts training if validation RMSE has not improved for 10 rounds.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="39-41" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Evaluate</span>
    </div>
    <div class="code-callout__body">
      <p>RMSE measures the average prediction error in the original units; R² shows the fraction of variance explained, together they give a balanced picture of regression performance.</p>
    </div>
  </div>
</aside>
</div>

Excellent! You've now built a house price predictor. Notice how this implementation is similar to our spam detector but with some key differences:

1. We're predicting continuous values (prices) instead of categories
2. We're using different evaluation metrics
3. The data preparation is slightly different

Now, try something even more interesting: customer segmentation with CatBoost!

## CatBoost Implementation: Customer Segmentation

Customer segmentation is like grouping customers for targeted marketing. CatBoost is particularly good at handling categorical data, which is perfect for this task.

Before we start coding, get clear on what we're working with:

1. Customer data like age, income, education, and occupation
2. Categorical features that need special handling
3. A way to identify high-value customers

Implement this step by step:

![SHAP Values](assets/shap_values.png)

#### CatBoost: `Pool` + categorical feature indices

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.DataFrame({
    'age': np.random.normal(40, 10, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], 1000),
    'occupation': np.random.choice(['Tech', 'Finance', 'Healthcare'], 1000)
})

data['target'] = (
    (data['age'] > 35) &
    (data['income'] > 45000) |
    (data['education'].isin(['MS', 'PhD']))
).astype(int)

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

cat_features = ['education', 'occupation']
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=False
)
model.fit(train_pool, eval_set=test_pool)

y_pred = model.predict(test_pool)
print("Classification Report:")
print(classification_report(y_test, y_pred))
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-18" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data with Categoricals</span>
    </div>
    <div class="code-callout__body">
      <p>Two numeric and two categorical columns are generated; the binary target labels high-value customers using an age/income/education rule that mimics a real segmentation heuristic.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-43" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CatBoost Pool and Train</span>
    </div>
    <div class="code-callout__body">
      <p><code>Pool</code> bundles features, labels, and the categorical column indices so CatBoost can apply ordered target encoding natively; <code>eval_set</code> enables validation loss logging during training without a separate API call.</p>
    </div>
  </div>
</aside>
</div>

Great work! You've now built a customer segmentation model. Notice how CatBoost makes it easy to handle categorical data:

1. We specified which features are categorical
2. CatBoost automatically handles the encoding
3. The rest of the process is similar to our previous examples

Now, put everything together in a real-world example: predicting customer churn!

## Real-World Example: Customer Churn Prediction

Customer churn prediction is like having a crystal ball for customer retention. We'll use everything we've learned to build a practical system.

Before we start coding, get clear on what we're building:

1. A system that predicts which customers might leave
2. Features that help identify at-risk customers
3. A way to categorize customers by risk level

Implement this step by step:

![Customer Tenure Distribution](assets/churn_prediction.png)

#### Churn model: `fit` with `cat_features`, importance, risk bins

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    'tenure': np.random.normal(30, 15, 1000),
    'monthly_charges': np.random.normal(70, 20, 1000),
    'total_charges': np.random.normal(2000, 800, 1000),
    'contract_type': np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 1000
    ),
    'payment_method': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer'], 1000
    ),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 1000)
})

data['churn'] = (
    (data['tenure'] < 12) &
    (data['monthly_charges'] > 80) |
    (data['contract_type'] == 'Month-to-month')
).astype(int)

cat_features = ['contract_type', 'payment_method', 'internet_service']
X = data.drop('churn', axis=1)
y = data['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6,
                           loss_function='Logloss', verbose=False)
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)

y_prob = model.predict_proba(X_test)[:, 1]
risk_categories = pd.cut(y_prob, bins=[0, 0.3, 0.6, 1], labels=['Low', 'Medium', 'High'])

print("\nRisk Distribution:")
print(risk_categories.value_counts())
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-23" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data Generation</span>
    </div>
    <div class="code-callout__body">
      <p>Three numeric and three categorical columns simulate a telecom dataset; the churn label combines short tenure, high charges, and month-to-month contract, a realistic proxy for real churn signals.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-33" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train CatBoost</span>
    </div>
    <div class="code-callout__body">
      <p>Categorical columns are passed directly via <code>cat_features</code> without manual encoding; CatBoost uses ordered target statistics internally, avoiding target leakage.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="35-46" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Importance and Risk Bins</span>
    </div>
    <div class="code-callout__body">
      <p>Feature importances are ranked to surface the top churn drivers; <code>pd.cut</code> buckets predicted probabilities into Low/Medium/High tiers, enabling prioritised retention outreach.</p>
    </div>
  </div>
</aside>
</div>

You've now built a complete customer churn prediction system. Notice how we've combined everything we've learned:

1. We used CatBoost for handling categorical data
2. We analyzed feature importance to understand what drives churn
3. We created risk categories to help prioritize retention efforts

## Best Practices and Common Mistakes

Now that you've seen several implementations, review some best practices and common mistakes to avoid:

### 1. Data Preparation

- Always check for missing values
- Scale numerical features
- Handle categorical variables properly
- Remove irrelevant features

### 2. Model Tuning

- Start with default parameters
- Use cross-validation
- Tune one parameter at a time
- Keep track of changes

### 3. Evaluation

- Use appropriate metrics
- Check for overfitting
- Analyze feature importance
- Monitor training progress

## Common Mistakes to Avoid

1. **Using Too Many Trees**
   - Like studying the same material over and over
   - Can lead to overfitting
   - Solution: Use early stopping

2. **Ignoring Categorical Features**
   - Like not considering important customer segments
   - Can miss valuable patterns
   - Solution: Use proper encoding or CatBoost

3. **Skipping Feature Importance**
   - Like not learning from your mistakes
   - Miss insights about your data
   - Solution: Always analyze feature importance

## Next Steps

Ready to try these implementations? Start with the spam detection example and gradually move to more complex projects. Remember, practice makes perfect!

## Gotchas

- **Using the sklearn API vs the native XGBoost API interchangeably**: `xgb.train` (native) takes a `DMatrix` and a params dict; `XGBClassifier` (sklearn API) takes numpy arrays and uses `fit`. Mixing them (e.g., passing a `DMatrix` to `XGBClassifier.fit`) raises confusing type errors. Pick one API per project and stick with it.
- **`early_stopping_rounds` in the native XGBoost API uses the last entry in `evals`**: XGBoost monitors the *last* evaluation set passed to `evals` for early stopping. If you list `[(dtrain, 'train'), (dtest, 'test')]`, it correctly watches the test set. Reversing the order means early stopping fires on training loss and almost never stops.
- **LightGBM's `reference=train_data` in `Dataset` is not optional**: Passing `reference=train_data` when building the test `Dataset` ensures the two datasets share the same feature binning histogram. Omitting it can cause silent prediction drift, especially on categorical features.
- **CatBoost's `Pool` with `cat_features` expects column *names*, not integer indices, for DataFrames**, When your input is a pandas DataFrame, pass string column names to `cat_features`. Passing integer positions works for numpy arrays but silently misidentifies columns when a DataFrame has a non-default index.
- **`predict_proba` column ordering differs between sklearn and CatBoost**: In sklearn, `predict_proba(X)[:, 1]` gives P(positive class). In CatBoost, the column order depends on class ordering in the training labels. Always check `model.classes_` before slicing a specific column to avoid swapping the positive and negative class probabilities.
- **`scale_pos_weight` in XGBClassifier is not the same as SMOTE or resampling**: `scale_pos_weight` adjusts the gradient contribution of minority-class samples; it does not create new samples. For severe imbalance (>100:1), it helps but may still underperform proper resampling or threshold tuning on `predict_proba` output.

## Additional Resources

For more learning:

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Kaggle Gradient Boosting Tutorials](https://www.kaggle.com/learn/intro-to-deep-learning)
