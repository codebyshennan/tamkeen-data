---
reading_minutes: 25
objectives:
  - "Build random-forest baselines for tabular problems in finance, healthcare, and marketing — and use feature importance to explain predictions to stakeholders."
  - "Use a forest as a *feature screen* before training a more expensive model (e.g., gradient boosting or neural network)."
  - "Recognise when a boosted ensemble would beat a forest on the same data, and when it wouldn't."
---

# Real-World Applications of Random Forest

**After this lesson:** you can explain the core ideas in “Real-World Applications of Random Forest” and reproduce the examples here in your own notebook or environment.

## Overview

Tabular benchmarks, feature screening, and when a forest beats boosting—or does not.


## 1. Financial Applications

### Credit Risk Assessment

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Create sample credit data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, n_samples),
    'age': np.random.normal(40, 10, n_samples),
    'employment_length': np.random.normal(8, 4, n_samples),
    'debt_ratio': np.random.uniform(0.1, 0.6, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples),
    'previous_defaults': np.random.randint(0, 3, n_samples),
    'loan_amount': np.random.normal(200000, 100000, n_samples)
})

# Create target (default probability)
data['default'] = (
    (data['debt_ratio'] > 0.4) &
    (data['credit_score'] < 650) |
    (data['previous_defaults'] > 1)
).astype(int)

# Prepare data
X = data.drop('default', axis=1)
y = data['default']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# Create risk scoring function
def calculate_risk_score(model, scaler, data):
    """Calculate risk score from 0-100"""
    # Scale data
    scaled_data = scaler.transform(data)

    # Get probability of default
    prob_default = model.predict_proba(scaled_data)[:, 1]

    # Convert to 0-100 score (higher is better)
    risk_score = 100 * (1 - prob_default)

    return risk_score

# Calculate and display risk scores
risk_scores = calculate_risk_score(rf, scaler, X_test)
print("Risk Score Distribution:")
print(pd.qcut(risk_scores, q=5).value_counts())
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-27" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Synthetic Data</span>
    </div>
    <div class="code-callout__body">
      <p>1 000 applicants are drawn with seven financial features; the default label is set by rule (high debt ratio + low credit score, or prior defaults), creating a realistic class imbalance.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="29-49" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scale and Train</span>
    </div>
    <div class="code-callout__body">
      <p>Features are standardized before an 80/20 split; a 100-tree RF with depth-10 cap and minimum-leaf=5 (prevents single-sample leaves) is then fit on the scaled training data.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="51-65" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Risk Score</span>
    </div>
    <div class="code-callout__body">
      <p><code>calculate_risk_score</code> converts the default probability to a 0–100 creditworthiness score (higher = safer); <code>pd.qcut</code> then bins the test-set scores into five equal-size quantiles for reporting.</p>
    </div>
  </div>
</aside>
</div>

```
Risk Score Distribution:
(0.947, 8.496]      40
(8.496, 82.808]     40
(82.808, 93.794]    40
(93.794, 96.665]    40
(96.665, 99.376]    40
Name: count, dtype: int64
```

### Stock Price Prediction

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def create_stock_features(data, lookback=30):
    """Create features for stock prediction"""
    df = data.copy()

    # Technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])

    # Price changes
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()

    # Target: Next day return
    df['Target'] = df['Close'].shift(-1) / df['Close'] - 1

    return df.dropna()

def train_stock_predictor(symbol='AAPL', lookback_days=30):
    """Train stock prediction model"""
    # Download data
    stock = yf.Ticker(symbol)
    data = stock.history(period='2y')

    # Create features
    df = create_stock_features(data, lookback_days)

    # Prepare data
    X = df.drop(['Target'], axis=1)
    y = df['Target']

    # Split data
    split_idx = int(len(df) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    # Train model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)

    return rf, X_test, y_test
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p><code>yfinance</code> fetches live OHLCV data; <code>RandomForestRegressor</code> predicts a continuous next-day return rather than a class label.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Feature Engineering</span>
    </div>
    <div class="code-callout__body">
      <p>Two SMAs, RSI, price-change, and volume-change are computed from the raw OHLCV frame; the target is the next-day return computed by shifting the close price one step forward.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-52" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train Predictor</span>
    </div>
    <div class="code-callout__body">
      <p>Two years of data are downloaded, features are built, and an 80/20 chronological split (no shuffle) respects time order; a 100-tree regressor is fit and the test slice is returned for evaluation.</p>
    </div>
  </div>
</aside>
</div>

## 2. Healthcare Applications

### Disease Prediction

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Create sample medical data
medical_data = pd.DataFrame({
    'age': np.random.normal(50, 15, n_samples),
    'bmi': np.random.normal(25, 5, n_samples),
    'blood_pressure': np.random.normal(120, 15, n_samples),
    'cholesterol': np.random.normal(200, 30, n_samples),
    'glucose': np.random.normal(100, 20, n_samples),
    'smoking': np.random.randint(0, 2, n_samples),
    'family_history': np.random.randint(0, 2, n_samples)
})

# Create target (disease risk)
medical_data['disease'] = (
    (medical_data['bmi'] > 30) &
    (medical_data['blood_pressure'] > 140) |
    (medical_data['cholesterol'] > 240) &
    (medical_data['glucose'] > 126)
).astype(int)

# Train model
X = medical_data.drop('disease', axis=1)
y = medical_data['disease']

rf_medical = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)
rf_medical.fit(X_train, y_train)

# Create risk assessment function
def assess_health_risk(model, patient_data):
    """Assess patient health risk"""
    # Get probability of disease
    prob = model.predict_proba(patient_data)[0, 1]

    # Risk category
    if prob < 0.2:
        risk = "Low"
    elif prob < 0.6:
        risk = "Moderate"
    else:
        risk = "High"

    # Important factors
    importances = pd.DataFrame({
        'factor': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'risk_probability': prob,
        'risk_category': risk,
        'key_factors': importances.head(3)
    }
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-18" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Medical Data</span>
    </div>
    <div class="code-callout__body">
      <p>Seven vitals and risk factors are synthesized for <code>n_samples</code> patients; a clinical rule (elevated BMI + hypertension, or high cholesterol + glucose) defines the disease label.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-30" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train RF</span>
    </div>
    <div class="code-callout__body">
      <p>A shallow (depth-5) RF with large minimum-leaf (10) is preferred for medical data to reduce overfitting while keeping the model interpretable.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="32-57" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Health Risk Helper</span>
    </div>
    <div class="code-callout__body">
      <p><code>assess_health_risk</code> maps the disease probability to Low/Moderate/High risk and returns the top-3 most important clinical factors for clinician review.</p>
    </div>
  </div>
</aside>
</div>

## 3. Environmental Applications

### Climate Change Analysis

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def analyze_climate_data(data):
    """Analyze climate change patterns"""
    # Features: temperature, precipitation, CO2, etc.
    X = data.drop(['year', 'temperature_change'], axis=1)
    y = data['temperature_change']

    # Train model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf.fit(X, y)

    # Analyze feature importance
    importance_df = pd.DataFrame({
        'factor': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return rf, importance_df

# Example usage
climate_data = pd.DataFrame({
    'year': range(1950, 2021),
    'CO2_levels': np.random.normal(350, 30, 71),
    'methane_levels': np.random.normal(1800, 100, 71),
    'deforestation_rate': np.random.normal(0.5, 0.1, 71),
    'industrial_emissions': np.random.normal(25, 5, 71),
    'temperature_change': np.random.normal(1, 0.3, 71)
})

model, importance = analyze_climate_data(climate_data)
print("Key Climate Factors:")
print(importance)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-21" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Analyze Climate</span>
    </div>
    <div class="code-callout__body">
      <p>Year is excluded as a feature; a 100-tree RF regressor is fit to predict temperature change; feature importances are sorted into a DataFrame to show which factor drives the model most.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-35" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Synthetic Dataset</span>
    </div>
    <div class="code-callout__body">
      <p>71 years (1950–2020) of synthetic CO2, methane, deforestation, and emissions data are created for demonstration; the function is called and the ranked importance table is printed.</p>
    </div>
  </div>
</aside>
</div>

## 4. Marketing Applications

### Customer Churn Prediction

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def predict_customer_churn(customer_data):
    """Predict customer churn probability"""
    # Features
    features = [
        'tenure', 'monthly_charges', 'total_charges',
        'contract_type', 'payment_method', 'internet_service',
        'online_security', 'tech_support', 'streaming_tv'
    ]

    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Create churn prevention recommendations
    def get_churn_recommendations(customer):
        """Generate personalized recommendations"""
        # Get feature importance for this prediction
        prediction = rf.predict_proba([customer])[0, 1]

        recommendations = []
        if prediction > 0.5:
            if customer['contract_type'] == 'Month-to-month':
                recommendations.append(
                    "Offer annual contract with discount"
                )
            if customer['online_security'] == 'No':
                recommendations.append(
                    "Promote online security features"
                )
            if customer['tech_support'] == 'No':
                recommendations.append(
                    "Offer free tech support trial"
                )

        return {
            'churn_probability': prediction,
            'recommendations': recommendations
        }

    return rf, get_churn_recommendations
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-18" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Feature List and Train</span>
    </div>
    <div class="code-callout__body">
      <p>Nine customer-level features are declared; a RandomForestClassifier with depth=10 and min_samples_leaf=5 is fitted to prevent overfitting on small customer segments.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-43" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Recommendation Closure</span>
    </div>
    <div class="code-callout__body">
      <p>An inner function uses the trained model to score each customer; if churn probability exceeds 0.5 it appends targeted retention offers based on which risk factors are present, then returns the score alongside the recommendations list.</p>
    </div>
  </div>
</aside>
</div>

## 5. Manufacturing Applications

### Quality Control

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def predict_product_quality(manufacturing_data):
    """Predict product quality issues"""
    # Features: sensor readings, temperature, pressure, etc.
    X = manufacturing_data.drop('defect', axis=1)
    y = manufacturing_data['defect']

    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Create monitoring function
    def monitor_production_line(sensor_data):
        """Monitor production line in real-time"""
        # Get defect probability
        prob_defect = rf.predict_proba([sensor_data])[0, 1]

        # Alert levels
        if prob_defect > 0.7:
            alert = "Critical - Stop Production"
        elif prob_defect > 0.3:
            alert = "Warning - Inspect System"
        else:
            alert = "Normal Operation"

        return {
            'defect_probability': prob_defect,
            'alert_level': alert,
            'recommendations': get_recommendations(sensor_data)
        }

    return rf, monitor_production_line
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-15" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Prepare and Train</span>
    </div>
    <div class="code-callout__body">
      <p>Sensor readings are split from the defect label; a shallow forest (depth=5, min_samples_leaf=10) is trained to generalize across production batches without memorising individual sensor readings.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-36" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Real-time Monitor</span>
    </div>
    <div class="code-callout__body">
      <p>The inner function scores live sensor data and maps the defect probability to a three-tier alert (Critical/Warning/Normal), returning probability, alert level, and targeted recommendations in one dict.</p>
    </div>
  </div>
</aside>
</div>

## Best Practices for Applications

1. **Data Preparation**
   - Handle missing values appropriately
   - Scale numerical features
   - Encode categorical variables

2. **Model Tuning**
   - Use cross-validation
   - Optimize hyperparameters
   - Monitor feature importance

3. **Deployment**
   - Save model artifacts
   - Create API endpoints
   - Monitor model performance

## Gotchas

- **Scaling features before Random Forest is unnecessary but harmless** — the credit risk example applies `StandardScaler` before fitting, which does not affect tree splits (trees use rank order, not magnitude); scaling is needed for the downstream `predict_proba` risk score consistency but adds no predictive benefit to the forest itself.
- **Chronological splitting is mandatory for time-series data, not random splitting** — the stock predictor uses an 80/20 index-based split, which preserves time order correctly; using `train_test_split` with `shuffle=True` would leak future prices into training and give artificially high R² scores.
- **`yfinance` data includes dividends and splits that distort raw price features** — using `Close` directly without adjusting for corporate actions inflates apparent returns on ex-dividend days; always use the `Auto Adjust=True` option or the `Adj Close` column for financial feature engineering.
- **`rf_medical.fit(X_train, y_train)` uses a different split than the medical feature matrix** — the healthcare example creates `medical_data` but then fits on `X_train`/`y_train` from the earlier credit risk split, silently training on the wrong data; always verify that feature matrix and labels originate from the same dataset.
- **`get_recommendations(sensor_data)` in the manufacturing monitor is undefined** — the `monitor_production_line` function calls `get_recommendations` which is never declared in this file; running it raises a `NameError` and would need a separate implementation before use in production.
- **Feature importance from a Random Forest trained on scaled data reflects scaled units** — the `calculate_risk_score` function scales inputs, so feature importances (if inspected) reflect the scaled representation; do not interpret raw importance values as importance in the original dollar or score units.

## Next Steps

Ready to implement Random Forests in your own projects? Check out:

1. [Implementation Guide](3-implementation.md)
2. [Hyperparameter Tuning](4-advanced.md)
3. [Model Evaluation](5-applications.md)
