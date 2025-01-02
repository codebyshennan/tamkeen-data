# Real-World Applications of Regularization 🌍

Let's explore how regularization is used to solve real-world problems across different industries!

## 1. Financial Applications 💰

### Credit Risk Assessment
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create sample credit data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, n_samples),
    'age': np.random.normal(40, 10, n_samples),
    'employment_length': np.random.normal(8, 4, n_samples),
    'debt_ratio': np.random.uniform(0.1, 0.6, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples),
    'previous_defaults': np.random.randint(0, 3, n_samples)
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

# Train regularized logistic regression
model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    C=0.1  # Inverse of regularization strength
)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

## 2. Healthcare Applications 🏥

### Disease Prediction
```python
def predict_disease_risk(patient_data):
    """Predict disease risk using regularized model"""
    # Create features
    features = pd.DataFrame({
        'age': [patient_data['age']],
        'bmi': [patient_data['bmi']],
        'blood_pressure': [patient_data['bp']],
        'cholesterol': [patient_data['chol']],
        'glucose': [patient_data['glucose']],
        'smoking': [patient_data['smoking']],
        'family_history': [patient_data['family_history']]
    })
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict probability
    risk_prob = model.predict_proba(features_scaled)[0, 1]
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': features.columns,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    return {
        'risk_probability': risk_prob,
        'risk_factors': importance.head(3)
    }
```

## 3. Marketing Applications 📊

### Customer Churn Prediction
```python
def analyze_churn_factors():
    """Analyze factors contributing to customer churn"""
    # Create features
    features = [
        'usage_decline',
        'support_calls',
        'payment_delay',
        'competitor_offers',
        'service_issues',
        'contract_length',
        'total_spend'
    ]
    
    # Train elastic net
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)
    
    # Analyze coefficients
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_
    }).sort_values('coefficient', ascending=False)
    
    return coef_df
```

## 4. Real Estate Applications 🏠

### House Price Prediction
```python
def predict_house_price(features):
    """Predict house price with regularized model"""
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=0.1))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make prediction
    price = pipeline.predict([features])[0]
    
    # Get feature importance
    importance = abs(pipeline.named_steps['model'].coef_)
    
    return {
        'predicted_price': price,
        'key_factors': pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    }
```

## 5. Environmental Applications 🌱

### Climate Change Analysis
```python
def analyze_climate_factors():
    """Analyze factors affecting climate change"""
    # Features
    features = [
        'co2_levels',
        'methane_levels',
        'deforestation_rate',
        'industrial_emissions',
        'renewable_energy_usage',
        'ocean_temperature',
        'arctic_ice_coverage'
    ]
    
    # Train model with cross-validation
    model = LassoCV(cv=5)
    model.fit(X_train, y_train)
    
    # Analyze important factors
    factors = pd.DataFrame({
        'factor': features,
        'impact': model.coef_
    }).sort_values('impact', ascending=False)
    
    return factors
```

## 6. Sports Analytics ⚽

### Player Performance Prediction
```python
def predict_player_performance():
    """Predict player performance with regularization"""
    # Features
    features = [
        'previous_performance',
        'training_intensity',
        'rest_days',
        'injury_history',
        'age',
        'experience',
        'team_chemistry'
    ]
    
    # Create and train model
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        cv=5
    )
    model.fit(X_train, y_train)
    
    return {
        'predictions': model.predict(X_test),
        'key_factors': pd.DataFrame({
            'factor': features,
            'importance': abs(model.coef_)
        }).sort_values('importance', ascending=False)
    }
```

## Best Practices for Applications 🌟

### 1. Feature Engineering
```python
def engineer_features(data):
    """Create domain-specific features"""
    # Create interaction terms
    data['income_per_age'] = data['income'] / data['age']
    data['debt_to_income'] = data['debt'] / data['income']
    
    # Create polynomial features
    data['age_squared'] = data['age'] ** 2
    
    # Create categorical interactions
    data['high_risk'] = (
        (data['debt_ratio'] > 0.5) & 
        (data['credit_score'] < 600)
    ).astype(int)
    
    return data
```

### 2. Model Selection
```python
def select_best_regularization(X, y):
    """Select best regularization method"""
    models = {
        'ridge': Ridge(),
        'lasso': Lasso(),
        'elastic': ElasticNet()
    }
    
    # Perform cross-validation
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=5, scoring='r2'
        )
        cv_scores[name] = scores.mean()
    
    return cv_scores
```

### 3. Monitoring and Maintenance
```python
def monitor_model_performance(model, X, y, window=30):
    """Monitor model performance over time"""
    predictions = []
    actuals = []
    
    for i in range(len(X)):
        # Make prediction
        pred = model.predict([X.iloc[i]])[0]
        actual = y.iloc[i]
        
        predictions.append(pred)
        actuals.append(actual)
        
        # Calculate metrics over window
        if i >= window:
            window_rmse = np.sqrt(mean_squared_error(
                actuals[-window:],
                predictions[-window:]
            ))
            
            if window_rmse > threshold:
                print(f"Warning: High RMSE detected: {window_rmse}")
```

## Next Steps 🚀

Congratulations! You've completed the Regularization section. Continue exploring other advanced algorithms in the course!
