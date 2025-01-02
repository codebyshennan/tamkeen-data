# Real-World Applications of Random Forest 🌍

Let's explore how Random Forests are used to solve real-world problems across different industries!

## 1. Financial Applications 💰

### Credit Risk Assessment
```python
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
```

### Stock Price Prediction
```python
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
```

## 2. Healthcare Applications 🏥

### Disease Prediction
```python
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
```

## 3. Environmental Applications 🌱

### Climate Change Analysis
```python
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
```

## 4. Marketing Applications 📊

### Customer Churn Prediction
```python
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
```

## 5. Manufacturing Applications 🏭

### Quality Control
```python
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
```

## Best Practices for Applications 🌟

1. **Data Quality**
   - Clean and validate input data
   - Handle missing values appropriately
   - Scale features when necessary

2. **Model Monitoring**
   - Track prediction accuracy over time
   - Monitor feature importance stability
   - Set up alerts for performance degradation

3. **Deployment Considerations**
   - Use model versioning
   - Implement A/B testing
   - Set up automated retraining

4. **Performance Optimization**
   - Use feature selection
   - Tune hyperparameters
   - Consider ensemble methods

## Next Steps 🚀

Congratulations! You've completed the Random Forest section. Continue exploring other advanced algorithms in the course!
