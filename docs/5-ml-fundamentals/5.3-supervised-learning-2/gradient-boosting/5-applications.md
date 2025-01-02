# Real-World Applications of Gradient Boosting 🌍

Let's explore how Gradient Boosting is used to solve real-world problems across different industries!

## 1. Financial Applications 💰

### Credit Risk Assessment
```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Train credit risk model
def train_credit_model(data):
    """Train credit risk assessment model"""
    # Prepare data
    X = data.drop('default', axis=1)
    y = data['default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Create risk scoring function
def calculate_credit_risk(model, scaler, applicant_data):
    """Calculate credit risk score and recommendations"""
    # Scale data
    scaled_data = scaler.transform(applicant_data)
    
    # Get probability of default
    default_prob = model.predict_proba(scaled_data)[:, 1]
    
    # Calculate credit score (inverse of default probability)
    credit_score = 100 * (1 - default_prob)
    
    # Get feature importance for this prediction
    feature_imp = pd.DataFrame({
        'feature': applicant_data.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Generate recommendations
    recommendations = []
    if default_prob > 0.3:
        if applicant_data['debt_ratio'].values[0] > 0.4:
            recommendations.append("Reduce debt ratio")
        if applicant_data['credit_score'].values[0] < 650:
            recommendations.append("Improve credit score")
    
    return {
        'credit_score': credit_score[0],
        'default_probability': default_prob[0],
        'key_factors': feature_imp.head(3),
        'recommendations': recommendations
    }
```

### Stock Market Prediction
```python
import yfinance as yf
from lightgbm import LGBMRegressor

def create_stock_features(data, lookback=30):
    """Create technical indicators"""
    df = data.copy()
    
    # Price-based indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Volatility indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
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
    
    # Train model
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5
    )
    
    # Implement walk-forward optimization
    predictions = []
    train_size = 252  # One year of trading days
    
    for i in range(train_size, len(df)):
        # Get training data
        train_data = df.iloc[i-train_size:i]
        X_train = train_data.drop(['Target'], axis=1)
        y_train = train_data['Target']
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make prediction
        X_test = df.iloc[i:i+1].drop(['Target'], axis=1)
        pred = model.predict(X_test)
        predictions.append(pred[0])
    
    return model, predictions
```

## 2. Healthcare Applications 🏥

### Disease Risk Prediction
```python
def train_disease_predictor(medical_data):
    """Train disease risk prediction model"""
    # Prepare features
    features = [
        'age', 'bmi', 'blood_pressure', 'cholesterol',
        'glucose', 'smoking', 'family_history'
    ]
    
    X = medical_data[features]
    y = medical_data['disease']
    
    # Train model with cross-validation
    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100
    )
    
    # Use stratified k-fold
    cv_scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(5),
        scoring='roc_auc'
    )
    
    # Train final model
    model.fit(X, y)
    
    return model

def assess_patient_risk(model, patient_data):
    """Assess patient's disease risk"""
    # Get probability
    risk_prob = model.predict_proba(patient_data)[0, 1]
    
    # Risk category
    if risk_prob < 0.2:
        risk_level = "Low"
    elif risk_prob < 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    # Get feature importance
    importance = pd.DataFrame({
        'factor': patient_data.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Generate recommendations
    recommendations = []
    if risk_prob > 0.3:
        if patient_data['smoking'].values[0] == 1:
            recommendations.append("Stop smoking")
        if patient_data['bmi'].values[0] > 30:
            recommendations.append("Reduce BMI")
        if patient_data['blood_pressure'].values[0] > 140:
            recommendations.append("Control blood pressure")
    
    return {
        'risk_probability': risk_prob,
        'risk_level': risk_level,
        'key_factors': importance.head(3),
        'recommendations': recommendations
    }
```

## 3. Marketing Applications 📊

### Customer Churn Prediction
```python
from catboost import CatBoostClassifier

def predict_customer_churn(customer_data):
    """Predict customer churn probability"""
    # Prepare features
    features = [
        'tenure', 'monthly_charges', 'total_charges',
        'contract_type', 'payment_method', 'internet_service',
        'online_security', 'tech_support', 'streaming_tv'
    ]
    
    # Specify categorical features
    cat_features = [
        'contract_type', 'payment_method', 'internet_service',
        'online_security', 'tech_support', 'streaming_tv'
    ]
    
    # Train model
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        verbose=False
    )
    
    model.fit(
        customer_data[features],
        customer_data['churn'],
        cat_features=cat_features
    )
    
    return model

def generate_retention_strategies(model, customer):
    """Generate personalized retention strategies"""
    # Predict churn probability
    churn_prob = model.predict_proba(customer)[0, 1]
    
    # Get feature importance for this prediction
    importance = model.get_feature_importance(
        type='ShapValues',
        data=customer
    )
    
    # Generate recommendations
    recommendations = []
    if churn_prob > 0.5:
        if customer['contract_type'].values[0] == 'Month-to-month':
            recommendations.append({
                'action': "Offer annual contract",
                'discount': "20% off for first year"
            })
        if customer['online_security'].values[0] == 'No':
            recommendations.append({
                'action': "Add online security",
                'discount': "Free for 3 months"
            })
        if customer['tech_support'].values[0] == 'No':
            recommendations.append({
                'action': "Add tech support",
                'discount': "50% off for 6 months"
            })
    
    return {
        'churn_probability': churn_prob,
        'risk_level': 'High' if churn_prob > 0.5 else 'Low',
        'key_factors': importance,
        'recommendations': recommendations
    }
```

## 4. Manufacturing Applications 🏭

### Quality Control
```python
def predict_manufacturing_defects(sensor_data):
    """Predict manufacturing defects from sensor data"""
    # Create features from sensor readings
    features = [
        'temperature', 'pressure', 'vibration',
        'humidity', 'power_consumption', 'noise_level'
    ]
    
    # Train model
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        class_weight='balanced'
    )
    
    model.fit(
        sensor_data[features],
        sensor_data['defect']
    )
    
    return model

def monitor_production_line(model, current_readings):
    """Monitor production line in real-time"""
    # Get defect probability
    defect_prob = model.predict_proba(current_readings)[0, 1]
    
    # Set alert levels
    if defect_prob > 0.7:
        alert = "Critical - Stop Production"
        color = "red"
    elif defect_prob > 0.3:
        alert = "Warning - Inspect System"
        color = "yellow"
    else:
        alert = "Normal Operation"
        color = "green"
    
    # Get contributing factors
    importance = pd.DataFrame({
        'factor': current_readings.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'defect_probability': defect_prob,
        'alert_status': alert,
        'alert_color': color,
        'key_factors': importance.head(3)
    }
```

## Best Practices for Applications 🌟

1. **Data Quality**
   - Validate input data
   - Handle missing values
   - Scale features appropriately
   - Handle categorical variables

2. **Model Monitoring**
   - Track prediction accuracy
   - Monitor feature distributions
   - Set up alerts for drift
   - Regular model retraining

3. **Deployment**
   - Version control models
   - A/B testing
   - Gradual rollout
   - Fallback strategies

4. **Performance**
   - Optimize inference time
   - Batch predictions
   - GPU acceleration
   - Model compression

## Next Steps 🚀

Congratulations! You've completed the Gradient Boosting section. Continue exploring other advanced algorithms in the course!
