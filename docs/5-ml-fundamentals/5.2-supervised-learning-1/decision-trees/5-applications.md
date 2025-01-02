# Real-World Applications of Decision Trees 🌍

Let's explore how decision trees are used in various industries with practical examples and deployment strategies.

## 1. Medical Diagnosis 🏥

> **Medical Decision Support Systems** help doctors make diagnostic decisions by following a tree-like sequence of questions.

### Disease Diagnosis System

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class MedicalDiagnosisSystem:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight='balanced'
        )
        
    def preprocess_symptoms(self, symptoms_data):
        """Process patient symptoms"""
        # Convert categorical symptoms to numerical
        symptom_map = {
            'none': 0,
            'mild': 1,
            'moderate': 2,
            'severe': 3
        }
        return np.array([
            [symptom_map[s] for s in patient]
            for patient in symptoms_data
        ])
        
    def train(self, symptoms, diagnoses):
        """Train the diagnosis system"""
        X = self.preprocess_symptoms(symptoms)
        self.model.fit(X, diagnoses)
        
    def explain_diagnosis(self, patient_symptoms):
        """Provide explanation for diagnosis"""
        # Get decision path
        path = self.model.decision_path(
            self.preprocess_symptoms([patient_symptoms])
        )
        
        # Extract rules
        feature_names = [
            'fever', 'cough', 'fatigue', 
            'breathing', 'blood_pressure'
        ]
        rules = []
        
        for node_id in path.indices:
            if node_id != self.model.tree_.node_count - 1:
                feature = self.model.tree_.feature[node_id]
                threshold = self.model.tree_.threshold[node_id]
                
                if patient_symptoms[feature] <= threshold:
                    rules.append(
                        f"{feature_names[feature]} <= {threshold}"
                    )
                else:
                    rules.append(
                        f"{feature_names[feature]} > {threshold}"
                    )
                    
        return rules
```

## 2. Credit Risk Assessment 💰

### Loan Approval System

```python
class LoanApprovalSystem:
    def __init__(self):
        self.model = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('num', StandardScaler(), 
                 ['income', 'debt', 'credit_score']),
                ('cat', OneHotEncoder(), 
                 ['employment_type', 'loan_purpose'])
            ])),
            ('classifier', DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=100,
                class_weight='balanced'
            ))
        ])
        
    def train_with_monitoring(self, X, y):
        """Train with performance monitoring"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Monitor performance
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob[:, 1])
        }
        
        return metrics
        
    def assess_risk(self, application):
        """Assess loan application risk"""
        # Get probability
        prob = self.model.predict_proba([application])[0]
        
        # Define risk levels
        if prob[1] < 0.2:
            risk = 'Low'
        elif prob[1] < 0.6:
            risk = 'Medium'
        else:
            risk = 'High'
            
        return {
            'risk_level': risk,
            'approval_probability': 1 - prob[1],
            'requires_review': 0.4 <= prob[1] <= 0.6
        }
```

## 3. Customer Churn Prediction 👥

### Churn Prevention System

```python
class ChurnPredictor:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50
        )
        self.feature_names = None
        
    def prepare_features(self, customer_data):
        """Prepare customer features"""
        # Calculate derived features
        customer_data['tenure_months'] = (
            customer_data['end_date'] - 
            customer_data['start_date']
        ).dt.total_seconds() / (30 * 24 * 60 * 60)
        
        customer_data['avg_monthly_spend'] = (
            customer_data['total_spend'] / 
            customer_data['tenure_months']
        )
        
        # Select features
        features = [
            'tenure_months', 'avg_monthly_spend',
            'support_calls', 'product_usage'
        ]
        
        self.feature_names = features
        return customer_data[features]
        
    def identify_risk_factors(self, customer):
        """Identify factors contributing to churn risk"""
        # Get feature importance for this prediction
        path = self.model.decision_path([customer])
        
        risk_factors = []
        for node_id in path.indices:
            if node_id != self.model.tree_.node_count - 1:
                feature = self.model.tree_.feature[node_id]
                threshold = self.model.tree_.threshold[node_id]
                
                if customer[feature] > threshold:
                    risk_factors.append({
                        'feature': self.feature_names[feature],
                        'value': customer[feature],
                        'threshold': threshold
                    })
                    
        return sorted(
            risk_factors,
            key=lambda x: abs(x['value'] - x['threshold']),
            reverse=True
        )
```

## 4. Fraud Detection 🔍

### Transaction Monitoring System

```python
class FraudDetector:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', RobustScaler()),
            ('tree', DecisionTreeClassifier(
                max_depth=6,
                class_weight={0: 1, 1: 10}  # Fraud is class 1
            ))
        ])
        
    def extract_features(self, transaction):
        """Extract features from transaction"""
        features = {
            'amount': transaction['amount'],
            'time_of_day': transaction['timestamp'].hour,
            'day_of_week': transaction['timestamp'].dayofweek,
            'distance_from_home': self._calculate_distance(
                transaction['location'],
                transaction['home_location']
            ),
            'frequency': self._get_frequency(
                transaction['user_id'],
                transaction['merchant']
            )
        }
        return features
        
    def predict_with_threshold(self, transaction, threshold=0.8):
        """Make prediction with custom threshold"""
        # Get probability
        prob = self.model.predict_proba([transaction])[0]
        
        return {
            'is_fraud': prob[1] >= threshold,
            'confidence': prob[1],
            'needs_review': 0.6 <= prob[1] < threshold
        }
```

## 5. Equipment Maintenance 🔧

### Predictive Maintenance System

```python
class MaintenancePredictor:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=20
        )
        
    def process_sensor_data(self, sensor_readings):
        """Process sensor data for prediction"""
        features = {
            'temperature_mean': np.mean(sensor_readings['temp']),
            'temperature_std': np.std(sensor_readings['temp']),
            'vibration_max': np.max(sensor_readings['vibration']),
            'pressure_change': (
                sensor_readings['pressure'].iloc[-1] -
                sensor_readings['pressure'].iloc[0]
            ),
            'runtime_hours': len(sensor_readings) / 60
        }
        return features
        
    def predict_maintenance(self, equipment_data):
        """Predict maintenance needs"""
        features = self.process_sensor_data(equipment_data)
        
        # Make prediction
        needs_maintenance = self.model.predict([features])[0]
        
        if needs_maintenance:
            # Get decision path for explanation
            path = self.model.decision_path([features])
            rules = self._extract_rules(path, features)
            
            return {
                'needs_maintenance': True,
                'reason': rules,
                'urgency': 'high' if features['temperature_std'] > 10 else 'normal'
            }
        
        return {'needs_maintenance': False}
```

## Deployment Best Practices 🚀

### 1. Model Versioning

```python
class ModelManager:
    def __init__(self, base_path='models/'):
        self.base_path = base_path
        
    def save_model(self, model, metadata):
        """Save model with version control"""
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f"{self.base_path}tree_{version}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        meta_path = f"{self.base_path}metadata_{version}.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'version': version,
                'timestamp': str(datetime.now()),
                'metrics': metadata
            }, f)
```

### 2. Performance Monitoring

```python
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        
    def log_prediction(self, prediction, actual=None):
        """Log prediction details"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        
    def analyze_performance(self, window_size=1000):
        """Analyze recent performance"""
        if len(self.predictions) < window_size:
            return
            
        recent_preds = self.predictions[-window_size:]
        recent_actuals = [a for a in self.actuals[-window_size:]
                         if a is not None]
        
        if recent_actuals:
            return {
                'accuracy': accuracy_score(
                    recent_actuals,
                    recent_preds
                ),
                'timestamp': datetime.now()
            }
```

## Next Steps 🎯

After exploring these applications:
1. Choose a specific domain
2. Start with simple models
3. Gradually add complexity
4. Monitor and improve

Remember:
- Validate domain assumptions
- Test thoroughly
- Monitor performance
- Update models regularly
