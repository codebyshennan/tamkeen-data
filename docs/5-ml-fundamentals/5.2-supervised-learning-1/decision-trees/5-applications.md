# Real-World Applications of Decision Trees

## 1. Medical Diagnosis System

Imagine you're building a system to help doctors diagnose patients. Decision trees are perfect for this because they're easy to understand and explain.

### Step-by-Step Implementation

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class MedicalDiagnosisSystem:
    def __init__(self):
        """Create a diagnosis system"""
        self.model = DecisionTreeClassifier(
            max_depth=5,          # Keep it simple for doctors
            min_samples_leaf=10,  # Need enough cases to be confident
            class_weight='balanced'  # Handle rare diseases
        )
        
    def preprocess_symptoms(self, symptoms_data):
        """Convert symptoms to numbers"""
        # Map symptom severity to numbers
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
        """Show how the diagnosis was made"""
        # Get the decision path
        path = self.model.decision_path(
            self.preprocess_symptoms([patient_symptoms])
        )
        
        # List of symptoms
        feature_names = [
            'fever', 'cough', 'fatigue', 
            'breathing', 'blood_pressure'
        ]
        
        # Extract the rules used
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

### Example Usage

```python
# Create the system
diagnosis_system = MedicalDiagnosisSystem()

# Train with example data
symptoms = [
    ['severe', 'moderate', 'mild', 'none', 'normal'],
    ['none', 'none', 'none', 'none', 'normal'],
    # ... more cases
]
diagnoses = ['flu', 'healthy', 'pneumonia', 'healthy']

diagnosis_system.train(symptoms, diagnoses)

# Make a diagnosis
new_patient = ['moderate', 'mild', 'severe', 'none', 'high']
rules = diagnosis_system.explain_diagnosis(new_patient)
print("Diagnosis Rules:")
for rule in rules:
    print(f"- {rule}")
```

## 2. Loan Approval System

Banks use decision trees to decide whether to approve loans. Let's build a simple version:

```python
class LoanApprovalSystem:
    def __init__(self):
        """Create a loan approval system"""
        self.model = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('num', StandardScaler(), 
                 ['income', 'debt', 'credit_score']),
                ('cat', OneHotEncoder(), 
                 ['employment_type', 'loan_purpose'])
            ])),
            ('classifier', DecisionTreeClassifier(
                max_depth=4,          # Keep it simple
                min_samples_leaf=100, # Need enough cases
                class_weight='balanced'  # Handle rare defaults
            ))
        ])
        
    def train_with_monitoring(self, X, y):
        """Train and monitor performance"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Check performance
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
        """Evaluate loan application"""
        # Get probability of default
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

## 3. Customer Churn Prediction

Businesses use decision trees to predict which customers might leave:

```python
class ChurnPredictor:
    def __init__(self):
        """Create a churn prediction system"""
        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50
        )
        self.feature_names = None
        
    def prepare_features(self, customer_data):
        """Calculate important features"""
        # Calculate time as customer
        customer_data['tenure_months'] = (
            customer_data['end_date'] - 
            customer_data['start_date']
        ).dt.total_seconds() / (30 * 24 * 60 * 60)
        
        # Calculate average spending
        customer_data['avg_monthly_spend'] = (
            customer_data['total_spend'] / 
            customer_data['tenure_months']
        )
        
        # Select features to use
        features = [
            'tenure_months', 'avg_monthly_spend',
            'support_calls', 'product_usage'
        ]
        
        self.feature_names = features
        return customer_data[features]
        
    def identify_risk_factors(self, customer):
        """Find why customer might leave"""
        # Get decision path
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

## 4. Fraud Detection System

Banks and credit card companies use decision trees to detect fraudulent transactions:

```python
class FraudDetector:
    def __init__(self):
        """Create a fraud detection system"""
        self.model = Pipeline([
            ('scaler', RobustScaler()),  # Handle outliers
            ('tree', DecisionTreeClassifier(
                max_depth=6,
                class_weight={0: 1, 1: 10}  # Fraud is rare
            ))
        ])
        
    def extract_features(self, transaction):
        """Calculate fraud indicators"""
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
        """Make fraud prediction"""
        # Get probability
        prob = self.model.predict_proba([transaction])[0]
        
        return {
            'is_fraud': prob[1] >= threshold,
            'confidence': prob[1],
            'needs_review': 0.6 <= prob[1] < threshold
        }
```

## 5. Equipment Maintenance Predictor

Factories use decision trees to predict when machines need maintenance:

```python
class MaintenancePredictor:
    def __init__(self):
        """Create a maintenance prediction system"""
        self.model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=20
        )
        
    def prepare_sensor_data(self, raw_data):
        """Process sensor readings"""
        # Calculate statistics
        features = {
            'avg_temperature': raw_data['temperature'].mean(),
            'max_vibration': raw_data['vibration'].max(),
            'pressure_change': raw_data['pressure'].diff().mean(),
            'runtime_hours': raw_data['runtime'].sum()
        }
        return features
        
    def predict_maintenance(self, equipment_data):
        """Predict maintenance needs"""
        features = self.prepare_sensor_data(equipment_data)
        prediction = self.model.predict([features])[0]
        
        if prediction == 'urgent':
            return {
                'action': 'Immediate maintenance required',
                'confidence': self.model.predict_proba([features])[0][1],
                'recommended_parts': self._get_recommended_parts(features)
            }
        elif prediction == 'soon':
            return {
                'action': 'Schedule maintenance within 2 weeks',
                'confidence': self.model.predict_proba([features])[0][1],
                'recommended_parts': self._get_recommended_parts(features)
            }
        else:
            return {
                'action': 'No maintenance needed',
                'confidence': self.model.predict_proba([features])[0][0]
            }
```

## Best Practices for Real-World Applications

1. **Data Quality**
   - Clean and preprocess data carefully
   - Handle missing values appropriately
   - Deal with outliers

2. **Model Validation**
   - Use cross-validation
   - Monitor performance metrics
   - Test with real-world data

3. **Interpretability**
   - Keep trees simple
   - Document decision rules
   - Provide explanations

4. **Maintenance**
   - Regular model updates
   - Monitor performance drift
   - Update with new data

## Common Challenges and Solutions

1. **Imbalanced Data**
   - Use class weights
   - Try different sampling techniques
   - Adjust decision thresholds

2. **Overfitting**
   - Limit tree depth
   - Use pruning
   - Regularize parameters

3. **Feature Importance**
   - Select relevant features
   - Remove redundant features
   - Consider feature interactions

## Next Steps

Ready to build your own application? Try:

1. Start with a simple problem
2. Collect and clean your data
3. Build and test your model
4. Deploy and monitor

Remember:

- Start simple
- Validate thoroughly
- Document everything
- Monitor performance
