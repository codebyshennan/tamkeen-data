# Real-World Applications of SVM 🌍

Let's explore how SVM is used in various industries with practical examples and deployment strategies.

## 1. Text Classification 📝

> **Text Classification** involves categorizing text documents into predefined categories.

### Spam Detection System

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np

class EmailClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', LinearSVC(
                C=1.0,
                class_weight='balanced',
                max_iter=1000
            ))
        ])
        
    def train(self, emails, labels):
        """Train the spam detector"""
        self.pipeline.fit(emails, labels)
        
    def predict(self, emails, threshold=0.8):
        """Predict with confidence threshold"""
        # Get decision function scores
        scores = self.pipeline.decision_function(emails)
        
        # Apply threshold
        predictions = np.where(
            np.abs(scores) >= threshold,
            np.sign(scores),
            0  # Mark as uncertain
        )
        
        return predictions, np.abs(scores)

# Example usage
emails = [
    "Get rich quick! Buy now!",
    "Meeting at 3pm tomorrow",
    "Win a free iPhone today!",
    "Project deadline reminder"
]
labels = [1, 0, 1, 0]  # 1 for spam, 0 for not spam

classifier = EmailClassifier()
classifier.train(emails, labels)
```

## 2. Image Recognition 🖼️

### Face Detection System

```python
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 
            'haarcascade_frontalface_default.xml'
        )
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True))
        ])
        
    def extract_features(self, image):
        """Extract features from face image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        resized = cv2.resize(gray, (64, 64))
        
        # Extract HOG features
        hog = cv2.HOGDescriptor()
        features = hog.compute(resized)
        
        return features.flatten()
        
    def train(self, face_images, labels):
        """Train the face detector"""
        # Extract features from all images
        features = np.array([
            self.extract_features(img)
            for img in face_images
        ])
        
        # Train model
        self.model.fit(features, labels)
        
    def detect(self, image, min_confidence=0.8):
        """Detect faces in image"""
        # Detect faces using cascade
        faces = self.face_cascade.detectMultiScale(
            image, 
            scaleFactor=1.1,
            minNeighbors=5
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face = image[y:y+h, x:x+w]
            
            # Get features
            features = self.extract_features(face)
            
            # Predict
            prob = self.model.predict_proba([features])[0]
            if max(prob) >= min_confidence:
                results.append({
                    'bbox': (x, y, w, h),
                    'confidence': max(prob)
                })
                
        return results
```

## 3. Medical Diagnosis 🏥

### Disease Classification

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

class MedicalDiagnosisSystem:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced'
            ))
        ])
        
    def train_with_validation(self, X, y):
        """Train with cross-validation"""
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=5)
        
        # Store validation results
        self.cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Split data
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            
            self.cv_results.append({
                'fold': fold,
                'auc': auc
            })
            
        # Train final model on all data
        self.model.fit(X, y)
        
    def diagnose(self, patient_data, threshold=0.5):
        """Make diagnosis with confidence"""
        # Get probability
        prob = self.model.predict_proba([patient_data])[0]
        
        # Make decision
        diagnosis = 1 if prob[1] >= threshold else 0
        
        return {
            'diagnosis': diagnosis,
            'confidence': prob[1],
            'needs_review': 0.4 <= prob[1] <= 0.6
        }
```

## 4. Financial Applications 💰

### Credit Risk Assessment

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

class CreditRiskAssessor:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced'
            ))
        ])
        
    def train_with_monitoring(self, X, y):
        """Train with performance monitoring"""
        # Get cross-validated predictions
        y_pred = cross_val_predict(
            self.model, X, y,
            cv=5,
            method='predict_proba'
        )
        
        # Calculate metrics
        self.performance_metrics = {
            'auc': roc_auc_score(y, y_pred[:, 1]),
            'precision': precision_score(
                y, y_pred[:, 1] > 0.5
            ),
            'recall': recall_score(
                y, y_pred[:, 1] > 0.5
            )
        }
        
        # Train final model
        self.model.fit(X, y)
        
    def assess_risk(self, applicant_data):
        """Assess credit risk"""
        # Get probability
        prob = self.model.predict_proba([applicant_data])[0]
        
        # Define risk levels
        if prob[1] < 0.2:
            risk_level = 'Low'
        elif prob[1] < 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
            
        return {
            'risk_level': risk_level,
            'probability': prob[1],
            'auto_approve': prob[1] < 0.2,
            'auto_reject': prob[1] > 0.8
        }
```

## 5. Anomaly Detection 🔍

### Network Intrusion Detection

```python
from sklearn.svm import OneClassSVM
import numpy as np

class NetworkMonitor:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('detector', OneClassSVM(
                kernel='rbf',
                nu=0.1  # Contamination rate
            ))
        ])
        
    def train_on_normal(self, normal_traffic):
        """Train on normal network traffic"""
        self.model.fit(normal_traffic)
        
    def detect_anomalies(self, traffic_data):
        """Detect anomalous traffic"""
        # Get anomaly scores
        scores = self.model.decision_function(traffic_data)
        
        # Detect anomalies
        predictions = self.model.predict(traffic_data)
        
        # Analyze results
        anomalies = []
        for i, (score, pred) in enumerate(zip(scores, predictions)):
            if pred == -1:  # Anomaly
                anomalies.append({
                    'index': i,
                    'score': abs(score),
                    'data': traffic_data[i]
                })
                
        return sorted(
            anomalies,
            key=lambda x: x['score'],
            reverse=True
        )
```

## Deployment Best Practices 🚀

### 1. Model Versioning

```python
import joblib
import json
from datetime import datetime

class ModelManager:
    def __init__(self, base_path='models/'):
        self.base_path = base_path
        
    def save_model(self, model, metadata):
        """Save model with version control"""
        # Create version ID
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f"{self.base_path}model_{version}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        meta_path = f"{self.base_path}metadata_{version}.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'version': version,
                'timestamp': str(datetime.now()),
                'performance': metadata
            }, f)
            
    def load_latest_model(self):
        """Load most recent model"""
        import glob
        import os
        
        # Find latest version
        model_files = glob.glob(
            f"{self.base_path}model_*.joblib"
        )
        latest = max(model_files, key=os.path.getctime)
        
        # Load model and metadata
        model = joblib.load(latest)
        meta_file = latest.replace(
            'model_', 'metadata_'
        ).replace('.joblib', '.json')
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata
```

### 2. Performance Monitoring

```python
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
    def log_prediction(self, prediction, actual=None):
        """Log prediction details"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(datetime.now())
        
    def analyze_performance(self, window_size=1000):
        """Analyze recent performance"""
        if len(self.predictions) < window_size:
            return
            
        recent_preds = self.predictions[-window_size:]
        recent_actuals = [a for a in self.actuals[-window_size:]
                         if a is not None]
        
        if recent_actuals:
            metrics = {
                'accuracy': accuracy_score(
                    recent_actuals,
                    recent_preds
                ),
                'timestamp': datetime.now()
            }
            
            return metrics
```

## Next Steps 🎯

After exploring these applications:
1. Choose a specific domain
2. Start with small-scale implementation
3. Gradually add complexity
4. Monitor and improve

Remember:
- Always validate assumptions
- Test thoroughly before deployment
- Monitor performance in production
- Update models regularly
