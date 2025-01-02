# Real-World Applications of KNN 🌍

Let's explore how KNN is used in various industries and implement some practical examples.

## 1. Recommendation Systems 🎬

> **Recommendation Systems** suggest items to users based on similarities between users or items.

### Movie Recommender

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class MovieRecommender:
    def __init__(self, k=5):
        self.k = k
        self.model = NearestNeighbors(
            n_neighbors=k+1,  # +1 because item itself is included
            algorithm='ball_tree'
        )
        
    def fit(self, ratings_matrix):
        """Train the recommender system"""
        # Normalize ratings
        self.scaler = StandardScaler()
        ratings_scaled = self.scaler.fit_transform(ratings_matrix)
        
        # Fit model
        self.model.fit(ratings_scaled)
        self.ratings_matrix = ratings_matrix
        
    def recommend(self, movie_id, n_recommendations=5):
        """Get movie recommendations"""
        # Find similar movies
        movie_features = self.ratings_matrix.iloc[movie_id].values.reshape(1, -1)
        movie_features_scaled = self.scaler.transform(movie_features)
        
        # Get nearest neighbors
        distances, indices = self.model.kneighbors(
            movie_features_scaled,
            n_neighbors=n_recommendations+1
        )
        
        # Remove the movie itself
        similar_movies = indices[0][1:]
        similarity_scores = 1 - distances[0][1:]
        
        return list(zip(similar_movies, similarity_scores))

# Example usage
ratings = pd.DataFrame({
    'user_1': [5, 3, 0, 4],
    'user_2': [4, 0, 0, 5],
    'user_3': [1, 1, 5, 2]
}, index=['movie_1', 'movie_2', 'movie_3', 'movie_4'])

recommender = MovieRecommender()
recommender.fit(ratings)
recommendations = recommender.recommend(0)  # For movie_1
```

## 2. Medical Diagnosis 🏥

> **Medical Diagnosis Systems** help identify diseases based on patient symptoms and test results.

### Disease Classifier

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

class MedicalDiagnosisSystem:
    def __init__(self, k=5):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(
                n_neighbors=k,
                weights='distance'  # Weight by inverse distance
            ))
        ])
        
    def train(self, patient_data, diagnoses):
        """Train the diagnosis system"""
        self.pipeline.fit(patient_data, diagnoses)
        
    def diagnose(self, patient_data):
        """Make diagnosis prediction with confidence"""
        # Get prediction probabilities
        probabilities = self.pipeline.predict_proba(patient_data)
        
        # Get predicted class and confidence
        prediction = self.pipeline.predict(patient_data)
        confidence = np.max(probabilities, axis=1)
        
        return prediction, confidence

# Example usage
patient_features = [
    'temperature', 'heart_rate', 'blood_pressure', 
    'white_blood_cell_count'
]

X = np.array([
    [38.5, 90, 140, 11000],  # Patient 1
    [37.0, 70, 120, 8000],   # Patient 2
    [39.0, 95, 150, 15000],  # Patient 3
])

y = ['flu', 'healthy', 'infection']

diagnosis_system = MedicalDiagnosisSystem()
diagnosis_system.train(X, y)

new_patient = np.array([[38.2, 85, 135, 12000]])
diagnosis, confidence = diagnosis_system.diagnose(new_patient)
```

## 3. Image Recognition 🖼️

> **Image Recognition Systems** classify images based on their visual features.

### Image Similarity Search

```python
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np

class ImageSimilarityFinder:
    def __init__(self, k=5):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k)
        
    def _preprocess_image(self, image):
        """Convert image to feature vector"""
        # Resize for consistency
        image = image.resize((64, 64))
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        # Flatten to 1D array
        return np.array(image).flatten()
        
    def fit(self, image_paths):
        """Build index of images"""
        self.image_paths = image_paths
        features = []
        
        for path in image_paths:
            image = Image.open(path)
            features.append(self._preprocess_image(image))
            
        self.model.fit(features)
        
    def find_similar(self, query_image_path):
        """Find similar images"""
        # Process query image
        query_image = Image.open(query_image_path)
        query_features = self._preprocess_image(query_image)
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors([query_features])
        
        # Return similar image paths and scores
        similar_images = [
            (self.image_paths[i], 1 - d) 
            for i, d in zip(indices[0], distances[0])
        ]
        
        return similar_images
```

## 4. Anomaly Detection 🔍

> **Anomaly Detection Systems** identify unusual patterns that don't conform to expected behavior.

### Network Intrusion Detection

```python
from sklearn.neighbors import LocalOutlierFactor

class NetworkAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination
        )
        
    def fit_predict(self, network_data):
        """Detect anomalies in network traffic"""
        # -1 for anomalies, 1 for normal instances
        predictions = self.model.fit_predict(network_data)
        
        # Get anomaly scores
        scores = -self.model.negative_outlier_factor_
        
        return predictions, scores
        
    def analyze_anomalies(self, network_data, predictions, scores):
        """Analyze detected anomalies"""
        anomaly_indices = np.where(predictions == -1)[0]
        
        results = []
        for idx in anomaly_indices:
            results.append({
                'index': idx,
                'data': network_data[idx],
                'anomaly_score': scores[idx]
            })
            
        return sorted(results, key=lambda x: x['anomaly_score'], 
                     reverse=True)

# Example usage
network_features = [
    'bytes_sent', 'bytes_received', 'connection_duration',
    'num_connections', 'port_number'
]

data = np.array([
    [1000, 2000, 30, 5, 80],    # Normal traffic
    [900, 1800, 25, 4, 443],    # Normal traffic
    [50000, 100, 2, 50, 1234],  # Potential anomaly
])

detector = NetworkAnomalyDetector()
predictions, scores = detector.fit_predict(data)
anomalies = detector.analyze_anomalies(data, predictions, scores)
```

## 5. Financial Applications 💰

### Credit Risk Assessment

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

class CreditRiskAssessor:
    def __init__(self, k=7):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(
                n_neighbors=k,
                weights='distance'
            ))
        ])
        
    def train_evaluate(self, X, y):
        """Train and evaluate model using cross-validation"""
        scores = cross_val_score(
            self.pipeline, X, y,
            cv=5, scoring='roc_auc'
        )
        
        print(f"ROC-AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Train final model
        self.pipeline.fit(X, y)
        
    def assess_risk(self, applicant_data):
        """Assess credit risk for new applicant"""
        # Get risk probabilities
        risk_proba = self.pipeline.predict_proba(applicant_data)
        
        # Get nearest neighbors for explanation
        classifier = self.pipeline.named_steps['classifier']
        neighbors = classifier.kneighbors(
            self.pipeline.named_steps['scaler'].transform(applicant_data),
            return_distance=True
        )
        
        return {
            'risk_probability': risk_proba[0][1],
            'similar_cases': neighbors
        }
```

## Deployment Best Practices 🚀

### 1. Model Versioning

```python
import joblib
import json
from datetime import datetime

class ModelManager:
    @staticmethod
    def save_model(model, metadata, path):
        """Save model with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"{path}/model_{timestamp}.joblib"
        meta_path = f"{path}/metadata_{timestamp}.json"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        with open(meta_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'model_type': str(type(model)),
                'parameters': model.get_params(),
                **metadata
            }, f)
            
    @staticmethod
    def load_latest_model(path):
        """Load most recent model"""
        import glob
        import os
        
        # Find latest model file
        model_files = glob.glob(f"{path}/model_*.joblib")
        latest_model = max(model_files, key=os.path.getctime)
        
        # Load model and metadata
        model = joblib.load(latest_model)
        meta_file = latest_model.replace('model_', 'metadata_')
        meta_file = meta_file.replace('.joblib', '.json')
        
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
        """Log prediction and actual value"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(datetime.now())
        
    def analyze_performance(self, window_size=100):
        """Analyze recent performance"""
        if len(self.predictions) < window_size:
            return
            
        recent_preds = self.predictions[-window_size:]
        recent_actuals = [a for a in self.actuals[-window_size:] 
                         if a is not None]
        
        if recent_actuals:
            accuracy = sum(p == a for p, a in 
                         zip(recent_preds, recent_actuals)) / len(recent_actuals)
            print(f"Recent accuracy: {accuracy:.3f}")
```

## Next Steps 🎯

After exploring these applications:
1. Choose a specific domain to focus on
2. Gather relevant datasets
3. Implement and test solutions
4. Deploy and monitor performance

Remember:
- Start with simple implementations
- Validate assumptions with domain experts
- Monitor and update models regularly
- Consider ethical implications of applications
