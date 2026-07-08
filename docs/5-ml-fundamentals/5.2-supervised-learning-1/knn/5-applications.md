---
reading_minutes: 25
objectives:
  - "Reframe recommendation, image retrieval, and fraud detection as nearest-neighbour search over feature vectors."
  - "Combine scaling, dimensionality reduction, and weighted voting to make kNN viable on real datasets."
  - "Recognise when a domain is wrong for kNN (very high `p` with sparse text, latency-critical inference)."
---
# Real-World Applications of KNN: From Theory to Practice

**After this lesson:** you can explain Real-World Applications of KNN: From Theory to Practice and try the examples in your own notebook.

## Overview

Worked-style scenarios (recommendation-style similarity, small tabular problems) with evaluation caveats.


## Why Applications Matter

Understanding real-world applications helps you:

- See how KNN solves actual problems
- Learn when to use KNN vs. other algorithms
- Get ideas for your own projects
- Understand the practical challenges of implementing KNN

## 1. Movie Recommendation System

Imagine you're building a movie streaming service. You want to recommend movies to users based on what they've watched before.

### How It Works

#### Item-item `NearestNeighbors` on a ratings matrix

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class MovieRecommender:
    def __init__(self, k=5):
        """Initialize with k neighbors"""
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k+1)  # +1 because we include the movie itself

    def fit(self, ratings_matrix):
        """Train the recommender"""
        # Scale the ratings
        self.scaler = StandardScaler()
        ratings_scaled = self.scaler.fit_transform(ratings_matrix)

        # Train the model
        self.model.fit(ratings_scaled)
        self.ratings_matrix = ratings_matrix

    def recommend(self, movie_id, n_recommendations=5):
        """Get movie recommendations"""
        # Get the movie's features
        movie_features = self.ratings_matrix.iloc[movie_id].values.reshape(1, -1)
        movie_features_scaled = self.scaler.transform(movie_features)

        # Find similar movies
        distances, indices = self.model.kneighbors(
            movie_features_scaled,
            n_neighbors=n_recommendations+1
        )

        # Remove the movie itself and return recommendations
        similar_movies = indices[0][1:]
        similarity_scores = 1 - distances[0][1:]

        return list(zip(similar_movies, similarity_scores))

# Example: Building a Simple Recommender
ratings = pd.DataFrame({
    'user_1': [5, 3, 0, 4],  # User 1's ratings
    'user_2': [4, 0, 0, 5],  # User 2's ratings
    'user_3': [1, 1, 5, 2]   # User 3's ratings
}, index=['movie_1', 'movie_2', 'movie_3', 'movie_4'])

# Create and train the recommender
recommender = MovieRecommender()
recommender.fit(ratings)

# Get recommendations for movie_1
recommendations = recommender.recommend(0)
print("Recommended movies for movie_1:")
for movie_id, similarity in recommendations:
    print(f"Movie {movie_id + 1} (similarity: {similarity:.2f})")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>Pandas for the ratings DataFrame, <code>NearestNeighbors</code> for item-item similarity, and <code>StandardScaler</code> to normalize user rating scales.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit the Index</span>
    </div>
    <div class="code-callout__body">
      <p>Ratings are scaled so heavy raters don't dominate; the <code>NearestNeighbors</code> index is built on movie vectors (rows = movies, cols = users).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-39" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Recommend Movies</span>
    </div>
    <div class="code-callout__body">
      <p>The query movie is transformed, neighbors are retrieved, the first result (itself) is stripped, and similarity is computed as <code>1 - distance</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="41-57" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Usage Demo</span>
    </div>
    <div class="code-callout__body">
      <p>A 4×3 ratings matrix is built as a DataFrame; recommendations for movie_1 are retrieved and printed with their similarity scores.</p>
    </div>
  </div>
</aside>
</div>

## 2. Medical Diagnosis Assistant

KNN can illustrate how case-based medical screening might compare a new record with similar historical records. This example uses three synthetic patients, so treat it as a modelling pattern rather than a diagnostic tool.

### Building a Diagnosis System

#### `Pipeline` with scaling and distance-weighted KNN + `predict_proba`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

class MedicalDiagnosisSystem:
    def __init__(self, k=5):
        """Initialize the diagnosis system"""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(
                n_neighbors=k,
                weights='distance'  # Closer cases matter more
            ))
        ])

    def train(self, patient_data, diagnoses):
        """Train the system with past cases"""
        self.pipeline.fit(patient_data, diagnoses)

    def diagnose(self, patient_data):
        """Make a diagnosis with confidence level"""
        # Get prediction probabilities
        probabilities = self.pipeline.predict_proba(patient_data)

        # Get the diagnosis and confidence
        prediction = self.pipeline.predict(patient_data)
        confidence = np.max(probabilities, axis=1)

        return prediction, confidence

# Example: Diagnosing Patients
# Features: [temperature, heart_rate, blood_pressure, white_blood_cell_count]
X = np.array([
    [38.5, 90, 140, 11000],  # Patient with flu
    [37.0, 70, 120, 8000],   # Healthy patient
    [39.0, 95, 150, 15000],  # Patient with infection
])

y = ['flu', 'healthy', 'infection']

# Create and train the system
diagnosis_system = MedicalDiagnosisSystem()
diagnosis_system.train(X, y)

# Diagnose a new patient
new_patient = np.array([[38.2, 85, 135, 12000]])
diagnosis, confidence = diagnosis_system.diagnose(new_patient)
print(f"Diagnosis: {diagnosis[0]} (confidence: {confidence[0]:.2f})")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-15" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline Setup</span>
    </div>
    <div class="code-callout__body">
      <p>The sklearn Pipeline chains scaling then distance-weighted KNN; <code>weights='distance'</code> means clinically similar past cases carry more weight.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train on History</span>
    </div>
    <div class="code-callout__body">
      <p>A single <code>fit</code> call processes all historical patient records through the pipeline, scaling is fit on training vitals only.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-30" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Diagnose with Confidence</span>
    </div>
    <div class="code-callout__body">
      <p><code>predict_proba</code> returns probabilities for each diagnosis; <code>np.max</code> gives the highest probability as a confidence score for the final prediction.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="32-48" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Demo Prediction</span>
    </div>
    <div class="code-callout__body">
      <p>Three labeled patients train the system; a new patient with moderate vitals is diagnosed and the confidence level is printed.</p>
    </div>
  </div>
</aside>
</div>

## 3. Image Similarity Search

KNN can help find similar images, useful for photo organization or product search.

### Building an Image Finder

#### Pixel-vector index with `NearestNeighbors`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np

class ImageSimilarityFinder:
    def __init__(self, k=5):
        """Initialize the image finder"""
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k)

    def _preprocess_image(self, image):
        """Convert image to a feature vector"""
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

        # Return similar images and their similarity scores
        similar_images = [
            (self.image_paths[i], 1 - d)
            for i, d in zip(indices[0], distances[0])
        ]

        return similar_images

# Example usage
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
finder = ImageSimilarityFinder()
finder.fit(image_paths)

# Find images similar to a query image
similar_images = finder.find_similar('query_image.jpg')
print("Similar images found:")
for path, similarity in similar_images:
    print(f"{path} (similarity: {similarity:.2f})")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>PIL opens image files; <code>NearestNeighbors</code> stores and queries the pixel-vector index; NumPy handles array flattening.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Preprocess to Vector</span>
    </div>
    <div class="code-callout__body">
      <p>Each image is resized to 64×64 grayscale and flattened to a 4,096-element vector so all images have the same feature dimension.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-30" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Build Index</span>
    </div>
    <div class="code-callout__body">
      <p>All image paths are opened, preprocessed, and fitted into the <code>NearestNeighbors</code> model; the index stores pixel-space vectors for fast lookup.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="32-57" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Query Similarity</span>
    </div>
    <div class="code-callout__body">
      <p>The query image is preprocessed the same way; <code>kneighbors</code> returns distances converted to similarity scores (<code>1 - distance</code>) paired with file paths.</p>
    </div>
  </div>
</aside>
</div>

## 4. Fraud Detection System

KNN-style anomaly methods can illustrate how unusual transaction patterns might be ranked for review. This tiny example is a fraud-detection pattern, not evidence of production fraud performance.

### Building a Fraud Detector

#### `LocalOutlierFactor` for anomaly scores and ranked fraud cases

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

class FraudDetector:
    def __init__(self, contamination=0.1):
        """Initialize the fraud detector"""
        self.model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination
        )

    def detect(self, transaction_data):
        """Detect potential fraud"""
        # -1 for anomalies (potential fraud), 1 for normal transactions
        predictions = self.model.fit_predict(transaction_data)

        # Get anomaly scores
        scores = -self.model.negative_outlier_factor_

        return predictions, scores

    def analyze_findings(self, transaction_data, predictions, scores):
        """Analyze detected anomalies"""
        fraud_indices = np.where(predictions == -1)[0]

        results = []
        for idx in fraud_indices:
            results.append({
                'transaction_id': idx,
                'data': transaction_data[idx],
                'fraud_score': scores[idx]
            })

        return sorted(results, key=lambda x: x['fraud_score'],
                     reverse=True)

# Example: Detecting Credit Card Fraud
# Features: [amount, time, location, etc.]
transactions = np.array([
    [100, 10, 1],    # Normal transaction
    [150, 12, 1],    # Normal transaction
    [5000, 2, 3],    # Potential fraud
])

detector = FraudDetector()
predictions, scores = detector.detect(transactions)
fraud_cases = detector.analyze_findings(transactions, predictions, scores)

print("Potential fraud cases:")
for case in fraud_cases:
    print(f"Transaction {case['transaction_id']}: Score {case['fraud_score']:.2f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">LOF Setup</span>
    </div>
    <div class="code-callout__body">
      <p><code>LocalOutlierFactor</code> computes each point's local density relative to its 20 neighbors; <code>contamination</code> sets the expected fraud fraction.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Detect Anomalies</span>
    </div>
    <div class="code-callout__body">
      <p><code>fit_predict</code> labels each transaction as +1 (normal) or -1 (anomaly); negating <code>negative_outlier_factor_</code> gives a positive fraud score where higher = more suspicious.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-34" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Rank Findings</span>
    </div>
    <div class="code-callout__body">
      <p>Anomalous indices are collected, packed with their data and score, and sorted descending by fraud score so the most suspicious cases surface first.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="36-51" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Usage Demo</span>
    </div>
    <div class="code-callout__body">
      <p>Three transactions, two normal and one outlier, demonstrate the pipeline; the high-amount late-night transaction is flagged with a high fraud score.</p>
    </div>
  </div>
</aside>
</div>

```
Potential fraud cases:
Transaction 1: Score 1.01
```

## Common Challenges and Solutions

1. **Handling Large Datasets**

   #### `algorithm='ball_tree'` for faster neighbor queries

   ```python
   # Use ball tree for faster searches
   knn = KNeighborsClassifier(
       n_neighbors=5,
       algorithm='ball_tree',
       leaf_size=30
   )
   ```

2. **Dealing with Imbalanced Data**

   #### SMOTE oversampling before fitting KNN

   ```python
   from imblearn.over_sampling import SMOTE

   # Balance the classes
   smote = SMOTE()
   X_balanced, y_balanced = smote.fit_resample(X, y)
   ```

3. **Choosing the Right Distance Metric**

   #### Cosine vs Euclidean for different feature types

   ```python
   # For text data
   knn = KNeighborsClassifier(metric='cosine')

   # For numerical data
   knn = KNeighborsClassifier(metric='euclidean')
   ```

## Best Practices for Real-World Applications

1. **Always Preprocess Your Data**

   #### Standardize `X` before KNN

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Validate Your Model**

   #### 5-fold CV mean accuracy

   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(knn, X_scaled, y, cv=5)
   print(f"Average accuracy: {scores.mean():.3f}")
   ```

3. **Monitor Performance**

   #### Per-class metrics from `classification_report`

   ```python
   from sklearn.metrics import classification_report
   y_pred = knn.predict(X_test)
   print(classification_report(y_test, y_pred))
   ```

## Gotchas

- **Using `1 - distance` as a similarity score without normalizing distances**: The `MovieRecommender` and `ImageSimilarityFinder` examples compute similarity as `1 - distance`. This only makes intuitive sense if distances are bounded in [0, 1]. With Euclidean distance on unscaled features, `1 - distance` can be negative or exceed 1, producing scores with no meaningful interpretation.
- **Assuming LOF's `contamination` parameter has no effect on scores**: `LocalOutlierFactor(contamination=0.1)` sets the decision threshold for `fit_predict` labels, but `negative_outlier_factor_` scores are computed independently. Changing `contamination` changes which points are labelled `-1` without changing the underlying scores, learners often expect scores to shift, leading to confusion when re-fitting with different contamination values.
- **Scaling inside `fit` but forgetting to scale inference inputs the same way**: The `MedicalDiagnosisSystem` correctly puts `StandardScaler` inside a `Pipeline`. If instead the scaler is applied manually outside the class, it is easy to forget `scaler.transform(new_patient)` at prediction time, causing silently wrong diagnoses with no error raised.
- **Building the pixel-vector index on color images without fixing the channel count**: The `ImageSimilarityFinder` converts images to grayscale (single channel). If a query image has a different number of channels than the index images, `kneighbors` will raise a shape mismatch or silently compare wrong dimensions. Enforce consistent preprocessing for both indexed and query images.
- **Using `NearestNeighbors` with `n_neighbors=k+1` then forgetting to strip self**: The recommender sets `n_neighbors=k+1` to account for the fact that querying a movie against the index returns itself as the nearest neighbor. If you later increase `n_recommendations` without adjusting `n_neighbors`, you lose the self-exclusion buffer and get back one fewer real recommendation than expected.
- **Evaluating real-world KNN models with accuracy alone on imbalanced data**: Fraud datasets (like the LOF example) are highly skewed. A model that labels every transaction as normal can score 99% accuracy on a 1% fraud rate. Always pair accuracy with precision, recall, and AUC-ROC for anomaly and fraud detection tasks.

## Additional Resources

For more learning:

- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Real-World KNN Examples](https://www.kdnuggets.com/2020/04/most-popular-distance-metrics-knn.html)
- [KNN in Industry](https://towardsdatascience.com/knn-in-real-world-applications-5b3e0c5a0c5a)

Remember: The key to successful KNN applications is understanding your data and choosing the right parameters for your specific problem!
