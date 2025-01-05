# Introduction to Python for Data Science

## Overview

Welcome to Python! Whether you're new to programming or coming from another language, Python is an excellent choice for your data science journey. Let's understand why Python has become the de facto language for data science and analytics.

{% stepper %}
{% step %}
### What is Python?
Python is a high-level, interpreted programming language that emphasizes code readability with its notable use of significant whitespace. In data science, Python serves as your Swiss Army knife for:
- Data analysis and manipulation
- Statistical computations
- Machine learning model development
- Data visualization
- Automated reporting
- ETL (Extract, Transform, Load) processes

**Quick Example**:
```python
# Simple data analysis in Python
import pandas as pd
import matplotlib.pyplot as plt

# Read and analyze data
data = pd.read_csv('sales_data.csv')
monthly_sales = data.groupby('month')['sales'].sum()

# Create visualization
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='bar')
plt.title('Monthly Sales Performance')
plt.show()
```
{% endstep %}

{% step %}
### Why Python for Data Science?
Python stands out because it:
- **Readable Syntax**: Compare these examples:
  ```python
  # Python
  sales_data = pd.read_csv('sales.csv')
  average_sales = sales_data['amount'].mean()
  
  # R equivalent
  sales_data <- read.csv('sales.csv')
  average_sales <- mean(sales_data$amount)
  ```

- **Rich Ecosystem**: Essential data science libraries:
  ```python
  import numpy as np        # Numerical computations
  import pandas as pd       # Data manipulation
  import matplotlib.pyplot as plt  # Visualization
  import seaborn as sns    # Statistical visualization
  import scikit-learn as sklearn  # Machine learning
  ```

- **Integration Capabilities**: Connect with various data sources:
  ```python
  # Database connection
  from sqlalchemy import create_engine
  engine = create_engine('postgresql://user:pass@localhost:5432/db')
  
  # API integration
  import requests
  api_data = requests.get('https://api.example.com/data').json()
  
  # Big Data processing
  import pyspark
  from pyspark.sql import SparkSession
  ```
{% endstep %}

{% step %}
### Python in Industry
Real-world applications across industries:

**1. Finance**
```python
# Stock price analysis
import yfinance as yf

# Get Tesla stock data
tesla = yf.Ticker("TSLA")
history = tesla.history(period="1y")

# Calculate moving average
history['MA50'] = history['Close'].rolling(window=50).mean()
```

**2. Healthcare**
```python
# Patient data analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Predict patient readmission risk
def predict_readmission(patient_data):
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        patient_data.drop('readmitted', axis=1),
        patient_data['readmitted']
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)
```

**3. E-commerce**
```python
# Customer segmentation
from sklearn.cluster import KMeans

def segment_customers(customer_data):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=4)
    segments = kmeans.fit_predict(customer_data)
    return segments
```

**4. Marketing**
```python
# Social media sentiment analysis
from textblob import TextBlob

def analyze_sentiment(tweets):
    sentiments = []
    for tweet in tweets:
        analysis = TextBlob(tweet)
        sentiments.append(analysis.sentiment.polarity)
    return pd.Series(sentiments).mean()
```
{% endstep %}

{% step %}
### Modern Python Features
Latest Python features that enhance data science work:

**1. Type Hints (Python 3.5+)**
```python
from typing import List, Dict
import pandas as pd

def process_sales_data(
    data: pd.DataFrame,
    columns: List[str]
) -> Dict[str, float]:
    """Process sales data with type hints"""
    return {
        'total': data[columns].sum().to_dict(),
        'average': data[columns].mean().to_dict()
    }
```

**2. Walrus Operator (Python 3.8+)**
```python
# Efficient data processing
if (n_rows := len(df)) > 1000:
    print(f"Processing {n_rows} rows in batches")
    process_in_batches(df)
```

**3. Pattern Matching (Python 3.10+)**
```python
def analyze_data_point(point):
    match point:
        case {'type': 'sales', 'amount': amount} if amount > 1000:
            return 'high_value_sale'
        case {'type': 'refund', 'amount': amount}:
            return 'refund_case'
        case _:
            return 'standard_transaction'
```
{% endstep %}
{% endstepper %}

## Advantages for Data Scientists

{% stepper %}
{% step %}
### Efficient Data Analysis
```python
# Quick data exploration
import pandas as pd
import seaborn as sns

def quick_eda(df: pd.DataFrame) -> None:
    """Perform quick exploratory data analysis"""
    # Basic statistics
    print("Basic Statistics:")
    print(df.describe())
    
    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Matrix")
    plt.show()
```
{% endstep %}

{% step %}
### Data Visualization
```python
# Advanced visualization example
import plotly.express as px

def create_interactive_dashboard(sales_data: pd.DataFrame) -> None:
    """Create interactive sales dashboard"""
    # Sales trend
    fig1 = px.line(sales_data, x='date', y='amount',
                   title='Sales Trend Over Time')
    
    # Category distribution
    fig2 = px.pie(sales_data, values='amount', names='category',
                  title='Sales by Category')
    
    # Geographic distribution
    fig3 = px.scatter_mapbox(sales_data, lat='latitude', lon='longitude',
                            size='amount', color='category',
                            title='Sales Geographic Distribution')
    
    # Display dashboard
    fig1.show()
    fig2.show()
    fig3.show()
```
{% endstep %}

{% step %}
### Machine Learning Integration
```python
# End-to-end ML pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def create_ml_pipeline():
    """Create a machine learning pipeline"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
```
{% endstep %}
{% endstepper %}

## What You'll Learn

This chapter will take you through Python fundamentals with a data science focus:

{% stepper %}
{% step %}
### 1. Basic Syntax and Data Types
Learn the building blocks with data science context:
```python
# Numbers for statistical calculations
mean_value = 42.5
count = 100

# Strings for data cleaning
text_data = "  Customer Feedback  "
cleaned_text = text_data.strip().lower()

# Booleans for filtering
is_valid = True
has_missing_values = False
```
{% endstep %}

{% step %}
### 2. Data Structures
Master Python's data structures for data manipulation:
```python
# Lists for time series
stock_prices = [100.0, 101.5, 102.3, 101.7]

# Dictionaries for feature mapping
feature_mapping = {
    'age': 'numeric',
    'gender': 'categorical',
    'income': 'numeric'
}

# Sets for unique values
unique_categories = {'electronics', 'clothing', 'food'}

# Tuples for immutable data
data_point = (42.0, 'positive', True)
```
{% endstep %}

{% step %}
### 3. Control Flow
Learn flow control with data processing examples:
```python
# Data filtering
def filter_outliers(data, threshold):
    clean_data = []
    for value in data:
        if abs(value - mean(data)) < threshold:
            clean_data.append(value)
    return clean_data

# Data transformation
def process_transactions(transactions):
    while transactions:
        transaction = transactions.pop()
        if transaction['amount'] > 1000:
            flag_for_review(transaction)
        elif transaction['amount'] < 0:
            process_refund(transaction)
        else:
            process_normal(transaction)
```
{% endstep %}

{% step %}
### 4. Functions
Create reusable data analysis components:
```python
def calculate_metrics(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistical metrics
    
    Parameters:
        data: List of numerical values
    
    Returns:
        Dictionary of calculated metrics
    """
    return {
        'mean': statistics.mean(data),
        'median': statistics.median(data),
        'std_dev': statistics.stdev(data),
        'range': max(data) - min(data)
    }
```
{% endstep %}

{% step %}
### 5. Object-Oriented Programming
Learn to organize data science code:
```python
class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.original_shape = data.shape
    
    def remove_missing_values(self) -> pd.DataFrame:
        """Remove rows with missing values"""
        return self.data.dropna()
    
    def standardize_columns(self) -> pd.DataFrame:
        """Standardize column names"""
        self.data.columns = [
            col.lower().replace(' ', '_')
            for col in self.data.columns
        ]
        return self.data
    
    def get_cleaning_report(self) -> Dict:
        """Generate cleaning report"""
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': len(self.data),
            'removed_rows': self.original_shape[0] - len(self.data)
        }
```
{% endstep %}

{% step %}
### 6. Modules and Packages
Learn to use and create data science packages:
```python
# Custom data science utility module
# data_utils.py
import pandas as pd
import numpy as np
from typing import List, Dict

class DataAnalyzer:
    @staticmethod
    def summarize_numeric(data: pd.Series) -> Dict:
        """Summarize numeric column"""
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'skew': data.skew()
        }
    
    @staticmethod
    def summarize_categorical(data: pd.Series) -> Dict:
        """Summarize categorical column"""
        return {
            'unique_values': data.nunique(),
            'mode': data.mode()[0],
            'frequencies': data.value_counts().to_dict()
        }

# Using the module
from data_utils import DataAnalyzer
analyzer = DataAnalyzer()
numeric_summary = analyzer.summarize_numeric(df['sales'])
```
{% endstep %}
{% endstepper %}

## Ready to Start?

Remember these data science best practices:
- Always explore your data before analysis
- Document your code with clear comments
- Use version control (git) for your projects
- Write modular, reusable code
- Test your functions with sample data
- Consider performance for large datasets

Let's begin your Python data science journey! 🚀
