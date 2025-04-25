# Data Structures for Data Analysis

## Introduction to Data Structures

{% stepper %}
{% step %}

### Data Structures in Data Science

Each data structure serves specific purposes in data analysis:

```python
import numpy as np
import pandas as pd

# Lists: Time series data
stock_prices = [100.23, 101.45, 99.78, 102.34]

# Tuples: Fixed structure records
data_point = ('2023-01-01', 'AAPL', 173.57, 1000000)

# Sets: Unique categories
unique_symbols = {'AAPL', 'GOOGL', 'MSFT', 'AMZN'}

# Dictionaries: Feature mappings
feature_info = {
    'price': {'type': 'numeric', 'missing': 0.02},
    'volume': {'type': 'numeric', 'missing': 0.00},
    'sector': {'type': 'categorical', 'unique_values': 11}
}

# NumPy Arrays: Efficient numerical computations
prices_array = np.array(stock_prices)

# Pandas Series: Labeled data
prices_series = pd.Series(stock_prices, 
                         index=pd.date_range('2023-01-01', periods=4))
```

Each structure optimized for different operations:

- Lists for flexible data collection
- Tuples for immutable records
- Sets for unique value operations
- Dictionaries for key-based lookups
- NumPy arrays for numerical computations
- Pandas for labeled data analysis
{% endstep %}

{% step %}

### Performance Considerations

Choose structures based on operation needs:

```python
import time
import numpy as np

# Comparing list vs. numpy array operations
def compare_performance(size=1000000):
    # Create data
    list_data = list(range(size))
    array_data = np.array(list_data)
    
    # List operations
    start = time.time()
    list_result = [x * 2 for x in list_data]
    list_time = time.time() - start
    
    # NumPy operations
    start = time.time()
    array_result = array_data * 2
    array_time = time.time() - start
    
    print(f"List time: {list_time:.4f} seconds")
    print(f"NumPy time: {array_time:.4f} seconds")
    print(f"NumPy is {list_time/array_time:.1f}x faster")

# Memory usage comparison
def compare_memory():
    import sys
    
    # Create equivalent data structures
    data = list(range(1000))
    list_mem = sys.getsizeof(data)
    array_mem = np.array(data).nbytes
    
    print(f"List memory: {list_mem} bytes")
    print(f"NumPy memory: {array_mem} bytes")
```

{% endstep %}
{% endstepper %}

## Lists in Data Analysis

{% stepper %}
{% step %}

### Advanced List Operations

Common data manipulation patterns:

```python
# Time series manipulation
prices = [100.23, 101.45, 99.78, 102.34, 101.89]

# Calculate returns
returns = [
    ((prices[i] - prices[i-1]) / prices[i-1]) * 100
    for i in range(1, len(prices))
]

# Moving average
def moving_average(data, window=3):
    return [
        sum(data[i:i+window]) / window
        for i in range(len(data) - window + 1)
    ]

# Data cleaning
def clean_data(data):
    """Remove outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return [x for x in data if lower_bound <= x <= upper_bound]
```

 **Performance Tip**: For numerical computations, prefer NumPy arrays over lists!
{% endstep %}

{% step %}

### List Comprehensions in Data Science

Efficient data transformations:

```python
import pandas as pd
import numpy as np

# Feature engineering
dates = ['2023-01-01', '2023-01-02', '2023-01-03']
values = [100, 101, 99]

# Create time features
features = [{
    'date': pd.to_datetime(date),
    'value': value,
    'year': pd.to_datetime(date).year,
    'month': pd.to_datetime(date).month,
    'day': pd.to_datetime(date).day,
    'day_of_week': pd.to_datetime(date).dayofweek
} for date, value in zip(dates, values)]

# Data normalization
def normalize_features(data):
    """Min-max normalization"""
    min_val = min(data)
    max_val = max(data)
    return [
        (x - min_val) / (max_val - min_val)
        if max_val > min_val else 0
        for x in data
    ]
```

{% endstep %}
{% endstepper %}

## Tuples in Data Analysis

{% stepper %}
{% step %}

### Efficient Data Records

Using tuples for fixed-structure data:

```python
# Dataset records
records = [
    ('2023-01-01', 'AAPL', 173.57, 1000000),
    ('2023-01-01', 'GOOGL', 2951.88, 500000),
    ('2023-01-01', 'MSFT', 339.32, 750000)
]

# Efficient unpacking
for date, symbol, price, volume in records:
    # Process each field
    pass

# Named tuples for better readability
from collections import namedtuple

StockRecord = namedtuple('StockRecord', 
                        ['date', 'symbol', 'price', 'volume'])

records = [
    StockRecord('2023-01-01', 'AAPL', 173.57, 1000000),
    StockRecord('2023-01-01', 'GOOGL', 2951.88, 500000)
]

# Access by name
print(records[0].price)  # 173.57
```

{% endstep %}

{% step %}

### Tuple Performance Advantages

Memory and speed benefits:

```python
import sys
from timeit import timeit

# Memory comparison
tuple_data = tuple(range(1000))
list_data = list(range(1000))

print(f"Tuple size: {sys.getsizeof(tuple_data)} bytes")
print(f"List size: {sys.getsizeof(list_data)} bytes")

# Performance comparison
def compare_access():
    # Setup
    setup = """
    tuple_data = tuple(range(1000))
    list_data = list(range(1000))
    """
    
    # Test tuple access
    tuple_time = timeit(
        'x = tuple_data[500]',
        setup=setup,
        number=1000000
    )
    
    # Test list access
    list_time = timeit(
        'x = list_data[500]',
        setup=setup,
        number=1000000
    )
    
    print(f"Tuple access time: {tuple_time:.6f} seconds")
    print(f"List access time: {list_time:.6f} seconds")
```

{% endstep %}
{% endstepper %}

## Sets in Data Analysis

{% stepper %}
{% step %}

### Advanced Set Operations

Efficient unique value operations:

```python
# Feature selection
numerical_features = {'price', 'volume', 'returns'}
categorical_features = {'sector', 'industry', 'exchange'}

# Find features present in both types
common_features = numerical_features & categorical_features

# Find unique features to each type
numerical_only = numerical_features - categorical_features
categorical_only = categorical_features - numerical_features

# Efficient unique value counting
def get_unique_counts(df):
    """Get unique value counts for each column"""
    return {
        col: len(set(df[col].dropna()))
        for col in df.columns
    }

# Duplicate detection
def find_duplicates(data):
    """Find duplicate values in a sequence"""
    seen = set()
    duplicates = set()
    
    for item in data:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    
    return duplicates
```

{% endstep %}

{% step %}

### Set Operations for Data Cleaning

Common data cleaning patterns:

```python
class DataCleaner:
    def __init__(self, df):
        self.df = df
        self.categorical_cols = set()
        self.numeric_cols = set()
        self._identify_column_types()
    
    def _identify_column_types(self):
        """Identify column types"""
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                self.numeric_cols.add(col)
            else:
                self.categorical_cols.add(col)
    
    def standardize_categories(self, columns=None):
        """Standardize categorical values"""
        columns = columns or self.categorical_cols
        
        for col in columns:
            # Get unique values
            unique_values = set(self.df[col].dropna())
            
            # Create mapping for similar values
            mapping = {}
            for value in unique_values:
                key = str(value).lower().strip()
                if key not in mapping:
                    mapping[key] = value
            
            # Apply standardization
            self.df[col] = self.df[col].apply(
                lambda x: mapping.get(
                    str(x).lower().strip(), x
                ) if pd.notna(x) else x
            )
```

{% endstep %}
{% endstepper %}

## Dictionaries in Data Analysis

{% stepper %}
{% step %}

### Advanced Dictionary Patterns

Efficient data organization:

```python
class DatasetMetadata:
    def __init__(self, df):
        self.df = df
        self.metadata = self._generate_metadata()
    
    def _generate_metadata(self):
        """Generate comprehensive dataset metadata"""
        metadata = {}
        
        for column in self.df.columns:
            metadata[column] = {
                'dtype': str(self.df[column].dtype),
                'missing_count': self.df[column].isna().sum(),
                'missing_percentage': (
                    self.df[column].isna().mean() * 100
                ),
                'unique_count': self.df[column].nunique(),
                'memory_usage': self.df[column].memory_usage(deep=True)
            }
            
            # Add type-specific metadata
            if np.issubdtype(self.df[column].dtype, np.number):
                metadata[column].update({
                    'mean': self.df[column].mean(),
                    'std': self.df[column].std(),
                    'min': self.df[column].min(),
                    'max': self.df[column].max()
                })
            else:
                metadata[column].update({
                    'most_common': self.df[column].mode().iloc[0],
                    'unique_values': list(
                        self.df[column].value_counts()
                        .head()
                        .to_dict()
                    )
                })
        
        return metadata
```

{% endstep %}

{% step %}

### Dictionary Comprehensions for Analysis

Efficient data transformations:

```python
# Feature statistics
def calculate_feature_stats(df):
    """Calculate statistics for each numeric feature"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    return {
        col: {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skew': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        for col in numeric_cols
    }

# Correlation analysis
def analyze_correlations(df, threshold=0.7):
    """Find highly correlated feature pairs"""
    corr_matrix = df.corr()
    
    return {
        (col1, col2): corr_matrix.loc[col1, col2]
        for col1 in corr_matrix.columns
        for col2 in corr_matrix.columns
        if col1 < col2 and  # Avoid duplicates
        abs(corr_matrix.loc[col1, col2]) >= threshold
    }
```

{% endstep %}
{% endstepper %}

## Practice Exercises for Data Analysis

Try these data science exercises:

1. **Time Series Analysis**

   ```python
   # Create a system that:
   # - Stores time series data efficiently
   # - Calculates rolling statistics
   # - Detects anomalies
   # - Generates summary reports
   ```

2. **Feature Engineering**

   ```python
   # Build a feature engineering pipeline that:
   # - Handles different data types
   # - Creates derived features
   # - Manages feature metadata
   # - Validates feature quality
   ```

3. **Data Quality Analysis**

   ```python
   # Implement a data quality checker that:
   # - Profiles dataset characteristics
   # - Identifies data issues
   # - Suggests cleaning steps
   # - Tracks quality metrics
   ```

Remember:

- Choose appropriate data structures for your task
- Consider performance implications
- Handle edge cases
- Document your code

Happy analyzing!
