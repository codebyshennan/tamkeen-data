# Functions in Data Analysis

## Understanding Functions in Data Science

{% stepper %}
{% step %}
### Functions in Data Analysis
Think of functions as reusable data processing components:
- Input: Raw data (e.g., DataFrame, array, list)
- Process: Data transformation, analysis, or modeling
- Output: Processed data, statistics, or visualizations

```python
import pandas as pd
import numpy as np

def analyze_numeric_column(data: pd.Series) -> dict:
    """
    Analyze a numeric column and return basic statistics.
    
    Args:
        data: Pandas Series containing numeric data
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'skew': data.skew(),
        'missing': data.isna().sum()
    }

# Using the function
df = pd.DataFrame({'values': [1, 2, 3, np.nan, 5]})
stats = analyze_numeric_column(df['values'])
print(stats)
```
{% endstep %}

{% step %}
### Why Functions in Data Science?
Functions help you:
1. **Create reproducible analysis pipelines**
2. **Standardize data processing steps**
3. **Share analysis methods with team**
4. **Ensure consistent data handling**

Example without functions:
```python
# Without functions (repetitive and error-prone)
# Dataset 1
df1_nulls = df1.isnull().sum()
df1_cleaned = df1.dropna()
df1_scaled = (df1_cleaned - df1_cleaned.mean()) / df1_cleaned.std()

# Dataset 2 (repeating same steps)
df2_nulls = df2.isnull().sum()
df2_cleaned = df2.dropna()
df2_scaled = (df2_cleaned - df2_cleaned.mean()) / df2_cleaned.std()
```

Example with functions:
```python
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardized preprocessing pipeline"""
    # Check missing values
    nulls = df.isnull().sum()
    print(f"Missing values:\n{nulls}")
    
    # Clean and scale
    df_cleaned = df.dropna()
    df_scaled = (df_cleaned - df_cleaned.mean()) / df_cleaned.std()
    
    return df_scaled

# Now we can process any dataset consistently
df1_processed = preprocess_dataset(df1)
df2_processed = preprocess_dataset(df2)
```
{% endstep %}
{% endstepper %}

## Creating Data Analysis Functions

{% stepper %}
{% step %}
### Basic Function Structure
Modern data analysis function structure:

```python
from typing import Union, List, Dict
import pandas as pd
import numpy as np

def process_timeseries(
    data: Union[pd.Series, np.ndarray],
    window: int = 3,
    method: str = 'mean'
) -> Dict[str, Union[pd.Series, float]]:
    """
    Process time series data with rolling statistics.
    
    Args:
        data: Time series data
        window: Rolling window size
        method: Aggregation method ('mean' or 'median')
    
    Returns:
        Dictionary containing processed data and statistics
    
    Raises:
        ValueError: If method is not supported
    """
    # Convert to pandas Series if numpy array
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # Validate method
    if method not in ['mean', 'median']:
        raise ValueError(f"Method {method} not supported")
    
    # Calculate rolling statistics
    if method == 'mean':
        rolling = data.rolling(window).mean()
    else:
        rolling = data.rolling(window).median()
    
    return {
        'original': data,
        'rolling': rolling,
        'volatility': data.std(),
        'trend': rolling.iloc[-1] - rolling.iloc[0]
    }

# Using the function
data = pd.Series([1, 2, 3, 2, 3, 4, 3, 4, 5])
results = process_timeseries(data, window=3, method='mean')
```
{% endstep %}

{% step %}
### Parameters for Data Processing
Different ways to configure data processing:

```python
from dataclasses import dataclass
from typing import Optional, List, Union
import pandas as pd
import numpy as np

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    remove_outliers: bool = True
    fill_method: str = 'mean'
    scaling: bool = True
    outlier_threshold: float = 3.0

def process_dataset(
    df: pd.DataFrame,
    numeric_columns: List[str],
    config: Optional[ProcessingConfig] = None
) -> pd.DataFrame:
    """
    Process dataset with configurable options.
    
    Args:
        df: Input DataFrame
        numeric_columns: Columns to process
        config: Processing configuration
    
    Returns:
        Processed DataFrame
    """
    # Use default config if none provided
    if config is None:
        config = ProcessingConfig()
    
    # Work on copy
    df = df.copy()
    
    for col in numeric_columns:
        # Remove outliers
        if config.remove_outliers:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df.loc[z_scores > config.outlier_threshold, col] = np.nan
        
        # Fill missing values
        if config.fill_method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif config.fill_method == 'median':
            df[col] = df[col].fillna(df[col].median())
        
        # Scale data
        if config.scaling:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

# Using the function
df = pd.DataFrame({
    'A': [1, 2, 100, 4, 5],  # Contains outlier
    'B': [1, 2, np.nan, 4, 5]  # Contains missing value
})

# Default processing
result_default = process_dataset(df, numeric_columns=['A', 'B'])

# Custom processing
custom_config = ProcessingConfig(
    remove_outliers=True,
    fill_method='median',
    scaling=False
)
result_custom = process_dataset(
    df, 
    numeric_columns=['A', 'B'],
    config=custom_config
)
```
{% endstep %}

{% step %}
### Return Values in Data Analysis
Functions can return different types of analysis results:

```python
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats

def analyze_distribution(
    data: Union[pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    Analyze distribution of data.
    
    Args:
        data: Numeric data to analyze
        
    Returns:
        Dictionary containing:
        - Basic statistics
        - Normality test results
        - Distribution parameters
    """
    # Convert to numpy array
    if isinstance(data, pd.Series):
        data = data.dropna().values
    
    # Basic statistics
    basic_stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'skew': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
    
    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    normality_test = {
        'statistic': shapiro_stat,
        'p_value': shapiro_p,
        'is_normal': shapiro_p > 0.05
    }
    
    # Fit distribution
    dist_params = stats.norm.fit(data)
    distribution = {
        'type': 'normal',
        'parameters': {
            'loc': dist_params[0],
            'scale': dist_params[1]
        }
    }
    
    return {
        'statistics': basic_stats,
        'normality_test': normality_test,
        'distribution': distribution
    }

# Using the function
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=1000)
analysis = analyze_distribution(normal_data)

# Print results
for key, value in analysis.items():
    print(f"\n{key.title()}:")
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"  {k}: {v}")
```
{% endstep %}
{% endstepper %}

## Advanced Data Analysis Functions

{% stepper %}
{% step %}
### Function Decorators for Data Validation
Use decorators to add validation:

```python
from functools import wraps
import pandas as pd
import numpy as np

def validate_dataframe(required_columns=None, numeric_only=False):
    """
    Decorator to validate DataFrame inputs.
    
    Args:
        required_columns: List of required column names
        numeric_only: Whether to check for numeric columns
    """
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            # Check DataFrame type
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")
            
            # Check required columns
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    raise ValueError(
                        f"Missing required columns: {missing_cols}"
                    )
            
            # Check numeric columns
            if numeric_only:
                non_numeric = df[required_columns].select_dtypes(
                    exclude=[np.number]
                ).columns
                if len(non_numeric) > 0:
                    raise ValueError(
                        f"Non-numeric columns found: {non_numeric}"
                    )
            
            return func(df, *args, **kwargs)
        return wrapper
    return decorator

# Using the decorator
@validate_dataframe(
    required_columns=['A', 'B'],
    numeric_only=True
)
def calculate_correlation(df):
    """Calculate correlation between columns A and B"""
    return df['A'].corr(df['B'])

# Test the function
df_good = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
print(calculate_correlation(df_good))

# This will raise an error
df_bad = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c']  # Non-numeric
})
try:
    calculate_correlation(df_bad)
except ValueError as e:
    print(f"Error: {e}")
```
{% endstep %}

{% step %}
### Performance Optimization
Optimize functions for large datasets:

```python
import pandas as pd
import numpy as np
from typing import List, Dict
from functools import lru_cache

class DataProcessor:
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
    
    @lru_cache(maxsize=128)
    def calculate_statistics(self, values: tuple) -> Dict:
        """
        Calculate statistics with caching for repeated calculations.
        
        Args:
            values: Tuple of values (must be tuple for caching)
            
        Returns:
            Dictionary of statistics
        """
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values)
        }
    
    def process_large_dataset(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Dict]:
        """
        Process large dataset in chunks.
        
        Args:
            df: Input DataFrame
            columns: Columns to process
            
        Returns:
            Dictionary of results per column
        """
        results = {col: [] for col in columns}
        
        # Process in chunks
        for start in range(0, len(df), self.chunk_size):
            end = start + self.chunk_size
            chunk = df.iloc[start:end]
            
            # Process each column
            for col in columns:
                # Convert to tuple for caching
                values = tuple(chunk[col].dropna())
                if values:
                    stats = self.calculate_statistics(values)
                    results[col].append(stats)
        
        # Combine chunk results
        final_results = {}
        for col in columns:
            if results[col]:
                final_results[col] = {
                    'mean': np.mean([r['mean'] for r in results[col]]),
                    'std': np.mean([r['std'] for r in results[col]]),
                    'median': np.median([r['median'] for r in results[col]])
                }
        
        return final_results

# Using the optimized processor
processor = DataProcessor(chunk_size=5000)

# Generate large dataset
np.random.seed(42)
large_df = pd.DataFrame({
    'A': np.random.normal(0, 1, 10000),
    'B': np.random.normal(5, 2, 10000)
})

# Process dataset
results = processor.process_large_dataset(
    large_df,
    columns=['A', 'B']
)

# Print results
for col, stats in results.items():
    print(f"\nColumn {col}:")
    for stat, value in stats.items():
        print(f"  {stat}: {value:.2f}")
```
{% endstep %}
{% endstepper %}

## Best Practices for Data Analysis Functions

{% stepper %}
{% step %}
### Writing Maintainable Functions
Follow these data science best practices:

1. **Clear Documentation and Type Hints**:
```python
from typing import Union, List, Dict, Optional
import pandas as pd
import numpy as np

def preprocess_features(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    scaling: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess features for machine learning.
    
    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        scaling: Whether to apply standard scaling
    
    Returns:
        Tuple containing:
        - Processed DataFrame
        - Dictionary of transformation parameters
    
    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'income': [50000, 60000, 70000],
        ...     'category': ['A', 'B', 'A']
        ... })
        >>> processed_df, params = preprocess_features(
        ...     df,
        ...     numeric_features=['age', 'income'],
        ...     categorical_features=['category']
        ... )
    """
    # Function implementation...
```

2. **Error Handling and Validation**:
```python
def analyze_timeseries(
    data: pd.Series,
    window_size: int = 3
) -> Dict[str, Union[float, pd.Series]]:
    """Analyze time series data"""
    # Input validation
    if not isinstance(data, pd.Series):
        raise TypeError("Input must be pandas Series")
    
    if not pd.api.types.is_numeric_dtype(data):
        raise ValueError("Series must contain numeric data")
    
    if window_size < 1:
        raise ValueError("Window size must be positive")
    
    if window_size >= len(data):
        raise ValueError("Window size too large for data")
    
    try:
        # Calculate statistics
        results = {
            'mean': data.mean(),
            'rolling_mean': data.rolling(window_size).mean(),
            'volatility': data.std()
        }
        return results
    except Exception as e:
        raise RuntimeError(f"Error analyzing time series: {str(e)}")
```

3. **Modular Design**:
```python
class DataAnalyzer:
    """Modular data analysis pipeline"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}
    
    def clean_data(self) -> 'DataAnalyzer':
        """Clean the dataset"""
        self.df = self.df.dropna()
        return self
    
    def calculate_statistics(self) -> 'DataAnalyzer':
        """Calculate basic statistics"""
        self.results['statistics'] = {
            col: {
                'mean': self.df[col].mean(),
                'std': self.df[col].std()
            }
            for col in self.df.select_dtypes(include=[np.number])
        }
        return self
    
    def analyze_correlations(self) -> 'DataAnalyzer':
        """Analyze feature correlations"""
        numeric_cols = self.df.select_dtypes(include=[np.number])
        self.results['correlations'] = numeric_cols.corr()
        return self
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results"""
        return self.results

# Using the modular analyzer
analyzer = DataAnalyzer(df)
results = (analyzer
    .clean_data()
    .calculate_statistics()
    .analyze_correlations()
    .get_results())
```
{% endstep %}

{% step %}
### Performance Optimization Patterns

1. **Vectorization Over Loops**:
```python
# Slow: Using loops
def calculate_zscore_slow(df: pd.DataFrame) -> pd.DataFrame:
    results = df.copy()
    for column in df.columns:
        mean = df[column].mean()
        std = df[column].std()
        for idx in range(len(df)):
            results.loc[idx, column] = (
                (df.loc[idx, column] - mean) / std
            )
    return results

# Fast: Using vectorization
def calculate_zscore_fast(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()
```

2. **Efficient Memory Usage**:
```python
def process_large_file(
    filepath: str,
    chunksize: int = 10000
) -> pd.DataFrame:
    """Process large CSV file in chunks"""
    results = []
    
    # Process file in chunks
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Process chunk
        processed = chunk.groupby('category')['value'].mean()
        results.append(processed)
    
    # Combine results
    return pd.concat(results).groupby(level=0).mean()
```

3. **Caching Expensive Computations**:
```python
from functools import lru_cache

class FeatureEngine:
    @lru_cache(maxsize=128)
    def calculate_feature(self, values: tuple) -> float:
        """Calculate expensive feature with caching"""
        # Expensive computation here
        return some_expensive_calculation(values)
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for group in df.groupby('category'):
            # Convert to tuple for caching
            values = tuple(group['values'])
            feature = self.calculate_feature(values)
            results.append(feature)
        return pd.Series(results)
```
{% endstep %}
{% endstepper %}

## Practice Exercises for Data Analysis 🎯

Try these data science exercises:

1. **Feature Engineering Pipeline**
   ```python
   # Create a function that:
   # - Handles numeric and categorical features
   # - Applies appropriate scaling/encoding
   # - Handles missing values
   # - Returns processed features and parameters
   ```

2. **Time Series Analysis**
   ```python
   # Build a function that:
   # - Calculates rolling statistics
   # - Detects seasonality
   # - Identifies trends
   # - Handles missing values
   ```

3. **Data Quality Assessment**
   ```python
   # Implement a function that:
   # - Checks data types
   # - Identifies missing values
   # - Detects outliers
   # - Validates value ranges
   # - Generates quality report
   ```

Remember:
- Use type hints for better code documentation
- Handle edge cases and errors
- Optimize for performance with large datasets
- Write modular and reusable code
- Include examples in docstrings

Happy analyzing! 🚀
