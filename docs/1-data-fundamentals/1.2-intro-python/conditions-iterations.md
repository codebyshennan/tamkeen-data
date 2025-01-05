# Conditions and Iterations in Data Analysis

## Making Decisions with Conditions

{% stepper %}
{% step %}
### Understanding If Statements in Data Analysis
Conditions are crucial for data filtering and validation:

```python
import pandas as pd
import numpy as np

# Data validation example
def validate_age(age):
    if age < 0:
        return np.nan  # Invalid age
    elif age > 120:
        return np.nan  # Likely invalid age
    else:
        return age

# Handling missing values
def process_value(value):
    if pd.isna(value):
        return 0  # Replace missing with default
    elif np.isinf(value):
        return np.nan  # Handle infinity
    else:
        return value
```

💡 **Remember**: Always validate your data before analysis!
{% endstep %}

{% step %}
### If-Else in Data Processing
Common data processing scenarios:

```python
import pandas as pd

# Data quality check
def check_data_quality(df):
    if df.isnull().sum().any():
        print("Warning: Dataset contains missing values")
        missing_stats = df.isnull().sum()
        print(f"Missing value counts:\n{missing_stats}")
    else:
        print("Data quality check passed: No missing values")

# Outlier detection
def flag_outlier(value, mean, std):
    if abs(value - mean) > 3 * std:
        return 'outlier'
    else:
        return 'normal'
```

Real-world example:
```python
# Sales data analysis
def analyze_sales_performance(sales_value, target):
    if sales_value >= target * 1.2:
        return 'Exceptional'
    elif sales_value >= target:
        return 'Met Target'
    elif sales_value >= target * 0.8:
        return 'Near Target'
    else:
        return 'Below Target'
```
{% endstep %}

{% step %}
### Multiple Conditions in Data Analysis
Complex data processing decisions:

```python
import pandas as pd
import numpy as np

def categorize_customer(purchase_amount, frequency, tenure):
    """Categorize customer based on multiple metrics"""
    if purchase_amount > 1000 and frequency > 12:
        if tenure > 2:
            return 'Premium'
        else:
            return 'High Value'
    elif purchase_amount > 500 or frequency > 6:
        return 'Regular'
    else:
        return 'Standard'

# Data transformation example
def transform_value(value, data_type):
    if pd.isna(value):
        return np.nan
    elif data_type == 'numeric':
        if isinstance(value, str):
            try:
                return float(value.replace(',', ''))
            except ValueError:
                return np.nan
        else:
            return float(value)
    elif data_type == 'categorical':
        return str(value).lower().strip()
    else:
        return value
```
{% endstep %}

{% step %}
### Nested Conditions in Feature Engineering
Complex feature creation:

```python
def create_age_features(df):
    """Create age-related features for analysis"""
    
    def categorize_age(age, gender):
        if pd.isna(age):
            return 'Unknown'
        else:
            if gender == 'F':
                if age < 25:
                    return 'Young Adult Female'
                elif age < 45:
                    return 'Adult Female'
                else:
                    return 'Senior Female'
            else:  # gender == 'M'
                if age < 25:
                    return 'Young Adult Male'
                elif age < 45:
                    return 'Adult Male'
                else:
                    return 'Senior Male'
    
    df['age_category'] = df.apply(
        lambda row: categorize_age(row['age'], row['gender']),
        axis=1
    )
    return df
```
{% endstep %}
{% endstepper %}

## Data Filtering and Comparison

{% stepper %}
{% step %}
### Comparison Operations in Pandas
Efficient data filtering:

```python
import pandas as pd
import numpy as np

# Load sample data
df = pd.DataFrame({
    'value': [10, 20, 30, 40, 50],
    'category': ['A', 'B', 'A', 'B', 'C']
})

# Single condition
high_values = df[df['value'] > 30]

# Multiple conditions
filtered_data = df[
    (df['value'] > 20) & 
    (df['category'] == 'A')
]

# Complex filtering
def filter_outliers(df, columns, n_std=3):
    """Filter outliers based on standard deviation"""
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[
            (df[col] >= mean - n_std * std) &
            (df[col] <= mean + n_std * std)
        ]
    return df
```

💡 **Performance Tip**: Use vectorized operations instead of loops for filtering!
{% endstep %}

{% step %}
### Logical Operations in Data Analysis
Combining multiple conditions:

```python
import pandas as pd

# Data quality checks
def check_data_validity(df):
    """Check various data quality conditions"""
    
    conditions = {
        'missing_values': df.isnull().sum().sum() > 0,
        'negative_values': (df.select_dtypes(include=[np.number]) < 0).any().any(),
        'duplicates': df.duplicated().any(),
        'outliers': detect_outliers(df)
    }
    
    if any(conditions.values()):
        print("Data quality issues found:")
        for issue, exists in conditions.items():
            if exists:
                print(f"- {issue.replace('_', ' ').title()}")
        return False
    else:
        print("All data quality checks passed")
        return True

def detect_outliers(df, threshold=3):
    """Detect outliers using Z-score method"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    has_outliers = False
    
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        if (z_scores > threshold).any():
            has_outliers = True
            break
    
    return has_outliers
```
{% endstep %}
{% endstepper %}

## Efficient Data Iteration

{% stepper %}
{% step %}
### Vectorized Operations vs. Loops
Understanding performance implications:

```python
import pandas as pd
import numpy as np

# ❌ Slow: Using loops
def slow_calculation(df):
    results = []
    for index, row in df.iterrows():
        value = row['value']
        if value > 0:
            results.append(np.log(value))
        else:
            results.append(np.nan)
    return results

# ✅ Fast: Using vectorized operations
def fast_calculation(df):
    return np.where(
        df['value'] > 0,
        np.log(df['value']),
        np.nan
    )

# ✅ Fast: Using pandas methods
def process_data(df):
    # Calculate statistics
    df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    
    # Apply multiple conditions
    conditions = [
        (df['z_score'] < -2),
        (df['z_score'] >= -2) & (df['z_score'] <= 2),
        (df['z_score'] > 2)
    ]
    choices = ['Low', 'Normal', 'High']
    
    df['category'] = np.select(conditions, choices, default='Unknown')
    return df
```
{% endstep %}

{% step %}
### Efficient Iteration When Necessary
Some cases require iteration:

```python
import pandas as pd
from tqdm import tqdm  # Progress bar

def process_large_dataset(df, chunk_size=1000):
    """Process large dataset in chunks"""
    results = []
    
    # Iterate over chunks
    for i in tqdm(range(0, len(df), chunk_size)):
        chunk = df.iloc[i:i + chunk_size].copy()
        
        # Process chunk
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    return pd.concat(results)

def process_chunk(chunk):
    """Process individual chunk of data"""
    # Perform calculations
    chunk['calculated'] = chunk['value'].apply(complex_calculation)
    
    # Apply transformations
    chunk['transformed'] = np.where(
        chunk['calculated'] > 0,
        np.log(chunk['calculated']),
        0
    )
    
    return chunk
```

💡 **Performance Tip**: Use chunking for large datasets that don't fit in memory!
{% endstep %}

{% step %}
### Working with Time Series Data
Efficient time series processing:

```python
import pandas as pd

def analyze_time_series(df):
    """Analyze time series data with rolling windows"""
    
    # Sort by date
    df = df.sort_values('date')
    
    # Calculate rolling statistics
    df['rolling_mean'] = df['value'].rolling(window=7).mean()
    df['rolling_std'] = df['value'].rolling(window=7).std()
    
    # Detect trends
    df['trend'] = np.where(
        df['rolling_mean'] > df['rolling_mean'].shift(1),
        'Upward',
        'Downward'
    )
    
    return df

def process_by_group(df, group_col, value_col):
    """Process data by groups efficiently"""
    
    def group_operation(group):
        return pd.Series({
            'mean': group[value_col].mean(),
            'std': group[value_col].std(),
            'count': len(group),
            'has_outliers': detect_outliers(group[value_col])
        })
    
    return df.groupby(group_col).apply(group_operation)
```
{% endstep %}
{% endstepper %}

## Common Data Processing Patterns

{% stepper %}
{% step %}
### Pattern: Data Validation
Common validation patterns:

```python
import pandas as pd
import numpy as np

class DataValidator:
    def __init__(self, df):
        self.df = df
        self.validation_results = []
    
    def validate_numeric_range(self, column, min_val, max_val):
        """Validate numeric values are within range"""
        mask = self.df[column].between(min_val, max_val)
        invalid = self.df[~mask]
        if len(invalid) > 0:
            self.validation_results.append(
                f"Found {len(invalid)} values outside range "
                f"[{min_val}, {max_val}] in {column}"
            )
    
    def validate_categorical(self, column, valid_categories):
        """Validate categorical values"""
        invalid = self.df[~self.df[column].isin(valid_categories)]
        if len(invalid) > 0:
            self.validation_results.append(
                f"Found {len(invalid)} invalid categories in {column}"
            )
    
    def get_validation_report(self):
        """Generate validation report"""
        if self.validation_results:
            return "\n".join(self.validation_results)
        return "All validations passed"
```
{% endstep %}

{% step %}
### Pattern: Data Cleaning
Standard cleaning operations:

```python
class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
    
    def clean_numeric(self, column):
        """Clean numeric column"""
        # Replace invalid values with NaN
        self.df[column] = pd.to_numeric(
            self.df[column], 
            errors='coerce'
        )
        
        # Remove outliers
        z_scores = np.abs(
            (self.df[column] - self.df[column].mean()) / 
            self.df[column].std()
        )
        self.df.loc[z_scores > 3, column] = np.nan
    
    def clean_categorical(self, column):
        """Clean categorical column"""
        # Standardize categories
        self.df[column] = self.df[column].str.lower().str.strip()
        
        # Replace rare categories
        value_counts = self.df[column].value_counts()
        rare_categories = value_counts[value_counts < 10].index
        self.df.loc[
            self.df[column].isin(rare_categories),
            column
        ] = 'other'
    
    def get_cleaned_data(self):
        """Return cleaned dataset"""
        return self.df
```
{% endstep %}
{% endstepper %}

## Practice Exercises for Data Analysis 🎯

Try these data science exercises:

1. **Data Quality Assessment**
   ```python
   # Create a function that:
   # - Checks for missing values
   # - Identifies outliers
   # - Validates data types
   # - Reports data quality metrics
   ```

2. **Time Series Analysis**
   ```python
   # Build a program that:
   # - Processes time series data
   # - Calculates rolling statistics
   # - Detects anomalies
   # - Generates summary reports
   ```

3. **Customer Segmentation**
   ```python
   # Implement a system that:
   # - Processes customer data
   # - Calculates key metrics
   # - Segments customers by behavior
   # - Generates insights
   ```

Remember:
- Use vectorized operations when possible
- Consider memory efficiency
- Handle edge cases
- Validate results

Happy analyzing! 🚀
