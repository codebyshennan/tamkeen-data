# Basic Syntax and Data Types for Data Science

## Getting Started with Python

{% stepper %}
{% step %}
### Your First Data Analysis Program
Let's start with a simple data analysis example:

```python
# Import essential libraries
import pandas as pd
import numpy as np

# Create some sample data
data = [10, 15, 20, 25, 30]

# Calculate basic statistics
mean = np.mean(data)
std = np.std(data)

print(f"Data Analysis Results:")
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
```

This example demonstrates:
- Importing libraries (`import` statement)
- Creating a data list
- Using functions for analysis
- Formatted output with f-strings
{% endstep %}

{% step %}
### Running Python Code for Data Analysis
Two main approaches for data analysis:

1. **Interactive Mode (Jupyter Notebook)**:
   ```python
   # In Jupyter cell
   import pandas as pd
   
   # Read and display data
   df = pd.read_csv('sales_data.csv')
   df.head()
   ```
   Perfect for exploratory data analysis!

2. **Script Mode (Production Code)**:
   ```python
   # analysis.py
   import pandas as pd
   import numpy as np
   
   def analyze_sales(file_path):
       df = pd.read_csv(file_path)
       return {
           'total_sales': df['amount'].sum(),
           'average_sale': df['amount'].mean()
       }
   
   results = analyze_sales('sales_data.csv')
   ```
{% endstep %}
{% endstepper %}

## Python Syntax for Data Analysis

{% stepper %}
{% step %}
### Indentation in Data Processing
Python's indentation is crucial in data processing flows:

```python
def process_data(data):
    # First level: Function body
    if len(data) > 0:
        # Second level: Inside if statement
        cleaned_data = []
        for value in data:
            # Third level: Inside loop
            if pd.notna(value):  # Check for non-NA values
                cleaned_data.append(value)
    
    return cleaned_data

# Example usage
raw_data = [10, None, 20, np.nan, 30]
clean_data = process_data(raw_data)
```

💡 **Pro Tip**: Consistent indentation is crucial for maintaining complex data processing pipelines.
{% endstep %}

{% step %}
### Comments in Data Analysis Code
Good documentation is essential in data science:

```python
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess the input DataFrame for analysis.
    
    Parameters:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Remove missing values
    df = df.dropna()  # Important for model training
    
    # Standardize numerical columns
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df
```

💡 **Best Practice**: Use docstrings for functions and detailed inline comments for complex operations.
{% endstep %}
{% endstepper %}

## Variables and Data Types in Data Science

{% stepper %}
{% step %}
### Variables in Data Analysis
Variables in data science often represent different types of data:

```python
# Numerical data
temperature = 23.5          # Continuous data
count_users = 1000         # Discrete data

# Categorical data
category = "Electronics"    # Nominal data
rating = "A"               # Ordinal data

# Time-based data
from datetime import datetime
timestamp = datetime.now()  # Time data

# Arrays and matrices
import numpy as np
data_array = np.array([1, 2, 3, 4, 5])
data_matrix = np.array([[1, 2], [3, 4]])

# DataFrame
import pandas as pd
df = pd.DataFrame({
    'id': range(1, 4),
    'value': [10, 20, 30]
})
```

💡 **Remember**: Choose appropriate data types for efficient memory usage and processing!
{% endstep %}

{% step %}
### Variable Naming in Data Science
Follow these conventions for clear data analysis code:

✅ **Do This**:
```python
mean_temperature = 23.5     # Clear statistical measure
customer_id = "C001"        # Entity identifier
is_outlier = True          # Boolean flag
daily_sales_data = df      # DataFrame content
MAX_ITERATIONS = 1000      # Constant value
```

❌ **Don't Do This**:
```python
temp = 23.5                # Too vague
data1 = pd.DataFrame()     # Uninformative name
x = np.array([1,2,3])     # Unclear purpose
```

💡 **Pro Tip**: Use descriptive names that indicate the variable's role in your analysis.
{% endstep %}

{% step %}
### Data Types for Analysis
Python data types commonly used in data science:

1. **Numeric Types for Statistical Analysis**:
   ```python
   import numpy as np
   
   # Integer types
   sample_size = 1000                  # int
   array_int = np.int32([1, 2, 3])    # numpy int32
   
   # Float types
   mean_value = 75.5                   # float
   array_float = np.float64([1.1, 1.2])  # numpy float64
   
   # Complex numbers (e.g., for signal processing)
   signal = 3 + 4j
   ```

2. **Text Data for Natural Language Processing**:
   ```python
   # String operations for text analysis
   text = "Data Science is fascinating"
   tokens = text.lower().split()
   
   # Regular expressions for pattern matching
   import re
   emails = re.findall(r'\S+@\S+', text)
   ```

3. **Boolean Arrays for Filtering**:
   ```python
   import pandas as pd
   
   df = pd.DataFrame({
       'value': [10, 20, 30, 40, 50]
   })
   
   # Boolean indexing
   mask = df['value'] > 30
   high_values = df[mask]
   ```

4. **Special Types for Missing Data**:
   ```python
   # None for missing values
   optional_value = None
   
   # NaN for numerical missing data
   missing_numeric = np.nan
   
   # Handling missing data in pandas
   df = pd.DataFrame({
       'value': [10, np.nan, 30]
   })
   clean_df = df.dropna()
   ```

💡 **Tip**: Use `dtype` to check array types in NumPy/Pandas:
```python
print(df['value'].dtype)  # dtype('float64')
print(np.array([1, 2]).dtype)  # dtype('int64')
```
{% endstep %}
{% endstepper %}

## Working with Numbers in Data Analysis

{% stepper %}
{% step %}
### Mathematical Operations for Data Science
Common numerical operations in data analysis:

```python
import numpy as np

# Basic statistics
data = [1, 2, 3, 4, 5]
mean = np.mean(data)       # 3.0
median = np.median(data)   # 3.0
std = np.std(data)         # Standard deviation

# Matrix operations
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
product = np.dot(matrix_a, matrix_b)

# Element-wise operations
sum_matrix = matrix_a + matrix_b
diff_matrix = matrix_a - matrix_b

# Statistical functions
correlation = np.corrcoef(data, data)
```

💡 **Pro Tip**: Use NumPy for efficient numerical computations with large datasets!
{% endstep %}

{% step %}
### Numerical Precision and Types
Understanding precision in data analysis:

```python
import numpy as np

# Integer precision
int32_array = np.array([1, 2, 3], dtype=np.int32)
int64_array = np.array([1, 2, 3], dtype=np.int64)

# Float precision
float32_array = np.array([1.1, 1.2, 1.3], dtype=np.float32)
float64_array = np.array([1.1, 1.2, 1.3], dtype=np.float64)

# Memory usage
print(f"Int32 memory: {int32_array.nbytes} bytes")
print(f"Float64 memory: {float64_array.nbytes} bytes")

# Precision considerations
a = 0.1 + 0.2
b = 0.3
print(f"0.1 + 0.2 == 0.3: {abs(a - b) < 1e-10}")  # Use tolerance for float comparison
```
{% endstep %}
{% endstepper %}

## String Operations in Data Analysis

{% stepper %}
{% step %}
### Text Data Processing
Common string operations in data analysis:

```python
# Text cleaning
text = " Data Science "
cleaned = text.strip().lower()  # Remove whitespace and convert to lowercase

# Pattern matching
import re
text = "Temperature: 23.5°C"
temperature = float(re.findall(r'\d+\.\d+', text)[0])

# String parsing for data extraction
date_str = "2023-01-01"
from datetime import datetime
date_obj = datetime.strptime(date_str, '%Y-%m-%d')

# Working with CSV data
csv_line = "id,name,value"
columns = csv_line.split(',')
```
{% endstep %}

{% step %}
### String Formatting in Reports
Format strings for data reporting:

```python
# Formatting numerical results
accuracy = 0.9567
print(f"Model Accuracy: {accuracy:.2%}")  # 95.67%

# Table-like output
data = {
    'precision': 0.95,
    'recall': 0.92,
    'f1_score': 0.93
}

# Create formatted report
report = """
Model Metrics:
-------------
Precision: {precision:.2%}
Recall: {recall:.2%}
F1 Score: {f1_score:.2%}
""".format(**data)

print(report)
```
{% endstep %}
{% endstepper %}

## Type Conversion in Data Processing

{% stepper %}
{% step %}
### Data Type Conversions
Common type conversions in data analysis:

```python
import pandas as pd
import numpy as np

# Converting strings to numbers
numeric_strings = ['1', '2.5', '3.14']
integers = [int(x) for x in numeric_strings if '.' not in x]
floats = [float(x) for x in numeric_strings]

# Converting to categorical
categories = pd.Categorical(['A', 'B', 'A', 'C'])
encoded = pd.get_dummies(categories)

# DateTime conversions
dates = ['2023-01-01', '2023-01-02']
datetime_objects = pd.to_datetime(dates)

# Array type conversion
float_array = np.array([1, 2, 3], dtype=float)
int_array = float_array.astype(int)
```

💡 **Warning**: Always validate data before conversion:
```python
def safe_float_convert(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan
```
{% endstep %}

{% step %}
### Data Type Checking and Validation
Best practices for type checking in data analysis:

```python
import pandas as pd
import numpy as np

def validate_dataset(df):
    """Validate DataFrame data types and contents"""
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Check for infinite values
        if np.any(np.isinf(df[col])):
            print(f"Warning: Infinite values in {col}")
        
        # Check for reasonable ranges
        if df[col].min() < 0 and col.endswith('_positive'):
            print(f"Warning: Negative values in {col}")
    
    # Check categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Check for unexpected categories
        unique_vals = df[col].nunique()
        if unique_vals > 100:  # Arbitrary threshold
            print(f"Warning: High cardinality in {col}")
    
    return df
```
{% endstep %}
{% endstepper %}

## Practice Exercises for Data Science 🎯

Try these data analysis exercises:

1. Create a numpy array of temperatures and calculate:
   - Mean, median, and standard deviation
   - Convert Celsius to Fahrenheit
   - Find outliers (values > 2 standard deviations)

2. Process a string of comma-separated values:
   - Split into individual values
   - Convert numeric strings to floats
   - Calculate summary statistics

3. Work with dates and times:
   - Convert string dates to datetime objects
   - Calculate time differences
   - Extract specific components (year, month, day)

4. Create a simple data cleaning function:
   - Remove missing values
   - Convert data types appropriately
   - Handle outliers

Remember:
- Use numpy for numerical operations
- Pandas for structured data
- Always validate your data
- Handle errors gracefully

Happy analyzing! 🚀
