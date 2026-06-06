# Functions in Data Analysis

**After this lesson:** you can explain the core ideas in “Functions in Data Analysis” and reproduce the examples here in your own notebook or environment.

> **Must Watch:** Functions are WHERE Python Tutor shines! Visualize every function call!

> **AI Prompt:** "Explain Python functions using cooking recipes as an analogy"

> **Interactive:** [Open Functions Notebook in Colab](./notebooks/03-functions.ipynb)

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/9Os0o3wzS_I" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*Corey Schafer — Python functions*

## Understanding Functions in Data Science

### Functions in Data Analysis

A **function** is a named block of code with parameters and (optional) return values. In data work you wrap anything repeated—cleaning a column, computing a metric, plotting a standard figure—so notebooks stay short and tests can target one piece of logic.

Think of each function as a reusable **input → process → output** pipeline:

- Input: Raw data (e.g., DataFrame, array, list)
- Process: Data transformation, analysis, or modeling
- Output: Processed data, statistics, or visualizations

{% include mermaid-diagram.html src="1-data-fundamentals/1.2-intro-python/diagrams/functions-1.mmd" %}

*Once defined, the same function can be called on any column — that's the reusability win.*

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}
```
{'mean': np.float64(2.75), 'median': np.float64(2.5), 'std': np.float64(1.707825127659933), 'skew': np.float64(0.7528371991317256), 'missing': np.int64(1)}
```


</div>
<aside class="code-explainer__callouts" aria-label="Walkthrough of this example">
  <div class="code-callout" data-lines="1-2" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Libraries</span>
    </div>
    <div class="code-callout__body">
      <p>Import pandas for tables and NumPy for numeric helpers.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="4-13" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Definition and docstring</span>
    </div>
    <div class="code-callout__body">
      <p>The function takes a numeric <code>Series</code>, documents inputs and return value, and keeps analysis in one place.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-20" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Returned metrics</span>
    </div>
    <div class="code-callout__body">
      <p>Return a dict of summary stats learners can print or log.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-25" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Call site</span>
    </div>
    <div class="code-callout__body">
      <p>Build a tiny DataFrame, run the function on one column, and display the result.</p>
    </div>
  </div>
</aside>
</div>

```
{'mean': np.float64(2.75), 'median': np.float64(2.5), 'std': np.float64(1.707825127659933), 'skew': np.float64(0.7528371991317256), 'missing': np.int64(1)}
```

> **Visualize Function Calls:**
> Paste this simpler version into Python Tutor to see how functions work:
> ```python
> def calculate_average(numbers):
>   total = sum(numbers)
>   count = len(numbers)
>   average = total / count
>   return average
> 
> data = [10, 20, 30, 40, 50]
> result = calculate_average(data)
> print(f"Average: {result}")
> ```

```
Average: 30.0
```
> 
> **Watch carefully:**
> - Function definition vs function call
> - How parameters receive values
> - Variables inside function scope
> - Return value flowing back

> **AI Learning:**
> Ask: "Explain the difference between defining a function and calling a function"
> Ask: "What is 'scope' in Python functions?"

---

### Why Functions in Data Science?

Functions help you:

1. **Create reproducible analysis pipelines**
2. **Standardize data processing steps**
3. **Share analysis methods with team**
4. **Ensure consistent data handling**

Example without functions:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Without functions (repetitive and error-prone)
# Dataset 1
df1_nulls = df1.isnull().sum()
df1_cleaned = df1.dropna()
df1_scaled = (df1_cleaned - df1_cleaned.mean()) / df1_cleaned.std()

# Dataset 2 (repeating same steps)
df2_nulls = df2.isnull().sum()
df2_cleaned = df2.dropna()
df2_scaled = (df2_cleaned - df2_cleaned.mean()) / df2_cleaned.std()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Repetitive preprocessing without functions">
  <div class="code-callout" data-lines="2-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Dataset 1</span>
    </div>
    <div class="code-callout__body">
      <p>Same three steps: count nulls, drop missing, z-score scale.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-10" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Dataset 2</span>
    </div>
    <div class="code-callout__body">
      <p>Identical logic repeated—motivation for a function.</p>
    </div>
  </div>
</aside>
</div>

Example with functions:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Single preprocessing function and reuse">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Reusable pipeline</span>
    </div>
    <div class="code-callout__body">
      <p>One function encapsulates inspect, clean, and scale.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Reuse</span>
    </div>
    <div class="code-callout__body">
      <p>Call the same preprocessing for df1 and df2.</p>
    </div>
  </div>
</aside>
</div>

> **See It Work:**
> Python Tutor can show you the BEFORE and AFTER:
> ```python
> def add_ten(number):
>   return number + 10
> 
> x = 5
> y = add_ten(x)
> print(f"Original: {x}, Result: {y}")
> ```

```
Original: 5, Result: 15
```
> **Notice:** `x` doesn't change! Functions don't modify originals (unless using lists/dicts)

> **Deep Dive:**
> Ask: "Explain pass-by-value vs pass-by-reference in Python with examples"

## Creating Data Analysis Functions

---

### Basic Function Structure

Modern data analysis function structure:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Rolling time series helper">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports and typing</span>
    </div>
    <div class="code-callout__body">
      <p>Union, List, Dict for flexible inputs and structured return.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">API contract</span>
    </div>
    <div class="code-callout__body">
      <p>Parameters, rolling method, docstring, and raised errors.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-43" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Implementation</span>
    </div>
    <div class="code-callout__body">
      <p>Coerce to Series, validate method, rolling stats, return bundle.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="45-47" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Example call</span>
    </div>
    <div class="code-callout__body">
      <p>Series input, window=3, method=mean.</p>
    </div>
  </div>
</aside>
</div>

---

### Parameters for Data Processing

Different ways to configure data processing:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Configurable pipeline with dataclass">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>dataclass, typing, pandas, numpy.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">ProcessingConfig</span>
    </div>
    <div class="code-callout__body">
      <p>Frozen defaults for outlier, fill, scaling, threshold.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-53" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">process_dataset</span>
    </div>
    <div class="code-callout__body">
      <p>Default config, copy, per-column z-score outliers, fill, scale.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="55-62" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sample data</span>
    </div>
    <div class="code-callout__body">
      <p>Build df with outlier and missing value.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="64-74" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Default vs custom</span>
    </div>
    <div class="code-callout__body">
      <p>Default run then override config for median fill and no scaling.</p>
    </div>
  </div>
</aside>
</div>

---

### Return Values in Data Analysis

Functions can return different types of analysis results:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}
```

Statistics:
  mean: 0.019332055822325486
  median: 0.02530061223488824
  std: 0.9787262077473543
  skew: 0.1168008311053351
  kurtosis: 0.06620589292148393

Normality_Test:
  statistic: 0.9986092190571157
  p_value: 0.627257829024364
  is_normal: True

Distribution:
  type: normal
  parameters: {'loc': np.float64(0.019332055822325486), 'scale': np.float64(0.9787262077473543)}
```


</div>
<aside class="code-explainer__callouts" aria-label="Distribution analysis return shape">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>typing, pandas, numpy, scipy.stats.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Signature and goal</span>
    </div>
    <div class="code-callout__body">
      <p>Union input; return dict of stats, tests, and fit.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-56" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Body</span>
    </div>
    <div class="code-callout__body">
      <p>Dropna, descriptive stats, Shapiro, normal fit, nested return.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="58-68" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Demo and print</span>
    </div>
    <div class="code-callout__body">
      <p>Seed RNG, analyze, iterate nested dict for display.</p>
    </div>
  </div>
</aside>
</div>

```

Statistics:
  mean: 0.019332055822325486
  median: 0.02530061223488824
  std: 0.9787262077473543
  skew: 0.1168008311053351
  kurtosis: 0.06620589292148393

Normality_Test:
  statistic: 0.9986092190571157
  p_value: 0.627257829024364
  is_normal: True

Distribution:
  type: normal
  parameters: {'loc': np.float64(0.019332055822325486), 'scale': np.float64(0.9787262077473543)}
```

## Advanced Data Analysis Functions

---

### Function Decorators for Data Validation

Use decorators to add validation:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}
```
1.0
Error: Non-numeric columns found: Index(['B'], dtype='str')
```


</div>
<aside class="code-explainer__callouts" aria-label="DataFrame validation decorator">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>wraps, pandas, numpy.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-40" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">validate_dataframe factory</span>
    </div>
    <div class="code-callout__body">
      <p>Nested decorator checks type, columns, dtypes before calling through.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="42-49" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Decorated function</span>
    </div>
    <div class="code-callout__body">
      <p>Correlation on validated numeric A/B.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="51-66" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Tests</span>
    </div>
    <div class="code-callout__body">
      <p>Good frame succeeds; bad frame raises ValueError.</p>
    </div>
  </div>
</aside>
</div>

```
1.0
Error: Non-numeric columns found: Index(['B'], dtype='str')
```

---

### Performance Optimization

Optimize functions for large datasets:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}
```

Column A:
  mean: -0.00
  std: 1.00
  median: -0.00

Column B:
  mean: 5.03
  std: 2.00
  median: 5.03
```


</div>
<aside class="code-explainer__callouts" aria-label="Chunked processing with caching">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>pandas, numpy, typing, lru_cache.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-25" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Cached stats</span>
    </div>
    <div class="code-callout__body">
      <p>Tuple keys for lru_cache; mean/std/median dict.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="27-67" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Chunked processing</span>
    </div>
    <div class="code-callout__body">
      <p>Window over rows, per-column tuples, aggregate chunk stats.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="69-89" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Driver</span>
    </div>
    <div class="code-callout__body">
      <p>Instantiate, synthetic large_df, process, print per column.</p>
    </div>
  </div>
</aside>
</div>

```

Column A:
  mean: -0.00
  std: 1.00
  median: -0.00

Column B:
  mean: 5.03
  std: 2.00
  median: 5.03
```

## Best Practices for Data Analysis Functions

---

### Writing Maintainable Functions

Follow these data science best practices:

1. **Clear Documentation and Type Hints**:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Documented preprocess_features sketch">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>typing helpers and pandas/numpy.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-10" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Signature</span>
    </div>
    <div class="code-callout__body">
      <p>Features lists, optional categoricals, scaling flag, Tuple return.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-36" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Docstring</span>
    </div>
    <div class="code-callout__body">
      <p>Args, returns, and doctest-style example.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="37" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Placeholder</span>
    </div>
    <div class="code-callout__body">
      <p>Implementation left for the lesson.</p>
    </div>
  </div>
</aside>
</div>

2. **Error Handling and Validation**:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Validated time series analysis">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">API</span>
    </div>
    <div class="code-callout__body">
      <p>Series input, window default, typed return.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Guards</span>
    </div>
    <div class="code-callout__body">
      <p>Type, numeric dtype, window bounds before compute.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-28" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compute and errors</span>
    </div>
    <div class="code-callout__body">
      <p>Rolling mean and volatility; wrap failures in RuntimeError.</p>
    </div>
  </div>
</aside>
</div>

3. **Modular Design**:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Fluent DataAnalyzer chain">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class header and constructor</span>
    </div>
    <div class="code-callout__body">
      <p><code>df.copy()</code> prevents accidental mutation of the caller's DataFrame. <code>self.results</code> is a shared dict that each method populates — returned at the end of the chain.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-11" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">clean_data</span>
    </div>
    <div class="code-callout__body">
      <p>Drops rows with any missing value. Returning <code>self</code> is what makes method chaining possible — the next method call goes on the same object.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-22" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">calculate_statistics</span>
    </div>
    <div class="code-callout__body">
      <p>A dict comprehension builds one <code>{mean, std}</code> entry per numeric column. <code>select_dtypes(include=[np.number])</code> automatically skips text columns — the result is stored in <code>self.results['statistics']</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-32" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">analyze_correlations and get_results</span>
    </div>
    <div class="code-callout__body">
      <p><code>numeric_cols.corr()</code> returns a pairwise correlation matrix. <code>get_results()</code> terminates the chain and hands back the accumulated <code>self.results</code> dict for printing or further analysis.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="34-40" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fluent usage</span>
    </div>
    <div class="code-callout__body">
      <p>Method chaining calls each step left-to-right in a single expression. Parentheses wrap the chain for readability across multiple lines.</p>
    </div>
  </div>
</aside>
</div>

---

### Performance Optimization Patterns

1. **Vectorization Over Loops**:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Loop versus vectorized z-score">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Loop version</span>
    </div>
    <div class="code-callout__body">
      <p>Nested loops and per-cell loc—slow on big frames.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Vectorized</span>
    </div>
    <div class="code-callout__body">
      <p>Whole-frame mean/std in one expression.</p>
    </div>
  </div>
</aside>
</div>

2. **Efficient Memory Usage**:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Chunked CSV aggregation">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Signature</span>
    </div>
    <div class="code-callout__body">
      <p>Path, chunk size, aggregated DataFrame return.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Chunk loop</span>
    </div>
    <div class="code-callout__body">
      <p>read_csv chunks, groupby mean per chunk, concat and re-average.</p>
    </div>
  </div>
</aside>
</div>

3. **Caching Expensive Computations**:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="lru_cache feature pattern">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">lru_cache</span>
    </div>
    <div class="code-callout__body">
      <p>Expensive feature memoized on tuple argument.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Batch use</span>
    </div>
    <div class="code-callout__body">
      <p>Groupby category, tuple per group, collect Series.</p>
    </div>
  </div>
</aside>
</div>

## Practice Exercises for Data Analysis

> **Learning Path:** Write code → Visualize in Python Tutor → Refine with AI feedback

### Exercise 1: Simple Statistics Function
<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def calculate_statistics(numbers):
   """
   Calculate basic statistics for a list of numbers.
   
   Your function should return a dictionary with:
   - mean
   - median
   - min
   - max
   - range
   """
   # Your code here...
   pass

# Test it:
data = [10, 15, 20, 25, 30, 35, 40]
stats = calculate_statistics(data)
print(stats)
{% endhighlight %}
```
None
```


</div>
<aside class="code-explainer__callouts" aria-label="Exercise: statistics dict">
  <div class="code-callout" data-lines="1-13" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Stub</span>
    </div>
    <div class="code-callout__body">
      <p>Docstring contract; learner fills statistics dict.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="15-18" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Test harness</span>
    </div>
    <div class="code-callout__body">
      <p>Sample list and print.</p>
    </div>
  </div>
</aside>
</div>

```
None
```

> **Visualize:** Paste into Python Tutor to see how your function processes the list
> **Get Help:** "Show me how to calculate median in Python"

### Exercise 2: Data Cleaning Function
<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def clean_data(data_list):
   """
   Clean a list by:
   1. Removing None values
   2. Converting strings to numbers where possible
   3. Removing negative numbers
   4. Returning cleaned list
   """
   # Your code here...
   pass

# Test it:
messy_data = [10, None, "20", -5, "30", 40, None, "-10", 50]
clean = clean_data(messy_data)
print(clean) # Should output: [10, 20, 30, 40, 50]
{% endhighlight %}
```
None
```


</div>
<aside class="code-explainer__callouts" aria-label="Exercise: clean mixed list">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Stub</span>
    </div>
    <div class="code-callout__body">
      <p>Cleaning rules as checklist in docstring.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Messy input</span>
    </div>
    <div class="code-callout__body">
      <p>Mixed types and sentinels for testing.</p>
    </div>
  </div>
</aside>
</div>

```
None
```

> **Debug Visually:** If something breaks, paste into Python Tutor to see where
> **Prompt:** "Help me handle edge cases in this data cleaning function"

### Exercise 3: Function with Multiple Return Values
<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def analyze_sales(sales_list):
   """
   Analyze sales data and return:
   - total_sales
   - average_sale
   - highest_sale
   - lowest_sale
   - number_of_sales
   """
   # Your code here...
   pass

# Test it:
sales = [100, 150, 200, 175, 225, 190, 210]
total, avg, high, low, count = analyze_sales(sales)
print(f"Total: ${total}, Average: ${avg:.2f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Exercise: sales tuple return">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Stub</span>
    </div>
    <div class="code-callout__body">
      <p>Multiple metrics to return (tuple unpacking practice).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-16" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sales test data</span>
    </div>
    <div class="code-callout__body">
      <p>Unpack five values into formatted print.</p>
    </div>
  </div>
</aside>
</div>

> **Observe:** Python Tutor shows how functions return multiple values as a tuple!
> **Learn:** "Explain tuple unpacking in Python with examples"

### Exercise 4: Nested Functions
<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def process_dataset(data):
   """
   Main function that uses helper functions.
   
   Create helper functions inside:
   - validate_data(data)
   - remove_outliers(data)
   - normalize_data(data)
   """
   
   def validate_data(d):
       # Your code...
       pass
   
   def remove_outliers(d):
       # Your code...
       pass
   
   def normalize_data(d):
       # Your code...
       pass
   
   # Use helper functions
   # Your code here...
   pass

# Test it:
raw_data = [10, 20, 15, 100, 18, 22, -5, 25]
processed = process_dataset(raw_data)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Exercise: nested helpers">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer contract</span>
    </div>
    <div class="code-callout__body">
      <p>Nested helpers listed in docstring.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-25" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Inner defs</span>
    </div>
    <div class="code-callout__body">
      <p>Placeholders for validate, outliers, normalize.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="27-29" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Test list</span>
    </div>
    <div class="code-callout__body">
      <p>Raw data passed into outer function.</p>
    </div>
  </div>
</aside>
</div>

> **Advanced Visualization:** Python Tutor shows nested function scopes beautifully!
> **Challenge:** "Explain when to use nested functions vs separate functions"

## Challenge Projects

### Project 1: Temperature Converter
Create a function that converts between Celsius, Fahrenheit, and Kelvin.
<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def convert_temperature(value, from_unit, to_unit):
   # Your implementation
   pass
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Project: temperature converter stub">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Signature</span>
    </div>
    <div class="code-callout__body">
      <p>Units in/out—implementation left to learner.</p>
    </div>
  </div>
</aside>
</div>

### Project 2: Grade Calculator
Build a function that calculates letter grades from percentages with customizable ranges.

### Project 3: Data Validator
Create a function that validates data according to specified rules.

> **Video Help:** Check [Video Resources](./video-resources.md) - Functions section
> **Code Review:** After completing, ask AI: "Review my function and suggest improvements: [paste code]"

## Debugging Functions

### Common Issues & Solutions

**Issue 1: Function returns None**
<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Wrong (avoid):
def add_numbers(a, b):
   result = a + b
   # Forgot return!

# Right (preferred):
def add_numbers(a, b):
   result = a + b
   return result
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Debugging: missing return">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Missing return</span>
    </div>
    <div class="code-callout__body">
      <p>Computes result but callers get None.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-9" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fix</span>
    </div>
    <div class="code-callout__body">
      <p>Explicit return passes value back.</p>
    </div>
  </div>
</aside>
</div>

> **Spot the Issue:** Python Tutor shows None being returned!

**Issue 2: Variable not found**
<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Wrong (avoid):
def calculate():
   x = 10
   return x

result = calculate()
print(x) # Error! x only exists inside function

# Right (preferred):
def calculate():
   x = 10
   return x

result = calculate()
print(result) # Use returned value
{% endhighlight %}
```
5
10
```


</div>
<aside class="code-explainer__callouts" aria-label="Debugging: scope and return value">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Name error</span>
    </div>
    <div class="code-callout__body">
      <p>x exists only inside calculate; print(x) fails at module scope.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fix</span>
    </div>
    <div class="code-callout__body">
      <p>Capture return value and print that.</p>
    </div>
  </div>
</aside>
</div>

```
5
10
```

> **See Scope:** Python Tutor visualizes function scope perfectly!

> **Debug Helper:** Paste error and code, ask: "Why am I getting this error?"

Remember:

- Use type hints for better code documentation
- Handle edge cases and errors
- Optimize for performance with large datasets
- Write modular and reusable code
- Include examples in docstrings
- **Visualize complex functions in Python Tutor**
- **Use AI to understand error messages**
- **Test functions with different inputs**

## Common pitfalls

- **Forgetting return** — If you omit **return**, the function returns **None**; Python Tutor shows this clearly.
- **Mutable default arguments** — Do not use **def f(items=[])**; use **None** and assign **items = items or []** inside.
- **Shadowing names** — Reusing a name for a parameter and an outer variable makes bugs hard to spot.

## Next steps

Continue to [Classes and objects](./classes-objects.md) for basic object-oriented programming.

Happy analyzing!
