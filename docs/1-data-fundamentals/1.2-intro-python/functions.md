# Functions in Data Analysis

**After this lesson:** you can explain Functions in Data Analysis and try the examples in your own notebook.

> **Must Watch:** Functions are WHERE Python Tutor shines! Visualize every function call!

> **AI Prompt:** "Explain Python functions using cooking recipes as an analogy"

> **Interactive:** [Open Functions Notebook in Colab](notebooks/03-functions.ipynb)

### Video

_Corey Schafer, Python functions_

## Understanding Functions in Data Science

### Functions in Data Analysis

A **function** is a named block of code with parameters and (optional) return values. In data work you wrap anything repeated, cleaning a column, computing a metric, plotting a standard figure, so notebooks stay short and tests can target one piece of logic.

Think of each function as a reusable **input → process → output** pipeline:

* Input: Raw data (e.g., DataFrame, array, list)
* Process: Data transformation, analysis, or modeling
* Output: Processed data, statistics, or visualizations

_Once defined, the same function can be called on any column, that's the reusability win._

```
{'mean': np.float64(2.75), 'median': np.float64(2.5), 'std': np.float64(1.707825127659933), 'skew': np.float64(0.7528371991317256), 'missing': np.int64(1)}
```

Libraries

Import pandas for tables and NumPy for numeric helpers.

Definition and docstring

The function takes a numeric `Series`, documents inputs and return value, and keeps analysis in one place.

Returned metrics

Return a dict of summary stats learners can print or log.

Call site

Build a tiny DataFrame, run the function on one column, and display the result.

```
{'mean': np.float64(2.75), 'median': np.float64(2.5), 'std': np.float64(1.707825127659933), 'skew': np.float64(0.7528371991317256), 'missing': np.int64(1)}
```

> **Visualize Function Calls:** Paste this simpler version into Python Tutor to see how functions work:
>
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

> **Watch carefully:**
>
> * Function definition vs function call
> * How parameters receive values
> * Variables inside function scope
> * Return value flowing back

> **AI Learning:** Ask: "Explain the difference between defining a function and calling a function" Ask: "What is 'scope' in Python functions?"

***

### Why Functions in Data Science?

Functions help you:

1. **Create reproducible analysis pipelines**
2. **Standardize data processing steps**
3. **Share analysis methods with team**
4. **Ensure consistent data handling**

Example without functions:

Dataset 1

Same three steps: count nulls, drop missing, z-score scale.

Dataset 2

Identical logic repeated, motivation for a function.

Example with functions:

Reusable pipeline

One function encapsulates inspect, clean, and scale.

Reuse

Call the same preprocessing for df1 and df2.

> **See It Work:** Python Tutor can show you the BEFORE and AFTER:
>
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

> **Deep Dive:** Ask: "Explain pass-by-value vs pass-by-reference in Python with examples"

## Creating Data Analysis Functions

***

### Basic Function Structure

Modern data analysis function structure:

Imports and typing

Union, List, Dict for flexible inputs and structured return.

API contract

Parameters, rolling method, docstring, and raised errors.

Implementation

Coerce to Series, validate method, rolling stats, return bundle.

Example call

Series input, window=3, method=mean.

***

### Parameters for Data Processing

Different ways to configure data processing:

Imports

dataclass, typing, pandas, numpy.

ProcessingConfig

Frozen defaults for outlier, fill, scaling, threshold.

process\_dataset

Default config, copy, per-column z-score outliers, fill, scale.

Sample data

Build df with outlier and missing value.

Default vs custom

Default run then override config for median fill and no scaling.

***

### Return Values in Data Analysis

Functions can return different types of analysis results:

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

Imports

typing, pandas, numpy, scipy.stats.

Signature and goal

Union input; return dict of stats, tests, and fit.

Body

Dropna, descriptive stats, Shapiro, normal fit, nested return.

Demo and print

Seed RNG, analyze, iterate nested dict for display.

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

***

### Function Decorators for Data Validation

Use decorators to add validation:

```
1.0
Error: Non-numeric columns found: Index(['B'], dtype='str')
```

Imports

wraps, pandas, numpy.

validate\_dataframe factory

Nested decorator checks type, columns, dtypes before calling through.

Decorated function

Correlation on validated numeric A/B.

Tests

Good frame succeeds; bad frame raises ValueError.

```
1.0
Error: Non-numeric columns found: Index(['B'], dtype='str')
```

***

### Performance Optimization

Optimize functions for large datasets:

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

Imports

pandas, numpy, typing, lru\_cache.

Cached stats

Tuple keys for lru\_cache; mean/std/median dict.

Chunked processing

Window over rows, per-column tuples, aggregate chunk stats.

Driver

Instantiate, synthetic large\_df, process, print per column.

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

***

### Writing Maintainable Functions

Follow these data science best practices:

1. **Clear Documentation and Type Hints**:

Imports

typing helpers and pandas/numpy.

Signature

Features lists, optional categoricals, scaling flag, Tuple return.

Docstring

Args, returns, and doctest-style example.

Placeholder

Implementation left for the lesson.

2. **Error Handling and Validation**:

API

Series input, window default, typed return.

Guards

Type, numeric dtype, window bounds before compute.

Compute and errors

Rolling mean and volatility; wrap failures in RuntimeError.

3. **Modular Design**:

Class header and constructor

`df.copy()` prevents accidental mutation of the caller's DataFrame. `self.results` is a shared dict that each method populates, returned at the end of the chain.

clean\_data

Drops rows with any missing value. Returning `self` is what makes method chaining possible, the next method call goes on the same object.

calculate\_statistics

A dict comprehension builds one `{mean, std}` entry per numeric column. `select_dtypes(include=[np.number])` automatically skips text columns, the result is stored in `self.results['statistics']`.

analyze\_correlations and get\_results

`numeric_cols.corr()` returns a pairwise correlation matrix. `get_results()` terminates the chain and hands back the accumulated `self.results` dict for printing or further analysis.

Fluent usage

Method chaining calls each step left-to-right in a single expression. Parentheses wrap the chain for readability across multiple lines.

***

### Performance Optimization Patterns

1. **Vectorization Over Loops**:

Loop version

Nested loops and per-cell loc, slow on big frames.

Vectorized

Whole-frame mean/std in one expression.

2. **Efficient Memory Usage**:

Signature

Path, chunk size, aggregated DataFrame return.

Chunk loop

read\_csv chunks, groupby mean per chunk, concat and re-average.

3. **Caching Expensive Computations**:

lru\_cache

Expensive feature memoized on tuple argument.

Batch use

Groupby category, tuple per group, collect Series.

## Practice Exercises for Data Analysis

> **Learning Path:** Write code → Visualize in Python Tutor → Refine with AI feedback

### Exercise 1: Simple Statistics Function

```
None
```

Stub

Docstring contract; learner fills statistics dict.

Test harness

Sample list and print.

```
None
```

> **Visualize:** Paste into Python Tutor to see how your function processes the list **Get Help:** "Show me how to calculate median in Python"

### Exercise 2: Data Cleaning Function

```
None
```

Stub

Cleaning rules as checklist in docstring.

Messy input

Mixed types and sentinels for testing.

```
None
```

> **Debug Visually:** If something breaks, paste into Python Tutor to see where **Prompt:** "Help me handle edge cases in this data cleaning function"

### Exercise 3: Function with Multiple Return Values

Stub

Multiple metrics to return (tuple unpacking practice).

Sales test data

Unpack five values into formatted print.

> **Observe:** Python Tutor shows how functions return multiple values as a tuple! **Learn:** "Explain tuple unpacking in Python with examples"

### Exercise 4: Nested Functions

Outer contract

Nested helpers listed in docstring.

Inner defs

Placeholders for validate, outliers, normalize.

Test list

Raw data passed into outer function.

> **Advanced Visualization:** Python Tutor shows nested function scopes beautifully! **Challenge:** "Explain when to use nested functions vs separate functions"

## Challenge Projects

### Project 1: Temperature Converter

Create a function that converts between Celsius, Fahrenheit, and Kelvin.

Signature

Units in/out, implementation left to learner.

### Project 2: Grade Calculator

Build a function that calculates letter grades from percentages with customizable ranges.

### Project 3: Data Validator

Create a function that validates data according to specified rules.

> **Video Help:** Check [Video Resources](video-resources.md) - Functions section **Code Review:** After completing, ask AI: "Review my function and suggest improvements: \[paste code]"

## Debugging Functions

### Common Issues & Solutions

**Issue 1: Function returns None**

Missing return

Computes result but callers get None.

Fix

Explicit return passes value back.

> **Spot the Issue:** Python Tutor shows None being returned!

**Issue 2: Variable not found**

\`\`\` 5 10 \`\`\`Name error

x exists only inside calculate; print(x) fails at module scope.

Fix

Capture return value and print that.

```
5
10
```

> **See Scope:** Python Tutor visualizes function scope perfectly!

> **Debug Helper:** Paste error and code, ask: "Why am I getting this error?"

Remember:

* Use type hints for better code documentation
* Handle edge cases and errors
* Optimize for performance with large datasets
* Write modular and reusable code
* Include examples in docstrings
* **Visualize complex functions in Python Tutor**
* **Use AI to understand error messages**
* **Test functions with different inputs**

## Common pitfalls

* **Forgetting return**: If you omit **return**, the function returns **None**; Python Tutor shows this clearly.
* **Mutable default arguments**: Do not use **def f(items=\[])**; use **None** and assign **items = items or \[]** inside.
* **Shadowing names**: Reusing a name for a parameter and an outer variable makes bugs hard to spot.

## Next steps

Continue to [Classes and objects](classes-objects.md) for basic object-oriented programming.

Happy analyzing!
