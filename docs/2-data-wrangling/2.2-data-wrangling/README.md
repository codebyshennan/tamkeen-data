# Data Wrangling: From Raw Data to Reliable Insights

**After this submodule:** You can assess quality, handle **missing values** and **outliers**, and **transform** columns so downstream analysis (EDA or modeling) is trustworthy.

## Helpful video

Pandas DataFrames in a quick walkthrough, useful for cleaning and wrangling.

## Overview

**Prerequisites:** [Pandas](../../1-data-fundamentals/1.5-data-analysis-pandas/) and [NumPy](../../1-data-fundamentals/1.4-data-foundation-linear-algebra/) basics. [SQL (Module 2.1)](../2.1-sql/) is useful when your raw data comes from databases.

> **Time needed:** Several hours across lessons and the tutorial notebook.

## Lesson path (site order)

1. [Data quality](data-quality.md)
2. [Missing values](missing-values.md)
3. [Outliers](outliers.md)
4. [Transformations](transformations.md)
5. [Wrangling project](project.md)

## Why this matters

Raw exports from databases, APIs, and spreadsheets are rarely analysis-ready: types drift, codes disagree, and missingness follows real processes. Wrangling is how you **measure** those issues, **fix** what you can justify, and **document** what you changed so EDA and modeling rest on solid ground.

Data wrangling, also known as data munging or data preprocessing, is the art and science of transforming raw data into a clean, reliable format suitable for analysis. Think of it as preparing ingredients before cooking: a chef needs clean, consistent ingredients; an analyst needs tables that match the question.

## The Data Wrangling Journey

we will look at the essential steps in transforming messy data into analysis-ready datasets:

## Learning Objectives

After completing this module, you will be able to:

1. **Assess Data Quality**
   * Identify data quality dimensions (accuracy, completeness, consistency, timeliness)
   * Measure data completeness using statistical methods
   * Evaluate data consistency across different sources
   * Detect anomalies using statistical and machine learning approaches
   * Example: Analyzing customer data to identify incorrect email formats or impossible age values
2. **Clean Data Effectively**
   * Handle missing values using advanced imputation techniques
   * Treat outliers using statistical methods (z-score, IQR)
   * Remove or merge duplicates while preserving data integrity
   * Fix inconsistencies in formats and representations
   * Example: Cleaning sales data by handling missing prices, removing duplicate orders, and standardizing product names
3. **Transform Data**
   * Scale numerical features using various methods (min-max, standard scaling)
   * Encode categorical variables (one-hot, label encoding)
   * Engineer new features to capture domain knowledge
   * Standardize formats (dates, currencies, units)
   * Example: Preparing customer transaction data by normalizing monetary values and creating time-based features
4. **Validate Results**
   * Implement automated quality checks
   * Verify transformations using statistical tests
   * Ensure data integrity through cross-validation
   * Document changes for reproducibility
   * Example: Validating cleaned customer data by checking for impossible combinations and verifying statistical properties

## Real-World Example: E-commerce Data Analysis

Walk through a comprehensive example of wrangling e-commerce data. This example demonstrates common challenges and solutions you'll encounter in real-world data science projects:

```
Data Quality Report
--------------------------------------------------
Total Records: 120
Missing Values:
date          0
order_date    0
sales         0
revenue       0
price         1
quantity      1
category      0
email         1
dtype: int64

Duplicate Records: 0
Data validation passed!
```

Imports and data loading

Bring in pandas, NumPy, and StandardScaler, then read the raw CSV into a DataFrame.

Data Quality Assessment

Print a quick quality report: total record count, missing-value counts per column, and duplicate row count.

Handle Missing Values

Fill numeric columns with their median and categorical columns with their mode, two safe default strategies.

Handle Outliers

Define `remove_outliers` using the z-score rule (keep rows within _n_ standard deviations), then apply it to the `price` column.

Feature Engineering

Create two derived columns: `total_value` (price × quantity) and `order_month` extracted from the order date.

Data Validation

`validate_data` asserts no nulls remain and that price and quantity are non-negative, then calls it to confirm the pipeline succeeded.

```
Data Quality Report
--------------------------------------------------
Total Records: 120
Missing Values:
date          0
order_date    0
sales         0
revenue       0
price         1
quantity      1
category      0
email         1
dtype: int64

Duplicate Records: 0
Data validation passed!
```

## Common Data Quality Issues and Solutions

Here's a guide to handling common data quality challenges:

| Issue                | Detection Method    | Solution Strategy         | Real-World Example                                   |
| -------------------- | ------------------- | ------------------------- | ---------------------------------------------------- |
| Missing Values       | `df.isnull().sum()` | Imputation, deletion      | Customer age missing: Use median age for segment     |
| Outliers             | Z-score, IQR        | Capping, removal          | Order amount $999,999: Cap at 3 std deviations       |
| Duplicates           | `df.duplicated()`   | Remove or merge           | Same order ID with different timestamps: Keep latest |
| Inconsistent Formats | Pattern matching    | Standardization           | Phone numbers: Convert all to +1-XXX-XXX-XXXX        |
| Invalid Values       | Domain validation   | Correction or removal     | Negative prices: Investigate and correct             |
| Typos                | String similarity   | Fuzzy matching            | Product names: "iPhone" vs "i-phone"                 |
| Date Format Issues   | Pattern validation  | Parsing & standardization | Convert all dates to ISO format                      |
| Case Sensitivity     | String operations   | Case normalization        | Email: Convert all to lowercase                      |

## Data Transformation Techniques

### 1. Scaling Methods

```
        price  scaled_price  normalized_price
0  445.942283      1.371460          0.908062
1  447.256087      1.380197          0.910811
2  261.834888      0.147135          0.522892
3  161.384881     -0.520863          0.312740
4         NaN           NaN               NaN
```

Standardization (Z-score)

Import and fit `StandardScaler`, then write the scaled values to a new `scaled_price` column. Output has mean ≈ 0 and std ≈ 1.

Min-Max Scaling

Swap in `MinMaxScaler` to compress prices into the \[0, 1] range, stored in `normalized_price`. The final print compares all three columns.

```
        price  scaled_price  normalized_price
0  445.942283      1.371460          0.908062
1  447.256087      1.380197          0.910811
2  261.834888      0.147135          0.522892
3  161.384881     -0.520863          0.312740
4         NaN           NaN               NaN
```

### 2. Encoding Categorical Variables

```
      category  encoded_category
0  Electronics                 1
1         Home                 2
2         Home                 2
3  Electronics                 1
4         Home                 2
encoded shape: (120, 10)
```

One-Hot Encoding

Use `pd.get_dummies` to expand the `category` column into binary indicator columns, one new column per unique category value.

Label Encoding

Apply scikit-learn's `LabelEncoder` to map each category string to an integer, stored in `encoded_category`. The prints show both representations and the one-hot shape.

```
      category  encoded_category
0  Electronics                 1
1         Home                 2
2         Home                 2
3  Electronics                 1
4         Home                 2
encoded shape: (120, 10)
```

## Best Practices for Data Wrangling

1.  **Document Everything**

    Data cleaning log

    Record what was changed in a dictionary: original row count, whether missing values were handled, how many outliers were removed, and which new features were added.
2.  **Create Reusable Functions**

    Function signature and docstring

    Defines `clean_dataset` and documents what it expects (a DataFrame) and returns (a cleaned DataFrame).

    Cleaning pipeline steps

    Chains the four helper functions in order: handle missing values, remove outliers, create features, then validate before returning the result.
3.  **Validate Transformations**

    Validate transformation results

    Two assertions guard against an empty DataFrame or any remaining nulls; if both pass, prints a success message.

## Performance Considerations

1.  **Memory Efficiency**

    Optimize datatypes

    Iterates over columns and downcasts `float64` → `float32` and `int64` → the smallest integer type that fits, cutting memory usage without changing values.
2.  **Processing Speed**

    Use vectorized operations

    The single-line vectorized multiply (good) runs in C under the hood; the commented-out row loop (avoid) is orders of magnitude slower on large DataFrames.

## Prerequisites

* Python 3.x
*   Key libraries:

    Install required libraries

    One pip command installs all five libraries needed for the module: pandas, NumPy, scikit-learn, matplotlib, and seaborn.

## Tools and Resources

1. **Python Libraries**
   * pandas: Data manipulation
   * numpy: Numerical operations
   * scikit-learn: Data preprocessing
   * matplotlib/seaborn: Visualization
2. **Development Environment**
   * Jupyter Notebook
   * VS Code with Python extension
   * Git for version control
3. **Additional Resources**
   * [Pandas Documentation](https://pandas.pydata.org/docs/)
   * [Data Cleaning Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
   * [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
4. **Session Notebooks**
   * [Data Wrangling Session Notebook (Google Colab)](https://colab.research.google.com/drive/18oENSAH3g8ULCjq7iynC0LdrAj14Dihi?usp=sharing)

## Assignment

Ready to practice your data wrangling skills? Head over to the [Module 2 assignment (student version)](../assignments/module-assignment-student.md) to apply what you have learned.

Transform messy data into analysis-ready datasets!
