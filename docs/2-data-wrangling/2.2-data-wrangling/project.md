# E-commerce Data Wrangling Project: From Raw Data to Actionable Insights

**After this lesson:** You deliver a cleaned, documented dataset (or notebook) that shows how you assessed quality, handled **missing values** and **outliers**, and validated results, aligned with the [wrangling lessons](README.md).

## Helpful video

Pandas DataFrames in a quick walkthrough, useful for cleaning and wrangling.

<iframe width="560" height="315" src="https://www.youtube.com/embed/m1_33jhhiLE" title="Learn PANDAS in 5 minutes" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Overview

**Prerequisites:** Complete [data quality](data-quality.md), [missing values](missing-values.md), and [transformations](transformations.md) (or equivalent experience). Same Python stack as the [module README](README.md).

> **Time needed:** Often 6-12 hours including documentation.

## Why this matters

This project is the wrangling capstone: you show **judgment** (what to impute, cap, or flag) and **traceability** (what changed between raw exports and analysis-ready tables), not only that you can call pandas functions.

## Business Context

As a data scientist at "GlobalMart", you face a critical challenge: the company's raw data needs significant cleaning and preparation before it can be used for advanced analytics. This project will guide you through the complete data wrangling process, from initial assessment to final validation.

### Business Objectives

1. **Customer Segmentation**: Identify distinct customer groups for targeted marketing
2. **Pricing Optimization**: Analyze price elasticity and optimize pricing strategies
3. **Recommendation System**: Build a reliable product recommendation engine
4. **Churn Prediction**: Develop early warning system for customer churn

## Project Workflow

{% include mermaid-diagram.html src="2-data-wrangling/2.2-data-wrangling/diagrams/project-1.mmd" %}

## Dataset Description

### Data Schema

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Customer Information
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    registration_date DATE,
    country VARCHAR(50),
    age INT,
    gender VARCHAR(10),
    email VARCHAR(100),
    last_login_date TIMESTAMP
);

-- Transaction Records
CREATE TABLE transactions (
    transaction_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    product_id INT REFERENCES products(product_id),
    timestamp TIMESTAMP,
    amount DECIMAL(10,2),
    payment_method VARCHAR(50),
    device_type VARCHAR(50)
);

-- Product Catalog
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    category VARCHAR(50),
    price DECIMAL(10,2),
    brand VARCHAR(50),
    description TEXT,
    stock_level INT,
    supplier_id INT
);
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Customers table</span>
    </div>
    <div class="code-callout__body">
      <p>Defines the <code>customers</code> entity with demographic fields (<code>age</code>, <code>gender</code>, <code>country</code>), contact info (<code>email</code>), and two date columns that will later be used to validate temporal consistency.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Transactions table</span>
    </div>
    <div class="code-callout__body">
      <p>Records each purchase event. Foreign keys to both <code>customers</code> and <code>products</code> enforce referential integrity; <code>amount</code>, <code>payment_method</code>, and <code>device_type</code> are the primary cleaning and outlier-detection targets.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-32" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Products table</span>
    </div>
    <div class="code-callout__body">
      <p>Holds catalogue attributes: <code>price</code> and <code>stock_level</code> need range validation, <code>category</code> and <code>brand</code> need standardised casing, and <code>description</code> is a free-text field prone to nulls.</p>
    </div>
  </div>
</aside>
</div>

## Implementation Guide

### 1. Data Quality Assessment (20%)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def assess_data_quality(df):
    """Comprehensive data quality assessment"""

    quality_report = {
        'completeness': {},
        'validity': {},
        'consistency': {},
        'uniqueness': {}
    }

    # Completeness check
    quality_report['completeness'] = {
        'missing_values': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df)) * 100
    }

    # Validity check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    quality_report['validity']['numeric_ranges'] = {
        col: {'min': df[col].min(), 'max': df[col].max()}
        for col in numeric_cols
    }

    # Consistency check
    if 'registration_date' in df.columns and 'last_login_date' in df.columns:
        quality_report['consistency']['date_order'] = (
            df['last_login_date'] >= df['registration_date']
        ).mean()

    # Uniqueness check
    quality_report['uniqueness'] = {
        col: df[col].nunique() / len(df)
        for col in df.columns
    }

    return quality_report
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Function signature and results scaffold</span>
    </div>
    <div class="code-callout__body">
      <p>Declares the function and initialises a <code>quality_report</code> dict with four empty sub-dicts, one for each quality dimension (completeness, validity, consistency, uniqueness), so downstream code can populate them independently.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Completeness and validity checks</span>
    </div>
    <div class="code-callout__body">
      <p>Completeness: counts and percentages of null values per column. Validity: iterates over numeric columns to record <code>min</code>/<code>max</code> ranges, useful for spotting impossible values like negative ages or prices above business thresholds.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-36" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Consistency and uniqueness checks, then return</span>
    </div>
    <div class="code-callout__body">
      <p>Consistency: guards with <code>if … in df.columns</code> before comparing <code>last_login_date >= registration_date</code>-the <code>.mean()</code> gives the pass rate. Uniqueness: cardinality ratio per column (values between 0 and 1; close to 1 signals near-unique, close to 0 signals high repetition). The complete report dict is returned.</p>
    </div>
  </div>
</aside>
</div>

### 2. Data Cleaning Implementation (30%)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DataCleaner:
    """Data cleaning pipeline for e-commerce data"""

    def __init__(self, config=None):
        self.config = config or {
            'age_range': (13, 100),
            'price_range': (0, 10000),
            'outlier_threshold': 3
        }

    def clean_customer_data(self, customers_df):
        """Clean customer dataset"""
        df = customers_df.copy()

        # Handle missing values
        df['age'] = df['age'].fillna(df['age'].median())
        df['country'] = df['country'].fillna(df['country'].mode()[0])

        # Fix data types
        df['registration_date'] = pd.to_datetime(df['registration_date'])

        # Validate age range
        df.loc[~df['age'].between(*self.config['age_range']), 'age'] = np.nan

        # Standardize country codes
        df['country'] = df['country'].str.upper()

        return df

    def clean_transaction_data(self, transactions_df):
        """Clean transaction dataset"""
        df = transactions_df.copy()

        # Remove duplicate transactions
        df = df.drop_duplicates(subset=['customer_id', 'timestamp', 'amount'])

        # Handle outliers in amount
        z_scores = np.abs(stats.zscore(df['amount']))
        df.loc[z_scores > self.config['outlier_threshold'], 'amount'] = np.nan

        # Standardize payment methods
        df['payment_method'] = df['payment_method'].str.lower()

        return df
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and configuration</span>
    </div>
    <div class="code-callout__body">
      <p>Defines the <code>DataCleaner</code> class and its <code>__init__</code>. The optional <code>config</code> dict sets domain-specific thresholds (<code>age_range</code>, <code>price_range</code>, <code>outlier_threshold</code>) so the same class works for different business contexts without code changes.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-28" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">clean_customer_data method</span>
    </div>
    <div class="code-callout__body">
      <p>Works on a copy to avoid mutating the original. Fills missing ages with the median and missing countries with the mode, casts the date column, nullifies ages outside the configured range (rather than dropping rows), and uppercases country codes for consistency.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="30-44" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">clean_transaction_data method</span>
    </div>
    <div class="code-callout__body">
      <p>Deduplicates on the natural key (<code>customer_id</code> + <code>timestamp</code> + <code>amount</code>), then uses a z-score threshold to null-out extreme <code>amount</code> values rather than deleting rows, preserving row count for downstream joins. Payment method strings are lowercased for grouping consistency.</p>
    </div>
  </div>
</aside>
</div>

### 3. Feature Engineering (30%)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class FeatureEngineer:
    """Feature engineering for e-commerce data"""

    def create_customer_features(self, customers_df, transactions_df):
        """Create customer-level features"""

        # Customer lifetime value
        clv = transactions_df.groupby('customer_id')['amount'].sum()

        # Purchase frequency
        purchase_freq = transactions_df.groupby('customer_id').size()

        # Average order value
        avg_order = transactions_df.groupby('customer_id')['amount'].mean()

        # Days since last purchase
        last_purchase = transactions_df.groupby('customer_id')['timestamp'].max()
        days_since = (pd.Timestamp.now() - last_purchase).dt.days

        # Combine features
        features = pd.DataFrame({
            'customer_lifetime_value': clv,
            'purchase_frequency': purchase_freq,
            'average_order_value': avg_order,
            'days_since_last_purchase': days_since
        })

        return features

    def create_product_features(self, products_df, transactions_df):
        """Create product-level features"""

        # Sales velocity
        sales_velocity = transactions_df.groupby('product_id').size()

        # Price elasticity
        def calculate_elasticity(group):
            price_pct_change = group['price'].pct_change()
            demand_pct_change = group['quantity'].pct_change()
            return (demand_pct_change / price_pct_change).mean()

        elasticity = transactions_df.groupby('product_id').apply(calculate_elasticity)

        # Combine features
        features = pd.DataFrame({
            'sales_velocity': sales_velocity,
            'price_elasticity': elasticity
        })

        return features
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-28" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">create_customer_features method</span>
    </div>
    <div class="code-callout__body">
      <p>Computes four RFM-style signals per customer from the transactions table: lifetime value (total spend), purchase frequency (transaction count), average order value, and recency (days since last purchase). All four series are assembled into a single <code>DataFrame</code> indexed by <code>customer_id</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="30-50" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">create_product_features method</span>
    </div>
    <div class="code-callout__body">
      <p>Derives two product signals: sales velocity (transaction count per product) and price elasticity. The nested <code>calculate_elasticity</code> helper computes the percentage change in demand divided by the percentage change in price for each product group, a standard economics metric for sensitivity analysis.</p>
    </div>
  </div>
</aside>
</div>

### 4. Data Validation (20%)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def validate_final_dataset(df, validation_rules):
    """Validate the final dataset"""

    validation_results = {
        'completeness': {},
        'consistency': {},
        'statistical_validity': {}
    }

    # Check completeness
    missing_values = df.isnull().sum()
    validation_results['completeness'] = {
        'missing_values': missing_values,
        'missing_percentage': (missing_values / len(df)) * 100
    }

    # Check consistency
    for rule_name, rule_func in validation_rules['consistency'].items():
        validation_results['consistency'][rule_name] = rule_func(df)

    # Statistical validation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    validation_results['statistical_validity'] = {
        col: {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'skew': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        for col in numeric_cols
    }

    return validation_results
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Function signature and results scaffold</span>
    </div>
    <div class="code-callout__body">
      <p>Takes the cleaned <code>df</code> and a <code>validation_rules</code> dict (allowing callers to inject custom rule functions). Initialises three empty sub-dicts so each check populates its own section of the report independently.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Completeness and consistency checks</span>
    </div>
    <div class="code-callout__body">
      <p>Completeness: stores null counts and percentages. Consistency: iterates over caller-supplied rule functions, each receives the full DataFrame and returns a scalar or Series, making the validator extensible without touching this code.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-33" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Statistical validation and return</span>
    </div>
    <div class="code-callout__body">
      <p>For every numeric column, records mean, standard deviation, skewness, and excess kurtosis, a quick distributional health-check. High skew or kurtosis after cleaning can indicate remaining outliers or transformation needs. The full report dict is returned.</p>
    </div>
  </div>
</aside>
</div>

## Project Deliverables

### 1. Code Repository Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── transactions.csv
│   │   └── products.csv
│   └── processed/
│       └── final_dataset.csv
├── notebooks/
│   ├── 1_exploration.ipynb
│   ├── 2_cleaning.ipynb
│   └── 3_transformation.ipynb
├── src/
│   ├── data_quality.py
│   ├── cleaning.py
│   └── transformation.py
└── docs/
    ├── data_dictionary.md
    └── quality_report.md
```

### 2. Quality Report Template

```markdown
# Data Quality Report

## Executive Summary
- Key findings
- Critical issues
- Recommendations

## Detailed Analysis
1. Missing Data
   - Patterns identified
   - Treatment strategies
   - Impact assessment

2. Outliers
   - Detection methods
   - Treatment decisions
   - Business implications

3. Transformations
   - Techniques applied
   - Validation results
   - Performance metrics
```

## Gotchas

- **Filling missing ages with the median before validating the range**: if you call `fillna(median)` first and then null out out-of-range values, you may silently re-introduce the median as a valid age for rows whose original value was actually invalid; validate first, then impute.
- **Z-score outlier detection breaks on skewed `amount` distributions**: z-scores assume approximate normality; transaction amounts are often right-skewed, so a z-score threshold of 3 may miss extreme high-value outliers while flagging legitimate high spenders. Consider IQR or domain-based caps instead.
- **Deduplicating on `(customer_id, timestamp, amount)` misses near-duplicate rows**: two rows with the same customer and amount but slightly different timestamps (e.g., from retry logic) are not caught; add `product_id` to the subset or round timestamps before deduplication.
- **`pd.Timestamp.now()` is timezone-naive by default**: computing `days_since_last_purchase` against a timezone-aware `timestamp` column raises a TypeError; use `pd.Timestamp.now(tz='UTC')` or strip timezone info from the column consistently.
- **`calculate_elasticity` divides by zero silently**: if `price_pct_change` is 0 (no price change in the period), the division produces `inf` or `NaN` without raising an error; add a guard or filter out groups with no price variation before computing elasticity.
- **Quality reports serialise to nested dicts, not DataFrames**: the `assess_data_quality` return value contains Series and scalar objects that may not render cleanly in a Jupyter notebook or export to JSON without explicit conversion; call `.to_dict()` on Series before packaging the report.

## Best Practices

1. **Version Control**
   - Use Git for code versioning
   - Document all data transformations
   - Track data quality metrics

2. **Performance Optimization**
   - Use efficient data structures
   - Implement parallel processing
   - Optimize memory usage

3. **Documentation**
   - Maintain clear documentation
   - Create data dictionaries
   - Document assumptions and decisions

## Evaluation Criteria

1. **Code Quality (30%)**
   - Clean, well-organized code
   - Proper error handling
   - Efficient implementations

2. **Documentation (20%)**
   - Clear explanations
   - Comprehensive data dictionary
   - Well-documented decisions

3. **Results (50%)**
   - Data quality improvements
   - Feature engineering effectiveness
   - Validation metrics

Remember: "The quality of your data wrangling directly impacts the reliability of your analytics!"
