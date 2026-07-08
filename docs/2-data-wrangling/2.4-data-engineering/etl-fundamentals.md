---
lesson_resources:
  - label: "DAG Examples"
    url: "https://github.com/codebyshennan/tamkeen-data/tree/main/docs/2-data-wrangling/2.4-data-engineering/dags"
    icon: "download"
---

# ETL Fundamentals

**After this lesson:** You can explain **ETL** (**Extract**, **Transform**, **Load**) as an ordered pipeline, pull data from sources, clean and reshape it, then load it into a target, and connect that idea to orchestration sketches (for example **DAG**-based schedulers).

## Helpful video

DAGs, tasks, and scheduling, conceptual background for ETL-style pipelines.

<iframe width="560" height="315" src="https://www.youtube.com/embed/eeSLDdz-aLg" title="Apache Airflow Tutorial for Beginners" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Overview

**Prerequisites:** [SQL](../2.1-sql/README.md), [Pandas](../../1-data-fundamentals/1.5-data-analysis-pandas/README.md), and [data wrangling](../2.2-data-wrangling/README.md). Skim the **Extract → Transform → Load** diagrams in the next sections before the Airflow-style figures.

> **Time needed:** 90+ minutes; treat long code blocks as reference material.

> **Note:** A **DAG** (Directed Acyclic Graph) is a workflow graph without cycles, common in orchestration tools such as Apache Airflow.

## Why this matters

**ETL** is the shared story: pull data from sources, apply rules and joins, then land it where consumers trust it. **Orchestration** (often DAG-based) turns that story into scheduled, retryable work, so failures are visible and reruns do not corrupt downstream tables.

## Introduction to ETL

ETL (Extract, Transform, Load) is a fundamental process in data engineering that forms the backbone of data integration and warehousing solutions.

### ETL Workflow Diagram

{% include mermaid-diagram.html src="2-data-wrangling/2.4-data-engineering/diagrams/etl-fundamentals-1.mmd" %}

Read left to right: raw extracts land in **Transform** (clean, enrich, validate), then **Load** stages them before they become warehouse tables consumers trust. Missing any step is how "the pipeline ran green" still ships bad data.

### Data Pipeline Architecture with Airflow

{% include mermaid-diagram.html src="2-data-wrangling/2.4-data-engineering/diagrams/etl-fundamentals-2.mmd" %}

Orchestrators such as Airflow model work as a **DAG**: tasks run in dependency order, retries apply per task, and the UI shows which step failed, so you fix the right layer (extract vs transform vs load).

### Error Handling Flowchart

{% include mermaid-diagram.html src="2-data-wrangling/2.4-data-engineering/diagrams/etl-fundamentals-3.mmd" %}

Healthy pipelines assume **sources go away**, **rows fail validation**, and **loads partially complete**. Retries, rollback paths, and alerts are not optional polish, they define whether you can safely rerun without doubling data.

### Airflow DAG Example

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# no-output
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'sales_etl_pipeline',
    default_args=default_args,
    description='Sales data ETL pipeline',
    schedule_interval='0 0 * * *',  # Daily at midnight
    catchup=False
)

# Extract task
extract_task = PythonOperator(
    task_id='extract_sales_data',
    python_callable=extract_sales_data,
    dag=dag
)

# Transform task
transform_task = PythonOperator(
    task_id='transform_sales_data',
    python_callable=transform_sales_data,
    dag=dag
)

# Load task
load_task = PythonOperator(
    task_id='load_sales_data',
    python_callable=load_sales_data,
    dag=dag
)

# Validation task
validate_task = PythonOperator(
    task_id='validate_sales_data',
    python_callable=validate_sales_data,
    dag=dag
)

# Define task dependencies
extract_task >> transform_task >> load_task >> validate_task
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-15" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports and default_args</span>
    </div>
    <div class="code-callout__body">
      <p>Imports Airflow's <code>DAG</code> and <code>PythonOperator</code>. <code>default_args</code> centralises retry policy (3 retries, 5-min delay), failure email alerts, and the pipeline start date, all tasks inherit these unless overridden.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">DAG definition</span>
    </div>
    <div class="code-callout__body">
      <p>Creates the DAG named <code>sales_etl_pipeline</code>, scheduled to run daily at midnight via cron expression <code>0 0 * * *</code>. <code>catchup=False</code> prevents Airflow from back-filling missed runs on first deploy.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-37" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Extract and transform tasks</span>
    </div>
    <div class="code-callout__body">
      <p>Each <code>PythonOperator</code> wraps a Python callable. <code>extract_task</code> pulls raw data; <code>transform_task</code> cleans and reshapes it. Both are registered to the <code>dag</code> object.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="39-54" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load, validate tasks, and dependency chain</span>
    </div>
    <div class="code-callout__body">
      <p><code>load_task</code> writes to the target; <code>validate_task</code> confirms data quality after loading. The <code>&gt;&gt;</code> operator chains them into a linear dependency: extract → transform → load → validate.</p>
    </div>
  </div>
</aside>
</div>

**What to notice:** `default_args` centralizes retries and alerts; `schedule_interval` pins the cadence; `>>` chains task order so **extract → transform → load → validate** is explicit. Your callable names would point at real functions that return or raise on failure.

### Monitoring Dashboard Example (Tableau)

```
[Tableau Dashboard Layout]
+------------------------+------------------------+
|    Pipeline Status     |    Data Quality KPIs   |
+------------------------+------------------------+
| - Success Rate         | - Completeness         |
| - Processing Time      | - Accuracy             |
| - Error Count         | - Timeliness           |
| - Resource Usage      | - Consistency          |
+------------------------+------------------------+
|        Error Distribution by Type              |
+-----------------------------------------------+
| - Connection Errors                            |
| - Validation Failures                          |
| - Processing Errors                            |
| - System Errors                                |
+-----------------------------------------------+
|        Performance Metrics Over Time           |
+-----------------------------------------------+
| - Processing Volume                            |
| - Response Time                                |
| - Resource Utilization                         |
| - Throughput                                   |
+-----------------------------------------------+
```

The sections below spell out **Extract**, **Transform**, and **Load** in more detail. Use them as a checklist when you design a pipeline: for each stage, ask what can fail, what "done" means, and what you log when it is not done.

### Core Concepts

#### 1. Extract

- **Data Sources**:
  - Databases (SQL, NoSQL)
  - APIs and web services
  - File systems (CSV, JSON)
  - Streaming sources
  - Legacy systems

- **Extraction Methods**:
  - Full extraction
  - Incremental extraction
  - Change data capture
  - Event-driven extraction

#### 2. Transform

- **Data Cleaning**:
  - Missing value handling
  - Duplicate removal
  - Error correction
  - Format standardization

- **Data Enhancement**:
  - Enrichment
  - Aggregation
  - Derivation
  - Validation

#### 3. Load

- **Loading Types**:
  - Full load
  - Incremental load
  - Merge load
  - Upsert operations

- **Target Systems**:
  - Data warehouses
  - Data marts
  - Operational databases
  - Analytics platforms

### Business Impact

- **Decision Making**:
  - Real-time insights
  - Historical analysis
  - Predictive modeling
  - Performance monitoring

- **Operational Efficiency**:
  - Process automation
  - Data consistency
  - Resource optimization
  - Error reduction

### Technical Considerations

- **Performance**:
  - Processing speed
  - Resource usage
  - Scalability
  - Optimization

- **Quality**:
  - Data accuracy
  - Completeness
  - Consistency
  - Timeliness

Here's a comprehensive implementation of an ETL pipeline:

## ETL Pipeline Components

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import pandas as pd
import sqlalchemy
import requests
import logging
from datetime import datetime

class ETLPipeline:
    """
    Basic ETL pipeline framework
    """
    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def extract(self, source):
        """Extract data from source"""
        try:
            self.logger.info(f"Starting extraction from {source}")
            # Implementation depends on source type
            if source.endswith('.csv'):
                data = pd.read_csv(source)
            elif source.startswith('http'):
                response = requests.get(source)
                data = pd.DataFrame(response.json())
            else:
                raise ValueError(f"Unsupported source type: {source}")

            self.logger.info(f"Extracted {len(data)} records")
            return data

        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            raise

    def transform(self, data):
        """Transform extracted data"""
        try:
            self.logger.info("Starting transformation")
            # Implement transformation logic
            return data

        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            raise

    def load(self, data, target):
        """Load transformed data to target"""
        try:
            self.logger.info(f"Starting load to {target}")
            # Implementation depends on target type
            if target.endswith('.csv'):
                data.to_csv(target, index=False)
            elif target.startswith('postgresql://'):
                engine = sqlalchemy.create_engine(target)
                data.to_sql('table_name', engine, if_exists='append')
            else:
                raise ValueError(f"Unsupported target type: {target}")

            self.logger.info(f"Loaded {len(data)} records")

        except Exception as e:
            self.logger.error(f"Load failed: {str(e)}")
            raise

    def run(self, source, target):
        """Run the complete ETL pipeline"""
        try:
            # Extract
            raw_data = self.extract(source)

            # Transform
            transformed_data = self.transform(raw_data)

            # Load
            self.load(transformed_data, target)

            self.logger.info("ETL pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-20" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports, class definition, constructor, and logging setup</span>
    </div>
    <div class="code-callout__body">
      <p>Imports pandas, sqlalchemy, requests, logging, and datetime. The constructor calls <code>_setup_logging</code>, which configures a timestamped INFO-level logger, every ETL stage uses this logger for traceability.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-40" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">extract: CSV or HTTP dispatch</span>
    </div>
    <div class="code-callout__body">
      <p>Detects the source type by suffix/prefix: reads a CSV file directly or fetches a URL with <code>requests.get</code> and parses the JSON response. Logs row counts on success; re-raises on failure so the pipeline can catch it.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="42-51" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">transform: stub with logging and error handling</span>
    </div>
    <div class="code-callout__body">
      <p>A minimal transform stub that logs entry and returns data unchanged. Replace the stub with cleaning, enrichment, or aggregation logic, the try/except wrapper ensures errors surface with context.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="53-70" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">load: CSV or PostgreSQL dispatch</span>
    </div>
    <div class="code-callout__body">
      <p>Dispatches by target: writes to CSV with <code>to_csv</code>, or creates a SQLAlchemy engine and appends to a table with <code>to_sql(if_exists='append')</code>. Logs the row count written; re-raises on failure.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="72-88" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">run: orchestrate extract → transform → load</span>
    </div>
    <div class="code-callout__body">
      <p>Calls the three phases in order and logs completion. Any stage failure propagates to the outer except block, which logs the error and re-raises, giving the caller a clear stack trace.</p>
    </div>
  </div>
</aside>
</div>

## Extract Phase

The Extract phase is responsible for retrieving data from various source systems while handling different formats, protocols, and potential issues.

### Key Considerations

- **Source Systems**:
  - Availability windows
  - Access patterns
  - Rate limits
  - Authentication

- **Data Volume**:
  - Batch size
  - Memory constraints
  - Network bandwidth
  - Processing capacity

- **Reliability**:
  - Connection stability
  - Error handling
  - Retry mechanisms
  - Fallback options

### 1. Data Sources

Different data sources require specific handling approaches:

#### Database Sources

- **Relational Databases**:
  - Connection pooling
  - Query optimization
  - Transaction isolation
  - Cursor management

- **NoSQL Databases**:
  - Document retrieval
  - Key-value access
  - Graph traversal
  - Column family queries

#### File Systems

- **Local Files**:
  - File formats
  - Encoding handling
  - Directory structure
  - File locking

- **Cloud Storage**:
  - Access credentials
  - Region selection
  - Transfer optimization
  - Cost management

#### APIs

- **REST APIs**:
  - Authentication
  - Rate limiting
  - Pagination
  - Error handling

- **Streaming APIs**:
  - Connection management
  - Backpressure handling
  - Message ordering
  - State management

Here's a comprehensive implementation:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DataExtractor:
    """
    Handle different types of data extraction
    """
    @staticmethod
    def from_csv(file_path):
        """Extract from CSV file"""
        return pd.read_csv(file_path)

    @staticmethod
    def from_api(url, params=None):
        """Extract from REST API"""
        response = requests.get(url, params=params)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    @staticmethod
    def from_database(connection_string, query):
        """Extract from database"""
        engine = sqlalchemy.create_engine(connection_string)
        return pd.read_sql(query, engine)

    @staticmethod
    def from_json(file_path):
        """Extract from JSON file"""
        return pd.read_json(file_path)
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and from_csv</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>DataExtractor</code> as a collection of static methods, no instance state needed. <code>from_csv</code> is the simplest extractor: a one-liner wrapping <code>pd.read_csv</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">from_api: HTTP GET with error checking</span>
    </div>
    <div class="code-callout__body">
      <p><code>raise_for_status()</code> turns HTTP 4xx/5xx responses into exceptions immediately, so downstream code never silently processes an error response. The JSON body is parsed into a DataFrame.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-26" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">from_database and from_json</span>
    </div>
    <div class="code-callout__body">
      <p><code>from_database</code> creates a SQLAlchemy engine and runs an arbitrary query via <code>pd.read_sql</code>. <code>from_json</code> reads a JSON file directly with <code>pd.read_json</code>-same pattern, different source.</p>
    </div>
  </div>
</aside>
</div>

### 2. Error Handling

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def extract_with_retry(source, max_retries=3):
    """
    Extract data with retry logic
    """
    for attempt in range(max_retries):
        try:
            if source.endswith('.csv'):
                return pd.read_csv(source)
            elif source.startswith('http'):
                response = requests.get(source)
                response.raise_for_status()
                return pd.DataFrame(response.json())
            else:
                raise ValueError(f"Unsupported source: {source}")

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-14" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Signature, retry loop, and source dispatch</span>
    </div>
    <div class="code-callout__body">
      <p>Loops up to <code>max_retries</code> times. On each attempt, dispatches to CSV or HTTP extraction. If extraction succeeds the function returns immediately; otherwise it falls through to the except block.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Exponential backoff on failure</span>
    </div>
    <div class="code-callout__body">
      <p>On the last allowed attempt, re-raises immediately. Otherwise sleeps for <code>2^attempt</code> seconds (1 s, 2 s, 4 s, …) before the next retry, exponential backoff reduces load on a struggling upstream system.</p>
    </div>
  </div>
</aside>
</div>

## Transform Phase

The Transform phase is where raw data is converted into a format suitable for analysis and loading into target systems.

### Transformation Types

- **Data Cleansing**:
  - Missing value handling
  - Outlier detection
  - Error correction
  - Format standardization

- **Data Enrichment**:
  - Lookup operations
  - Derived calculations
  - Data augmentation
  - Feature engineering

- **Data Restructuring**:
  - Schema mapping
  - Normalization
  - Denormalization
  - Aggregation

### Key Considerations

- **Data Quality**:
  - Validation rules
  - Business constraints
  - Data integrity
  - Consistency checks

- **Performance**:
  - Memory usage
  - Processing time
  - Resource allocation
  - Optimization

- **Maintainability**:
  - Code organization
  - Documentation
  - Testing
  - Version control

### 1. Data Cleaning

Data cleaning ensures data quality and consistency:

#### Cleaning Operations

- **Missing Values**:
  - Imputation strategies
  - Default values
  - Removal policies
  - Documentation

- **Duplicates**:
  - Detection methods
  - Resolution strategies
  - Business rules
  - Audit trails

- **Data Types**:
  - Type conversion
  - Format validation
  - Range checking
  - Custom types

Here's a comprehensive implementation:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DataTransformer:
    """
    Handle data transformation operations
    """
    @staticmethod
    def clean_data(df):
        """Basic data cleaning"""
        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = df.fillna({
            'numeric_col': 0,
            'string_col': 'unknown'
        })

        # Fix data types
        df['date_col'] = pd.to_datetime(df['date_col'])

        return df

    @staticmethod
    def validate_data(df, rules):
        """Validate data against rules"""
        for column, rule in rules.items():
            if not df[column].apply(rule).all():
                raise ValueError(f"Validation failed for {column}")
        return df

    @staticmethod
    def transform_data(df, transformations):
        """Apply custom transformations"""
        for column, transformation in transformations.items():
            df[column] = df[column].apply(transformation)
        return df
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-20" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and clean_data</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>DataTransformer</code> as static methods. <code>clean_data</code> runs three steps: remove duplicates with <code>drop_duplicates</code>, fill missing values per column with <code>fillna</code>, and parse a date column with <code>pd.to_datetime</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-28" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">validate_data: rule-based column checks</span>
    </div>
    <div class="code-callout__body">
      <p>Iterates <code>rules</code>-a dict mapping column names to callables, and applies each rule with <code>df[column].apply(rule).all()</code>. Raises <code>ValueError</code> naming the offending column if any row fails.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="30-35" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">transform_data: apply per-column lambdas</span>
    </div>
    <div class="code-callout__body">
      <p>Applies a mapping of <code>{column: transformation}</code> by calling <code>df[column].apply(transformation)</code> for each entry. Transformations are arbitrary callables, scaling, encoding, or string manipulation.</p>
    </div>
  </div>
</aside>
</div>

### 2. Data Validation

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def validate_dataset(df, schema):
    """
    Validate dataset against schema
    """
    errors = []

    # Check columns
    missing_cols = set(schema['required_columns']) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check data types
    for col, dtype in schema['dtypes'].items():
        if col in df.columns and df[col].dtype != dtype:
            errors.append(f"Invalid dtype for {col}: {df[col].dtype} != {dtype}")

    # Check constraints
    for col, constraints in schema['constraints'].items():
        if 'min' in constraints and df[col].min() < constraints['min']:
            errors.append(f"{col} contains values below {constraints['min']}")
        if 'max' in constraints and df[col].max() > constraints['max']:
            errors.append(f"{col} contains values above {constraints['max']}")

    if errors:
        raise ValueError("\n".join(errors))

    return True
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Signature, errors list, and required column check</span>
    </div>
    <div class="code-callout__body">
      <p>Takes a DataFrame and a <code>schema</code> dict. Initialises an <code>errors</code> list, then computes missing required columns via set difference and appends a message if any are absent.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data type check</span>
    </div>
    <div class="code-callout__body">
      <p>Iterates the <code>dtypes</code> dict and compares actual column dtypes against expected. Appends a descriptive error message when there is a mismatch, so all type errors are collected before raising.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-27" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Constraint checks, error raise, and return</span>
    </div>
    <div class="code-callout__body">
      <p>Checks min/max bounds per column from <code>schema['constraints']</code>. After all checks, raises <code>ValueError</code> with all accumulated error messages joined by newlines, or returns <code>True</code> if everything passed.</p>
    </div>
  </div>
</aside>
</div>

## Load Phase

The Load phase is responsible for writing transformed data to target systems efficiently and reliably.

### Loading Strategies

- **Batch Loading**:
  - Full loads
  - Incremental loads
  - Delta loads
  - Merge operations

- **Real-time Loading**:
  - Stream processing
  - Change data capture
  - Event-driven loads
  - Message queues

- **Hybrid Loading**:
  - Micro-batching
  - Lambda architecture
  - Kappa architecture
  - Hybrid patterns

### Key Considerations

- **Performance**:
  - Batch size optimization
  - Parallel loading
  - Index management
  - Resource utilization

- **Data Integrity**:
  - Transaction management
  - Consistency checks
  - Rollback strategies
  - Recovery procedures

- **Target Systems**:
  - System capacity
  - Load windows
  - Concurrency limits
  - Maintenance schedules

### 1. Data Loading

Different loading approaches for various target systems:

#### Database Loading

- **Bulk Loading**:
  - Batch inserts
  - COPY commands
  - Staging tables
  - Partition switching

- **Incremental Loading**:
  - Change tracking
  - Timestamp-based
  - Version-based
  - Merge operations

#### File System Loading

- **File Management**:
  - File naming
  - Directory structure
  - Compression
  - Archival

Here's a comprehensive implementation:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DataLoader:
    """
    Handle different types of data loading
    """
    @staticmethod
    def to_csv(df, file_path):
        """Load to CSV file"""
        df.to_csv(file_path, index=False)

    @staticmethod
    def to_database(df, connection_string, table_name):
        """Load to database"""
        engine = sqlalchemy.create_engine(connection_string)
        df.to_sql(table_name, engine, if_exists='append', index=False)

    @staticmethod
    def to_json(df, file_path):
        """Load to JSON file"""
        df.to_json(file_path, orient='records')
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and to_csv</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>DataLoader</code> as static methods. <code>to_csv</code> writes the DataFrame to a file path, suppressing the index column with <code>index=False</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-14" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">to_database: SQLAlchemy append</span>
    </div>
    <div class="code-callout__body">
      <p>Creates an engine from the connection string and appends rows using <code>df.to_sql(if_exists='append')</code>-safe for incremental loads because it never truncates the existing table.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-19" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">to_json: record-oriented export</span>
    </div>
    <div class="code-callout__body">
      <p>Writes to a JSON file using <code>orient='records'</code>-each row becomes a flat JSON object, which is the format most downstream APIs and tools expect.</p>
    </div>
  </div>
</aside>
</div>

### 2. Error Recovery

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class TransactionLoader:
    """
    Load data with transaction support
    """
    def __init__(self, connection_string):
        self.engine = sqlalchemy.create_engine(connection_string)

    def load_with_transaction(self, df, table_name):
        """Load data within a transaction"""
        with self.engine.begin() as connection:
            try:
                # Create temporary table
                temp_table = f"temp_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                df.to_sql(temp_table, connection, index=False)

                # Move data to final table
                connection.execute(f"""
                    INSERT INTO {table_name}
                    SELECT * FROM {temp_table}
                """)

                # Drop temporary table
                connection.execute(f"DROP TABLE {temp_table}")

            except Exception as e:
                # Transaction will automatically rollback
                raise
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and constructor</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>TransactionLoader</code> and creates a SQLAlchemy engine in the constructor. Storing the engine (not a connection) is correct, engines are thread-safe and manage the connection pool.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-27" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">load_with_transaction: stage, INSERT, DROP, rollback</span>
    </div>
    <div class="code-callout__body">
      <p>Opens an atomic transaction with <code>engine.begin()</code>. Inside: writes the DataFrame to a timestamped temp table, copies rows into the real table with <code>INSERT INTO … SELECT</code>, then drops the temp table. Any exception rolls the entire transaction back automatically.</p>
    </div>
  </div>
</aside>
</div>

## Pipeline Orchestration

Pipeline orchestration manages the execution, monitoring, and maintenance of ETL workflows.

### Orchestration Concepts

- **Workflow Management**:
  - Task scheduling
  - Dependency resolution
  - Resource allocation
  - Error handling

- **Pipeline Patterns**:
  - Sequential processing
  - Parallel execution
  - Fan-out/Fan-in
  - Branching logic

- **State Management**:
  - Checkpointing
  - Recovery points
  - State persistence
  - Failure recovery

### Key Features

- **Scheduling**:
  - Time-based triggers
  - Event-driven execution
  - Dependencies
  - Priorities

- **Monitoring**:
  - Health checks
  - Performance metrics
  - Resource usage
  - SLA compliance

- **Error Handling**:
  - Retry policies
  - Failure notifications
  - Recovery procedures
  - Fallback strategies

### 1. Pipeline Configuration

Configuration management for ETL pipelines:

#### Configuration Types

- **Source Config**:
  - Connection details
  - Authentication
  - Query parameters
  - Rate limits

- **Transform Config**:
  - Business rules
  - Validation rules
  - Mapping rules
  - Processing rules

- **Target Config**:
  - Connection details
  - Table mappings
  - Load options
  - Error handling

Here's a comprehensive implementation:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class PipelineConfig:
    """
    Configure ETL pipeline
    """
    def __init__(self, config_file):
        self.config = self._load_config(config_file)

    def _load_config(self, config_file):
        """Load configuration from file"""
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f)

    def get_source_config(self):
        """Get source configuration"""
        return self.config['source']

    def get_transform_config(self):
        """Get transformation configuration"""
        return self.config['transform']

    def get_target_config(self):
        """Get target configuration"""
        return self.config['target']
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition, constructor, and _load_config</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>PipelineConfig</code> and stores a parsed YAML config dict. <code>_load_config</code> lazily imports <code>yaml</code> and reads the file with <code>yaml.safe_load</code>-the safest YAML loader since it forbids arbitrary Python objects.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-24" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">get_source_config, get_transform_config, get_target_config</span>
    </div>
    <div class="code-callout__body">
      <p>Three thin accessors that return the corresponding top-level config section. Keeping them as methods (rather than direct dict access) allows subclasses to override individual sections or add validation.</p>
    </div>
  </div>
</aside>
</div>

### 2. Pipeline Monitoring

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class PipelineMonitor:
    """
    Monitor ETL pipeline execution
    """
    def __init__(self):
        self.start_time = None
        self.metrics = {}

    def start_pipeline(self):
        """Record pipeline start"""
        self.start_time = datetime.now()
        self.metrics = {
            'records_processed': 0,
            'errors': 0,
            'warnings': 0
        }

    def end_pipeline(self):
        """Record pipeline end"""
        duration = datetime.now() - self.start_time
        self.metrics['duration'] = duration.total_seconds()

        return {
            'start_time': self.start_time,
            'duration': self.metrics['duration'],
            'records_processed': self.metrics['records_processed'],
            'errors': self.metrics['errors'],
            'warnings': self.metrics['warnings']
        }

    def record_metric(self, metric_name, value):
        """Record custom metric"""
        self.metrics[metric_name] = value
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-16" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition, constructor, and start_pipeline</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>PipelineMonitor</code> with <code>start_time</code> and <code>metrics</code> fields. <code>start_pipeline</code> records the wall-clock start time and resets counters for records processed, errors, and warnings.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="18-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">end_pipeline: compute duration and return summary</span>
    </div>
    <div class="code-callout__body">
      <p>Calculates elapsed time with <code>datetime.now() - self.start_time</code>, stores it in metrics, then returns a summary dict covering start time, duration, records processed, errors, and warnings.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="31-33" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">record_metric: store custom KPI</span>
    </div>
    <div class="code-callout__body">
      <p>A generic setter that stores any named metric in the <code>metrics</code> dict, use it to track row counts, validation pass rates, or any other pipeline-specific KPI.</p>
    </div>
  </div>
</aside>
</div>

## Best Practices

1. **Error Handling**
   - Implement proper exception handling
   - Use retries for transient failures
   - Log errors with context
   - Implement fallback mechanisms

2. **Performance**
   - Process data in chunks
   - Use appropriate data types
   - Optimize database operations
   - Monitor resource usage

3. **Monitoring**
   - Track pipeline metrics
   - Set up alerts
   - Monitor data quality
   - Log important events

4. **Testing**
   - Unit test components
   - Integration test pipeline
   - Test with sample data
   - Validate outputs

## Practice Exercise

Build an ETL pipeline that:

1. Extracts data from multiple sources
2. Performs data cleaning and validation
3. Loads data to a target system
4. Includes error handling and monitoring
5. Follows best practices

## Solution Template

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Pipeline implementation
class MyETLPipeline(ETLPipeline):
    def transform(self, data):
        """
        Implement custom transformation logic
        """
        # Clean data
        data = DataTransformer.clean_data(data)

        # Validate data
        schema = {
            'required_columns': ['id', 'value'],
            'dtypes': {'id': 'int64', 'value': 'float64'},
            'constraints': {
                'value': {'min': 0, 'max': 100}
            }
        }
        validate_dataset(data, schema)

        # Transform data
        transformations = {
            'value': lambda x: x * 2
        }
        data = DataTransformer.transform_data(data, transformations)

        return data

# Pipeline execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MyETLPipeline()
    monitor = PipelineMonitor()

    try:
        # Start monitoring
        monitor.start_pipeline()

        # Run pipeline
        pipeline.run(
            source='data.csv',
            target='postgresql://localhost/db'
        )

        # Record metrics
        results = monitor.end_pipeline()
        print(f"Pipeline completed: {results}")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
{% endhighlight %}
```
Pipeline failed: [Errno 2] No such file or directory: 'data.csv'
```

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-26" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">MyETLPipeline: override transform with clean, validate, and transform</span>
    </div>
    <div class="code-callout__body">
      <p>Subclasses <code>ETLPipeline</code> and overrides <code>transform</code>. Runs three steps in sequence: clean with <code>DataTransformer.clean_data</code>, validate against a schema dict, then apply a custom transformation (doubling the <code>value</code> column here).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-49" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Main execution: monitor, run, record metrics, handle errors</span>
    </div>
    <div class="code-callout__body">
      <p>Instantiates the pipeline and monitor, starts monitoring, then runs the pipeline from <code>data.csv</code> to a PostgreSQL target. On success, calls <code>end_pipeline()</code> and prints the summary; on failure, prints the error message.</p>
    </div>
  </div>
</aside>
</div>

```
Pipeline failed: [Errno 2] No such file or directory: 'data.csv'
```

## Gotchas

- **`catchup=False` silently skips all historical runs**: when you first deploy an Airflow DAG with a past `start_date`, Airflow will back-fill by default; setting `catchup=False` prevents this, but if you forget it on a pipeline that processes yesterday's data you may ship incomplete history without any error.
- **Passing data between tasks with `return` values does not work in Airflow by default**: `PythonOperator` callables return values via XCom, but XCom has a size limit (48 KB by default); passing large DataFrames between extract and transform tasks will silently truncate or fail, write intermediate results to shared storage (S3, a temp table) instead.
- **`retries: 3` retries the entire task, including any side effects**: if your load task partially wrote rows before failing, retrying without an idempotency guard (e.g., `INSERT OR REPLACE`, a staging table, or a delete-then-insert pattern) will duplicate data in the target.
- **`validate_dataset` raising inside `transform` skips the load but leaves the partial state**: if transform validates row-by-row and raises on the first bad row, any rows already written to a staging area are orphaned; structure validation as an all-or-nothing pass/fail before any writes.
- **`schedule_interval='0 0 * * *'` runs at midnight UTC, not the analyst's local timezone**, this is a common source of off-by-one day errors in daily aggregations; explicitly set `timezone` in the DAG definition or document the UTC assumption.
- **`DataTransformer.transform_data` applying a lambda silently coerces `NaN`**: operations like `lambda x: x * 2` will propagate `NaN` without warning; rows nulled out during cleaning will produce `NaN` derived columns that look valid until a downstream model or BI tool chokes on them.

Remember: A well-designed ETL pipeline is important for reliable data processing!

## Next steps

- [Data engineering project](project.md), apply ETL ideas in one brief (last step in this submodule)
- [Module README](README.md), assignments and context
- Next in the course: [Data visualization (Module 3)](../../3-data-visualization/README.md) when you are ready to present findings
