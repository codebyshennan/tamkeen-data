# Data Engineering Assignment

**After this lesson:** You design or prototype an **ETL**-style pipeline (code or diagram plus narrative) with extract, transform, and load steps, basic quality checks, and a plausible failure-handling story.

## Helpful video

DAGs, tasks, and scheduling, conceptual background for ETL-style pipelines.

<iframe width="560" height="315" src="https://www.youtube.com/embed/eeSLDdz-aLg" title="Apache Airflow Tutorial for Beginners" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Overview

**Prerequisites:** [ETL fundamentals](etl-fundamentals.md), [data storage](data-storage.md), and [data integration](data-integration.md). Comfortable Python; optional [Airflow](../../0-prep/airflow.md) if you orchestrate with it.

> **Time needed:** Often 8-15 hours for a solid submission.

## Why this matters

This project checks whether you can narrate a pipeline end to end: **sources**, **transformations**, **load targets**, **quality checks**, and **what happens when a step fails**. Polished code is optional; coherent stages and failure thinking are not.

In this assignment, you design or prototype an ETL-style pipeline that could plausibly serve analysts, emphasizing structure, data-quality gates, and recovery, not only a single successful run.

### Learning Objectives

- Master ETL pipeline development
- Implement data quality controls
- Handle complex transformations
- Ensure system reliability
- Optimize performance
- Monitor pipeline health

## Project Description

You'll be building an enterprise-grade data pipeline for an e-commerce company that processes millions of transactions daily. The system needs to:

### Data Collection

- **Sales Data**:
  - API integration with order system
  - Database connection to inventory
  - File imports from legacy systems
  - Real-time transaction streams

### Data Processing

- **Transformation**:
  - Clean and standardize data
  - Apply business rules
  - Calculate derived metrics
  - Validate data quality

### Data Storage

- **Warehouse Design**:
  - Dimensional modeling
  - Fact table design
  - Incremental loading
  - Performance optimization

### Reporting

- **Analytics Support**:
  - Sales analytics
  - Inventory metrics
  - Customer insights
  - Performance KPIs

## Setup

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Required libraries
import pandas as pd
import numpy as np
import sqlalchemy
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Required libraries</span>
    </div>
    <div class="code-callout__body">
      <p>Seven imports covering data manipulation (pandas, numpy), database connectivity (sqlalchemy), HTTP requests, logging, datetime, and type hints. These cover the full stack needed for an ETL pipeline.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Logging configuration</span>
    </div>
    <div class="code-callout__body">
      <p>Configures a root logger at INFO level with a timestamped format. The module-level <code>logger = logging.getLogger(__name__)</code> creates a named logger so log messages include the module name for easier filtering.</p>
    </div>
  </div>
</aside>
</div>

## Tasks

### 1. Data Source Integration (25 points)

a) API Integration (10 points)

- Implement API client for sales data
- Handle authentication
- Implement error handling and retries
- Add rate limiting

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class SalesAPIClient:
    """
    Implement API client for sales data
    """
    def __init__(self, base_url: str, api_key: str):
        # Your code here
        pass

    def fetch_sales(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch sales data for date range"""
        # Your code here
        pass

    def handle_rate_limit(self):
        """Implement rate limiting"""
        # Your code here
        pass
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and constructor stub</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>SalesAPIClient</code> with typed constructor parameters (<code>base_url</code> and <code>api_key</code>). Your implementation should store these and set up an authenticated session for reuse across calls.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">fetch_sales stub</span>
    </div>
    <div class="code-callout__body">
      <p>Accepts a date range and should return a DataFrame of sales records. Implement pagination, error handling, and retry logic here, this is the main data extraction entry point.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-17" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">handle_rate_limit stub</span>
    </div>
    <div class="code-callout__body">
      <p>Should inspect the <code>Retry-After</code> header (or use exponential backoff) and sleep the appropriate duration before the next request when the API returns a 429 response.</p>
    </div>
  </div>
</aside>
</div>

b) Database Integration (10 points)

- Connect to source database
- Implement efficient query patterns
- Handle large data volumes
- Implement connection pooling

c) File Integration (5 points)

- Handle different file formats
- Implement file validation
- Process files in batches
- Track processed files

### 2. Data Transformation (25 points)

a) Data Cleaning (10 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DataCleaner:
    """
    Implement data cleaning operations
    """
    def clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean sales data
        - Handle missing values
        - Remove duplicates
        - Fix data types
        - Validate data
        """
        # Your code here
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Implement data validation rules
        """
        # Your code here
        pass
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-14" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and clean_sales_data stub</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>DataCleaner</code> with a <code>clean_sales_data</code> method. The docstring lists the four required steps: handle missing values, remove duplicates, fix data types, and validate. Implement all four inside the stub.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">validate_data stub</span>
    </div>
    <div class="code-callout__body">
      <p>Should apply business rules (e.g., non-negative amounts, valid customer IDs) and return <code>True</code> if all rows pass. Called after cleaning so validation failures indicate logic errors, not just dirty data.</p>
    </div>
  </div>
</aside>
</div>

b) Data Enrichment (10 points)

- Add derived columns
- Calculate aggregations
- Join with reference data
- Apply business rules

c) Data Quality (5 points)

- Implement data quality checks
- Generate quality metrics
- Handle validation failures
- Log quality issues

### 3. Data Storage (25 points)

a) Schema Design (10 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class WarehouseSchema:
    """
    Define data warehouse schema
    """
    def create_tables(self, engine):
        """Create warehouse tables"""
        # Define fact table
        fact_sales = """
        CREATE TABLE IF NOT EXISTS fact_sales (
            sale_id INTEGER PRIMARY KEY,
            date_id INTEGER,
            product_id INTEGER,
            customer_id INTEGER,
            quantity INTEGER,
            amount DECIMAL(10,2),
            FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
            FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
            FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id)
        )
        """

        # Define dimension tables
        dim_date = """
        CREATE TABLE IF NOT EXISTS dim_date (
            date_id INTEGER PRIMARY KEY,
            date DATE,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            quarter INTEGER,
            is_weekend BOOLEAN
        )
        """

        # Execute schema creation
        with engine.connect() as conn:
            conn.execute(dim_date)
            conn.execute(fact_sales)
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and method signature</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>WarehouseSchema</code> with a single <code>create_tables</code> method that takes a SQLAlchemy engine. All table DDL is defined as inline SQL strings and executed at the end.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">fact_sales DDL: star-schema fact table</span>
    </div>
    <div class="code-callout__body">
      <p>Defines the central fact table with <code>sale_id</code> as primary key and foreign keys to three dimension tables (<code>dim_date</code>, <code>dim_product</code>, <code>dim_customer</code>). This is a classic star-schema design.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-37" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">dim_date DDL and schema execution</span>
    </div>
    <div class="code-callout__body">
      <p>Defines the date dimension with time attributes (year, month, day, quarter, is_weekend) needed for time-based slicing in reports. The final block opens a connection and executes both DDL statements, dimension first, then fact.</p>
    </div>
  </div>
</aside>
</div>

b) Data Loading (10 points)

- Implement efficient load patterns
- Handle incremental loads
- Manage transactions
- Implement error recovery

c) Performance Optimization (5 points)

- Implement indexing strategy
- Optimize query performance
- Manage data partitioning
- Monitor performance

### 4. Pipeline Orchestration (15 points)

a) Pipeline Implementation (10 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DataPipeline:
    """
    Implement data pipeline
    """
    def __init__(self):
        self.api_client = SalesAPIClient(base_url, api_key)
        self.cleaner = DataCleaner()
        self.warehouse = WarehouseSchema()

    def run_pipeline(self, start_date: str, end_date: str):
        """
        Run complete pipeline
        - Extract data
        - Transform data
        - Load data
        - Handle errors
        - Log progress
        """
        try:
            # Extract
            raw_data = self.api_client.fetch_sales(start_date, end_date)

            # Transform
            clean_data = self.cleaner.clean_sales_data(raw_data)

            # Load
            self.load_to_warehouse(clean_data)

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class definition and constructor</span>
    </div>
    <div class="code-callout__body">
      <p>Defines <code>DataPipeline</code> and wires up the three main components in the constructor: the API client, the data cleaner, and the warehouse schema object, ready for use by <code>run_pipeline</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-31" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">run_pipeline: extract, transform, load, error handling</span>
    </div>
    <div class="code-callout__body">
      <p>Orchestrates the full ETL flow for a date range: fetch raw sales from the API, clean them, then load to the warehouse. Any exception is logged with context before re-raising so the caller knows the pipeline failed.</p>
    </div>
  </div>
</aside>
</div>

b) Monitoring and Logging (5 points)

- Implement logging
- Track metrics
- Monitor performance
- Generate alerts

### 5. Documentation and Testing (10 points)

a) Documentation

- Code documentation
- Architecture diagram
- Setup instructions
- Maintenance guide

b) Testing

- Unit tests
- Integration tests
- Performance tests
- Error handling tests

## Deliverables

1. Python Package containing:
   - All implementation code
   - Tests
   - Documentation
   - Requirements file

2. Technical Documentation:
   - Architecture overview
   - Setup instructions
   - API documentation
   - Maintenance guide

3. Test Results:
   - Unit test results
   - Integration test results
   - Performance metrics
   - Code coverage report

## Evaluation Criteria

- Code quality and organization (20%)
- Implementation completeness (30%)
- Error handling and resilience (20%)
- Documentation quality (15%)
- Test coverage (15%)

## Solution Template

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Configuration
config = {
    'api': {
        'base_url': 'https://api.example.com',
        'api_key': 'your_api_key'
    },
    'database': {
        'warehouse': 'postgresql://localhost/warehouse',
        'source': 'postgresql://localhost/source'
    },
    'files': {
        'input_path': 'data/input',
        'processed_path': 'data/processed',
        'failed_path': 'data/failed'
    }
}

# Pipeline implementation
class SalesDataPipeline:
    """
    Main pipeline implementation
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_components()

    def setup_components(self):
        """Initialize pipeline components"""
        # Setup API client
        self.api_client = SalesAPIClient(
            self.config['api']['base_url'],
            self.config['api']['api_key']
        )

        # Setup database connections
        self.warehouse = sqlalchemy.create_engine(
            self.config['database']['warehouse']
        )

        # Setup data cleaner
        self.cleaner = DataCleaner()

    def run_daily_load(self):
        """Run daily data load"""
        try:
            # Extract data
            sales_data = self.extract_daily_data()

            # Clean and transform
            clean_data = self.cleaner.clean_sales_data(sales_data)

            # Load to warehouse
            self.load_to_warehouse(clean_data)

            logger.info("Daily load completed successfully")

        except Exception as e:
            logger.error(f"Daily load failed: {str(e)}")
            raise

    def extract_daily_data(self) -> pd.DataFrame:
        """Extract daily data from all sources"""
        # Your code here
        pass

    def load_to_warehouse(self, df: pd.DataFrame):
        """Load data to warehouse"""
        # Your code here
        pass

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SalesDataPipeline(config)

    try:
        # Run daily load
        pipeline.run_daily_load()

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-16" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Configuration dict</span>
    </div>
    <div class="code-callout__body">
      <p>Centralises all connection strings and file paths: API base URL and key, warehouse and source database connection strings, and three local directories (input, processed, failed) for file-based sources.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="18-41" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">SalesDataPipeline: class, constructor, and setup_components</span>
    </div>
    <div class="code-callout__body">
      <p>Defines the main pipeline class. The constructor calls <code>setup_components</code>, which initialises the API client, creates the warehouse engine from config, and instantiates the data cleaner, wiring everything together before any pipeline run.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="43-59" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">run_daily_load: extract, clean, load, error handling</span>
    </div>
    <div class="code-callout__body">
      <p>The main execution method: calls the extract stub, passes the result through <code>clean_sales_data</code>, then calls the load stub. Logs success on completion; logs and re-raises on any failure.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="61-81" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Stub methods and main execution block</span>
    </div>
    <div class="code-callout__body">
      <p><code>extract_daily_data</code> and <code>load_to_warehouse</code> are stubs for you to implement. The <code>__main__</code> block instantiates the pipeline with the config and calls <code>run_daily_load</code>, catching any failure and logging it.</p>
    </div>
  </div>
</aside>
</div>

## Gotchas

- **`SalesAPIClient` constructor stores credentials as plain instance attributes**: hardcoding `api_key` as `self.api_key` means it appears in `repr()`, log output, and tracebacks; use environment variables or a secrets manager and never log the key directly.
- **`run_pipeline` re-raises exceptions but leaves no checkpoint**: if the load step fails partway through, re-running the pipeline from `start_date` will re-extract and re-transform data that was already partially loaded; add a staging table or idempotency key so reruns are safe.
- **`conn.execute(dim_date)` then `conn.execute(fact_sales)` without `conn.commit()`**: in SQLAlchemy 2.x, DDL within a connection context is not auto-committed; omitting `conn.commit()` (or using `engine.begin()`) means tables are created inside a rolled-back transaction and the schema silently disappears.
- **Incremental loads without a watermark column will re-insert all historical rows**: the `DataLoading` task in the assignment requires handling incremental loads, but the `load_to_warehouse` stub has no logic for filtering already-loaded records; without a `last_updated_at` or batch ID check, every daily run duplicates the full dataset.
- **`logger.error(f"Daily load failed: {str(e)}")` then `raise` logs the message but loses the stack trace context**: wrapping exceptions with `str(e)` discards the original traceback; use `logger.exception("Daily load failed")` inside the `except` block to log the full stack automatically.
- **File integration with no deduplication on reprocessing**: the assignment requires tracking processed files, but without a persistent manifest (e.g., a database table or a `.done` marker file), restarting after a crash will reprocess already-loaded files and silently double-count records.

## Bonus Challenges

1. **Real-time Processing**
   - Implement streaming data processing
   - Handle real-time updates
   - Implement real-time monitoring
   - Add alerting system

2. **Advanced Features**
   - Add data versioning
   - Implement data lineage
   - Add data quality scoring
   - Implement automated testing

3. **Performance Optimization**
   - Implement parallel processing
   - Optimize memory usage
   - Add caching layer
   - Implement query optimization

Good luck! Remember to focus on building a reliable and maintainable solution!
