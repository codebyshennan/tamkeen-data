---
lesson_resources:
  - label: DAG Examples
    url: >-
      https://github.com/codebyshennan/tamkeen-data/tree/main/docs/2-data-wrangling/2.4-data-engineering/dags
    icon: download
---

# ETL Fundamentals

**After this lesson:** You can explain **ETL** (**Extract**, **Transform**, **Load**) as an ordered pipeline, pull data from sources, clean and reshape it, then load it into a target, and connect that idea to orchestration sketches (for example **DAG**-based schedulers).

## Helpful video

DAGs, tasks, and scheduling, conceptual background for ETL-style pipelines.

## Overview

**Prerequisites:** [SQL](../2.1-sql/), [Pandas](../../1-data-fundamentals/1.5-data-analysis-pandas/), and [data wrangling](../2.2-data-wrangling/). Skim the **Extract → Transform → Load** diagrams in the next sections before the Airflow-style figures.

> **Time needed:** 90+ minutes; treat long code blocks as reference material.

> **Note:** A **DAG** (Directed Acyclic Graph) is a workflow graph without cycles, common in orchestration tools such as Apache Airflow.

## Why this matters

**ETL** is the shared story: pull data from sources, apply rules and joins, then land it where consumers trust it. **Orchestration** (often DAG-based) turns that story into scheduled, retryable work, so failures are visible and reruns do not corrupt downstream tables.

## Introduction to ETL

ETL (Extract, Transform, Load) is a fundamental process in data engineering that forms the backbone of data integration and warehousing solutions.

### ETL Workflow Diagram

Read left to right: raw extracts land in **Transform** (clean, enrich, validate), then **Load** stages them before they become warehouse tables consumers trust. Missing any step is how "the pipeline ran green" still ships bad data.

### Data Pipeline Architecture with Airflow

Orchestrators such as Airflow model work as a **DAG**: tasks run in dependency order, retries apply per task, and the UI shows which step failed, so you fix the right layer (extract vs transform vs load).

### Error Handling Flowchart

Healthy pipelines assume **sources go away**, **rows fail validation**, and **loads partially complete**. Retries, rollback paths, and alerts are not optional polish, they define whether you can safely rerun without doubling data.

### Airflow DAG Example

Imports and default\_args

Imports Airflow's `DAG` and `PythonOperator`. `default_args` centralises retry policy (3 retries, 5-min delay), failure email alerts, and the pipeline start date, all tasks inherit these unless overridden.

DAG definition

Creates the DAG named `sales_etl_pipeline`, scheduled to run daily at midnight via cron expression `0 0 * * *`. `catchup=False` prevents Airflow from back-filling missed runs on first deploy.

Extract and transform tasks

Each `PythonOperator` wraps a Python callable. `extract_task` pulls raw data; `transform_task` cleans and reshapes it. Both are registered to the `dag` object.

Load, validate tasks, and dependency chain

`load_task` writes to the target; `validate_task` confirms data quality after loading. The `>>` operator chains them into a linear dependency: extract → transform → load → validate.

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

* **Data Sources**:
  * Databases (SQL, NoSQL)
  * APIs and web services
  * File systems (CSV, JSON)
  * Streaming sources
  * Legacy systems
* **Extraction Methods**:
  * Full extraction
  * Incremental extraction
  * Change data capture
  * Event-driven extraction

#### 2. Transform

* **Data Cleaning**:
  * Missing value handling
  * Duplicate removal
  * Error correction
  * Format standardization
* **Data Enhancement**:
  * Enrichment
  * Aggregation
  * Derivation
  * Validation

#### 3. Load

* **Loading Types**:
  * Full load
  * Incremental load
  * Merge load
  * Upsert operations
* **Target Systems**:
  * Data warehouses
  * Data marts
  * Operational databases
  * Analytics platforms

### Business Impact

* **Decision Making**:
  * Real-time insights
  * Historical analysis
  * Predictive modeling
  * Performance monitoring
* **Operational Efficiency**:
  * Process automation
  * Data consistency
  * Resource optimization
  * Error reduction

### Technical Considerations

* **Performance**:
  * Processing speed
  * Resource usage
  * Scalability
  * Optimization
* **Quality**:
  * Data accuracy
  * Completeness
  * Consistency
  * Timeliness

Here's a comprehensive implementation of an ETL pipeline:

## ETL Pipeline Components

Imports, class definition, constructor, and logging setup

Imports pandas, sqlalchemy, requests, logging, and datetime. The constructor calls `_setup_logging`, which configures a timestamped INFO-level logger, every ETL stage uses this logger for traceability.

extract: CSV or HTTP dispatch

Detects the source type by suffix/prefix: reads a CSV file directly or fetches a URL with `requests.get` and parses the JSON response. Logs row counts on success; re-raises on failure so the pipeline can catch it.

transform: stub with logging and error handling

A minimal transform stub that logs entry and returns data unchanged. Replace the stub with cleaning, enrichment, or aggregation logic, the try/except wrapper ensures errors surface with context.

load: CSV or PostgreSQL dispatch

Dispatches by target: writes to CSV with `to_csv`, or creates a SQLAlchemy engine and appends to a table with `to_sql(if_exists='append')`. Logs the row count written; re-raises on failure.

run: orchestrate extract → transform → load

Calls the three phases in order and logs completion. Any stage failure propagates to the outer except block, which logs the error and re-raises, giving the caller a clear stack trace.

## Extract Phase

The Extract phase is responsible for retrieving data from various source systems while handling different formats, protocols, and potential issues.

### Key Considerations

* **Source Systems**:
  * Availability windows
  * Access patterns
  * Rate limits
  * Authentication
* **Data Volume**:
  * Batch size
  * Memory constraints
  * Network bandwidth
  * Processing capacity
* **Reliability**:
  * Connection stability
  * Error handling
  * Retry mechanisms
  * Fallback options

### 1. Data Sources

Different data sources require specific handling approaches:

#### Database Sources

* **Relational Databases**:
  * Connection pooling
  * Query optimization
  * Transaction isolation
  * Cursor management
* **NoSQL Databases**:
  * Document retrieval
  * Key-value access
  * Graph traversal
  * Column family queries

#### File Systems

* **Local Files**:
  * File formats
  * Encoding handling
  * Directory structure
  * File locking
* **Cloud Storage**:
  * Access credentials
  * Region selection
  * Transfer optimization
  * Cost management

#### APIs

* **REST APIs**:
  * Authentication
  * Rate limiting
  * Pagination
  * Error handling
* **Streaming APIs**:
  * Connection management
  * Backpressure handling
  * Message ordering
  * State management

Here's a comprehensive implementation:

Class definition and from\_csv

Defines `DataExtractor` as a collection of static methods, no instance state needed. `from_csv` is the simplest extractor: a one-liner wrapping `pd.read_csv`.

from\_api: HTTP GET with error checking

`raise_for_status()` turns HTTP 4xx/5xx responses into exceptions immediately, so downstream code never silently processes an error response. The JSON body is parsed into a DataFrame.

from\_database and from\_json

`from_database` creates a SQLAlchemy engine and runs an arbitrary query via `pd.read_sql`. `from_json` reads a JSON file directly with `pd.read_json`-same pattern, different source.

### 2. Error Handling

Signature, retry loop, and source dispatch

Loops up to `max_retries` times. On each attempt, dispatches to CSV or HTTP extraction. If extraction succeeds the function returns immediately; otherwise it falls through to the except block.

Exponential backoff on failure

On the last allowed attempt, re-raises immediately. Otherwise sleeps for `2^attempt` seconds (1 s, 2 s, 4 s, …) before the next retry, exponential backoff reduces load on a struggling upstream system.

## Transform Phase

The Transform phase is where raw data is converted into a format suitable for analysis and loading into target systems.

### Transformation Types

* **Data Cleansing**:
  * Missing value handling
  * Outlier detection
  * Error correction
  * Format standardization
* **Data Enrichment**:
  * Lookup operations
  * Derived calculations
  * Data augmentation
  * Feature engineering
* **Data Restructuring**:
  * Schema mapping
  * Normalization
  * Denormalization
  * Aggregation

### Key Considerations

* **Data Quality**:
  * Validation rules
  * Business constraints
  * Data integrity
  * Consistency checks
* **Performance**:
  * Memory usage
  * Processing time
  * Resource allocation
  * Optimization
* **Maintainability**:
  * Code organization
  * Documentation
  * Testing
  * Version control

### 1. Data Cleaning

Data cleaning ensures data quality and consistency:

#### Cleaning Operations

* **Missing Values**:
  * Imputation strategies
  * Default values
  * Removal policies
  * Documentation
* **Duplicates**:
  * Detection methods
  * Resolution strategies
  * Business rules
  * Audit trails
* **Data Types**:
  * Type conversion
  * Format validation
  * Range checking
  * Custom types

Here's a comprehensive implementation:

Class definition and clean\_data

Defines `DataTransformer` as static methods. `clean_data` runs three steps: remove duplicates with `drop_duplicates`, fill missing values per column with `fillna`, and parse a date column with `pd.to_datetime`.

validate\_data: rule-based column checks

Iterates `rules`-a dict mapping column names to callables, and applies each rule with `df[column].apply(rule).all()`. Raises `ValueError` naming the offending column if any row fails.

transform\_data: apply per-column lambdas

Applies a mapping of `{column: transformation}` by calling `df[column].apply(transformation)` for each entry. Transformations are arbitrary callables, scaling, encoding, or string manipulation.

### 2. Data Validation

Signature, errors list, and required column check

Takes a DataFrame and a `schema` dict. Initialises an `errors` list, then computes missing required columns via set difference and appends a message if any are absent.

Data type check

Iterates the `dtypes` dict and compares actual column dtypes against expected. Appends a descriptive error message when there is a mismatch, so all type errors are collected before raising.

Constraint checks, error raise, and return

Checks min/max bounds per column from `schema['constraints']`. After all checks, raises `ValueError` with all accumulated error messages joined by newlines, or returns `True` if everything passed.

## Load Phase

The Load phase is responsible for writing transformed data to target systems efficiently and reliably.

### Loading Strategies

* **Batch Loading**:
  * Full loads
  * Incremental loads
  * Delta loads
  * Merge operations
* **Real-time Loading**:
  * Stream processing
  * Change data capture
  * Event-driven loads
  * Message queues
* **Hybrid Loading**:
  * Micro-batching
  * Lambda architecture
  * Kappa architecture
  * Hybrid patterns

### Key Considerations

* **Performance**:
  * Batch size optimization
  * Parallel loading
  * Index management
  * Resource utilization
* **Data Integrity**:
  * Transaction management
  * Consistency checks
  * Rollback strategies
  * Recovery procedures
* **Target Systems**:
  * System capacity
  * Load windows
  * Concurrency limits
  * Maintenance schedules

### 1. Data Loading

Different loading approaches for various target systems:

#### Database Loading

* **Bulk Loading**:
  * Batch inserts
  * COPY commands
  * Staging tables
  * Partition switching
* **Incremental Loading**:
  * Change tracking
  * Timestamp-based
  * Version-based
  * Merge operations

#### File System Loading

* **File Management**:
  * File naming
  * Directory structure
  * Compression
  * Archival

Here's a comprehensive implementation:

Class definition and to\_csv

Defines `DataLoader` as static methods. `to_csv` writes the DataFrame to a file path, suppressing the index column with `index=False`.

to\_database: SQLAlchemy append

Creates an engine from the connection string and appends rows using `df.to_sql(if_exists='append')`-safe for incremental loads because it never truncates the existing table.

to\_json: record-oriented export

Writes to a JSON file using `orient='records'`-each row becomes a flat JSON object, which is the format most downstream APIs and tools expect.

### 2. Error Recovery

Class definition and constructor

Defines `TransactionLoader` and creates a SQLAlchemy engine in the constructor. Storing the engine (not a connection) is correct, engines are thread-safe and manage the connection pool.

load\_with\_transaction: stage, INSERT, DROP, rollback

Opens an atomic transaction with `engine.begin()`. Inside: writes the DataFrame to a timestamped temp table, copies rows into the real table with `INSERT INTO … SELECT`, then drops the temp table. Any exception rolls the entire transaction back automatically.

## Pipeline Orchestration

Pipeline orchestration manages the execution, monitoring, and maintenance of ETL workflows.

### Orchestration Concepts

* **Workflow Management**:
  * Task scheduling
  * Dependency resolution
  * Resource allocation
  * Error handling
* **Pipeline Patterns**:
  * Sequential processing
  * Parallel execution
  * Fan-out/Fan-in
  * Branching logic
* **State Management**:
  * Checkpointing
  * Recovery points
  * State persistence
  * Failure recovery

### Key Features

* **Scheduling**:
  * Time-based triggers
  * Event-driven execution
  * Dependencies
  * Priorities
* **Monitoring**:
  * Health checks
  * Performance metrics
  * Resource usage
  * SLA compliance
* **Error Handling**:
  * Retry policies
  * Failure notifications
  * Recovery procedures
  * Fallback strategies

### 1. Pipeline Configuration

Configuration management for ETL pipelines:

#### Configuration Types

* **Source Config**:
  * Connection details
  * Authentication
  * Query parameters
  * Rate limits
* **Transform Config**:
  * Business rules
  * Validation rules
  * Mapping rules
  * Processing rules
* **Target Config**:
  * Connection details
  * Table mappings
  * Load options
  * Error handling

Here's a comprehensive implementation:

Class definition, constructor, and \_load\_config

Defines `PipelineConfig` and stores a parsed YAML config dict. `_load_config` lazily imports `yaml` and reads the file with `yaml.safe_load`-the safest YAML loader since it forbids arbitrary Python objects.

get\_source\_config, get\_transform\_config, get\_target\_config

Three thin accessors that return the corresponding top-level config section. Keeping them as methods (rather than direct dict access) allows subclasses to override individual sections or add validation.

### 2. Pipeline Monitoring

Class definition, constructor, and start\_pipeline

Defines `PipelineMonitor` with `start_time` and `metrics` fields. `start_pipeline` records the wall-clock start time and resets counters for records processed, errors, and warnings.

end\_pipeline: compute duration and return summary

Calculates elapsed time with `datetime.now() - self.start_time`, stores it in metrics, then returns a summary dict covering start time, duration, records processed, errors, and warnings.

record\_metric: store custom KPI

A generic setter that stores any named metric in the `metrics` dict, use it to track row counts, validation pass rates, or any other pipeline-specific KPI.

## Best Practices

1. **Error Handling**
   * Implement proper exception handling
   * Use retries for transient failures
   * Log errors with context
   * Implement fallback mechanisms
2. **Performance**
   * Process data in chunks
   * Use appropriate data types
   * Optimize database operations
   * Monitor resource usage
3. **Monitoring**
   * Track pipeline metrics
   * Set up alerts
   * Monitor data quality
   * Log important events
4. **Testing**
   * Unit test components
   * Integration test pipeline
   * Test with sample data
   * Validate outputs

## Practice Exercise

Build an ETL pipeline that:

1. Extracts data from multiple sources
2. Performs data cleaning and validation
3. Loads data to a target system
4. Includes error handling and monitoring
5. Follows best practices

## Solution Template

\`\`\` Pipeline failed: \[Errno 2] No such file or directory: 'data.csv' \`\`\`MyETLPipeline: override transform with clean, validate, and transform

Subclasses `ETLPipeline` and overrides `transform`. Runs three steps in sequence: clean with `DataTransformer.clean_data`, validate against a schema dict, then apply a custom transformation (doubling the `value` column here).

Main execution: monitor, run, record metrics, handle errors

Instantiates the pipeline and monitor, starts monitoring, then runs the pipeline from `data.csv` to a PostgreSQL target. On success, calls `end_pipeline()` and prints the summary; on failure, prints the error message.

```
Pipeline failed: [Errno 2] No such file or directory: 'data.csv'
```

## Gotchas

* **`catchup=False` silently skips all historical runs**: when you first deploy an Airflow DAG with a past `start_date`, Airflow will back-fill by default; setting `catchup=False` prevents this, but if you forget it on a pipeline that processes yesterday's data you may ship incomplete history without any error.
* **Passing data between tasks with `return` values does not work in Airflow by default**: `PythonOperator` callables return values via XCom, but XCom has a size limit (48 KB by default); passing large DataFrames between extract and transform tasks will silently truncate or fail, write intermediate results to shared storage (S3, a temp table) instead.
* **`retries: 3` retries the entire task, including any side effects**: if your load task partially wrote rows before failing, retrying without an idempotency guard (e.g., `INSERT OR REPLACE`, a staging table, or a delete-then-insert pattern) will duplicate data in the target.
* **`validate_dataset` raising inside `transform` skips the load but leaves the partial state**: if transform validates row-by-row and raises on the first bad row, any rows already written to a staging area are orphaned; structure validation as an all-or-nothing pass/fail before any writes.
* **`schedule_interval='0 0 * * *'` runs at midnight UTC, not the analyst's local timezone**, this is a common source of off-by-one day errors in daily aggregations; explicitly set `timezone` in the DAG definition or document the UTC assumption.
* **`DataTransformer.transform_data` applying a lambda silently coerces `NaN`**: operations like `lambda x: x * 2` will propagate `NaN` without warning; rows nulled out during cleaning will produce `NaN` derived columns that look valid until a downstream model or BI tool chokes on them.

Remember: A well-designed ETL pipeline is important for reliable data processing!

## Next steps

* [Data engineering project](project.md), apply ETL ideas in one brief (last step in this submodule)
* [Module README](./), assignments and context
* Next in the course: [Data visualization (Module 3)](../../3-data-visualization/) when you are ready to present findings
