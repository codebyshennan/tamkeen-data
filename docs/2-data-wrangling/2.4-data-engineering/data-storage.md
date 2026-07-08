# Data Storage Solutions

**After this lesson:** You can contrast **OLTP** databases, **data warehouses**, and **data lakes** for typical analytics workloads, and name one sensible use case for each.

## Helpful video

DAGs, tasks, and scheduling, conceptual background for ETL-style pipelines.

## Overview

**Prerequisites:** [ETL fundamentals](etl-fundamentals.md) and [Intro to databases](../2.1-sql/intro-databases.md). Optional: skim [Snowflake](../../0-prep/snowflake.md) if your org uses it.

> **Time needed:** About 45-60 minutes.

> **Note:** **OLTP** (online transaction processing) systems optimize row-level transactions; warehouses optimize analytical queries across large history.

## Why this matters

Choosing where data lives, operational database, warehouse, lake, or a mix, shapes **cost**, **latency**, **schema strictness**, and **who** can query comfortably (applications vs analysts vs scientists). You do not need to pick vendors here; you need clear vocabulary for architecture discussions.

## Introduction to Data Storage

Data storage is a fundamental aspect of data engineering that requires careful consideration of various factors to ensure efficient, reliable, and scalable data management.

### Storage Types Comparison Chart

```
+------------------+------------------------+------------------------+------------------------+
| Characteristic   | Data Warehouse         | Data Lake             | Database              |
+------------------+------------------------+------------------------+------------------------+
| Data Structure   | Structured             | Any Structure         | Structured/Semi       |
| Schema           | Schema-on-Write        | Schema-on-Read        | Fixed Schema          |
| Data Quality     | Refined                | Raw                   | Validated             |
| Query Speed      | Fast                   | Varies                | Fast                  |
| Storage Cost     | Higher                 | Lower                 | Medium                |
| Processing       | Batch                  | Batch/Real-time       | Real-time             |
| Use Cases        | BI/Reporting           | Data Science/ML       | OLTP                  |
| Scalability      | Vertical               | Horizontal            | Both                  |
| Tools            | Snowflake, Redshift    | S3, Azure Blob        | PostgreSQL, MongoDB   |
+------------------+------------------------+-----------------------+------------------------+
```

### Data Warehouse Architecture

### Data Lake Organization

### Storage Performance Comparison (Tableau Dashboard)

```
[Tableau Dashboard Layout]
+------------------------+------------------------+
|    Query Performance   |    Storage Metrics     |
+------------------------+------------------------+
| - Response Time        | - Storage Usage        |
| - Throughput          | - Growth Rate          |
| - Concurrency         | - Cost per GB          |
| - Cache Hit Rate      | - Compression Ratio    |
+------------------------+------------------------+
|        Access Patterns by Storage Type         |
+-----------------------------------------------+
| - Read/Write Ratios                           |
| - Query Types                                 |
| - Data Volume                                 |
| - User Concurrency                            |
+-----------------------------------------------+
|        Performance Trends Over Time           |
+-----------------------------------------------+
| - Query Response Time                         |
| - Storage Growth                              |
| - Cost Trends                                 |
| - Resource Usage                              |
+-----------------------------------------------+
```

### Key Considerations

#### 1. Data Characteristics

* **Volume**:
  * Current data size
  * Growth projections
  * Storage capacity planning
  * Cost considerations
* **Velocity**:
  * Data ingestion rate
  * Processing requirements
  * Real-time vs batch
  * Access patterns
* **Variety**:
  * Structured data
  * Semi-structured data
  * Unstructured data
  * Binary objects

#### 2. Performance Requirements

* **Access Patterns**:
  * Read/write ratios
  * Query complexity
  * Concurrency needs
  * Latency requirements
* **Scalability**:
  * Horizontal scaling
  * Vertical scaling
  * Partitioning strategy
  * Load distribution

#### 3. Data Governance

* **Security**:
  * Access control
  * Encryption
  * Audit logging
  * Compliance requirements
* **Data Quality**:
  * Validation rules
  * Consistency checks
  * Data integrity
  * Error handling

#### 4. Operational Aspects

* **Maintenance**:
  * Backup strategies
  * Recovery procedures
  * Monitoring setup
  * Performance tuning
* **Cost Management**:
  * Storage costs
  * Operation costs
  * Scaling costs
  * Maintenance costs

### Storage Selection Criteria

#### 1. Business Requirements

* **Use Cases**:
  * Transaction processing
  * Analytics
  * Archival
  * Caching
* **SLA Requirements**:
  * Availability
  * Durability
  * Performance
  * Recovery time

#### 2. Technical Requirements

* **Data Model**:
  * Schema flexibility
  * Relationship handling
  * Indexing needs
  * Query capabilities
* **Integration**:
  * API support
  * Tool compatibility
  * Protocol support
  * Ecosystem integration

## Types of Data Storage

### 1. Relational Databases (RDBMS)

Imports and declarative base

Imports SQLAlchemy column types, `declarative_base` for ORM model definitions, and `sessionmaker` for connection management. `Base = declarative_base()` is the shared registry that maps Python classes to database tables.

SalesRecord ORM model

Defines the `sales` table as a Python class. Each ``Column(...)` call maps a class attribute to a typed database column, SQLAlchemy infers the DDL from these declarations when create_all runs.``

`setup_database: engine, schema creation, sessionCreates an engine from the connection string, calls Base.metadata.create_all to emit CREATE TABLE statements if the tables don't exist, then returns a bound Session instance ready for queries.2. NoSQL DatabasesNoSQL databases provide flexible schema design and horizontal scalability for handling diverse data types and high-volume workloads.Types of NoSQL Databases`**`Document Stores`**`:Schema flexibilityNested structuresJSON/BSON formatQuery capabilitiesExample: MongoDB, CouchDB`**`Key-Value Stores`**`:Simple data modelHigh performanceScalable architectureCache-friendlyExample: Redis, DynamoDB`**`Column-Family Stores`**`:Wide-column storageHigh write throughputEfficient compressionHorizontal scalingExample: Cassandra, HBase`**`Graph Databases`**`:Relationship-focusedNetwork analysisPath traversalPattern matchingExample: Neo4j, JanusGraphUse Cases`**`Document Stores`**`:Content managementUser profilesGame statesProduct catalogs`**`Key-Value Stores`**`:Session managementShopping cartsUser preferencesReal-time bidding`**`Column-Family`**`:Time-series dataEvent loggingSensor dataLarge-scale analytics`**`Graph Databases`**`:Social networksRecommendation enginesFraud detectionKnowledge graphsHere's a comprehensive MongoDB implementation:Imports, class definition, and constructorImports MongoClient and datetime, defines MongoDBHandler, and stores a client instance. All methods resolve the database and collection at call time, so one handler works across multiple databases.insert_document, find_documents, update_documentThree CRUD methods following the same pattern: resolve db and coll from the client, then call the pymongo method-insert_one, find, or update_many with a $set operator.Usage exampleInstantiates the handler, constructs a document dict with a live timestamp, and inserts it into the transactions collection of sales_db.3. Data LakesImports, class definition, and constructorImports boto3, pandas, and StringIO. The constructor creates an S3 client and stores the bucket name, the single point of configuration for all subsequent operations.upload_dataframe: partition key, CSV serialisation, S3 putOptionally prefixes the key with a Hive-style partition path (e.g., date=2023-01-01/), serialises the DataFrame to a CSV string via StringIO, then uploads the bytes with put_object.read_dataframe and list_filesread_dataframe fetches an object by key and parses the body as CSV directly. list_files uses list_objects_v2 with an optional prefix, returning the key of every matching object.4. Data WarehousesData warehouses are specialized databases optimized for analytics and reporting, providing a centralized repository for integrated data from multiple sources.Architecture Components`**`Staging Area`**`:Raw data landingInitial validationFormat conversionLoad preparation`**`Core Warehouse`**`:Dimensional modelsFact tablesSlowly changing dimensionsHistorical tracking`**`Data Marts`**`:Subject-specific viewsAggregated dataDepartment-specificOptimized accessDesign Patterns`**`Star Schema`**`:Fact tablesDimension tablesDenormalizationQuery optimization`**`Snowflake Schema`**`:Normalized dimensionsReduced redundancyComplex relationshipsStorage efficiencyPerformance Features`**`Columnar Storage`**`:CompressionQuery optimizationParallel processingAnalytical workloads`**`Materialized Views`**`:Pre-computed resultsFaster queriesRefresh strategiesResource optimizationHere's a comprehensive implementation:Imports, class definition, and constructorImports Snowflake's connect and pandas. The constructor unpacks connection_params (account, user, password, warehouse, database) into connect() and stores the live connection.execute_query: cursor, execute, fetchOpens a cursor, runs arbitrary SQL, and returns all rows as a list of tuples. Used internally by load_data to issue DDL commands (CREATE TEMPORARY STAGE, COPY INTO).load_data: stage, write_pandas, COPY INTOThree-step Snowflake bulk load: create a temporary stage, write the DataFrame into it with write_pandas, then issue a COPY INTO command to move the staged CSV data into the target table.Data Storage Patterns1. Data PartitioningSignature, init list, and groupbyTakes a DataFrame and a list of partition column names. Initialises an empty partitions list, then groups the data by those columns, each group becomes one partition file.Hive-style path building and partition listFor multi-column partitions, joins col=val segments with / to produce a Hive-style path like year=2023/month=01. Single-column partitions use a simpler format. Each group is appended as a {'path': ..., 'data': group} dict.2. Data Compressioncompress_data: JSON serialise then gzipImports gzip and json. compress_data serialises the input to a JSON string, encodes it to UTF-8 bytes, then compresses with gzip.compress. Unsupported compression types raise a ValueError.decompress_data: gunzip then JSON parseThe inverse of compress_data: decompresses bytes with gzip.decompress, decodes to a UTF-8 string, then parses with json.loads to restore the original Python object.3. Data VersioningImports, class definition, and constructorImports datetime and hashlib. The constructor only stores the storage path, all version data is computed on demand in save_version.save_version: generate ID, build metadata, persistGenerates a unique version ID, assembles a version_info dict with timestamp, caller metadata, and an MD5 checksum, then saves both the data and the metadata via private helpers. Returns version_info so callers can record the lineage._generate_version_id and _calculate_checksum_generate_version_id combines a datetime stamp with the first 8 characters of the checksum to produce a human-readable, collision-resistant ID. _calculate_checksum converts DataFrames to JSON or other objects to strings before hashing with MD5.Best Practices`**`Choose the Right Storage`**`Consider data structureEvaluate access patternsAccount for scalabilityConsider cost implications`**`Optimize Performance`**`Use appropriate indexingImplement partitioningApply compressionMonitor usage patterns`**`Ensure Data Quality`**`Validate dataMaintain consistencyHandle duplicatesMonitor integrity`**`Security Considerations`**`Implement access controlEncrypt sensitive dataMonitor accessRegular backupsPractice ExerciseImplement a data storage system that:Supports multiple storage backendsHandles data partitioningImplements versioningIncludes data validationProvides monitoring capabilitiesSolution TemplateClass definition, constructor, and _initialize_backendsDefines DataStorage, stores config and a versioning instance, then conditionally initialises each backend (RDBMS, MongoDB, data lake) based on which keys are present in config-making all three optional.store_data: version then dispatch to backendValidates the backend name, versions the data first (so every write is auditable), then dispatches to the appropriate private store method based on the backend string._store_in_rdbms: DataFrame to ORM objectsConverts each DataFrame row to a SalesRecord ORM object using row.to_dict(), bulk-inserts with session.add_all, and commits the transaction._store_in_mongodb and _store_in_data_lakeMongoDB: converts the DataFrame to a list of dicts with to_dict('records') and inserts. Data lake: delegates to DataLakeHandler.upload_dataframe with the key and optional partition from kwargs.Configuration and DataStorage initialisationThe config dict specifies connection strings for all three backends plus a local versioning path. Passing it to DataStorage bootstraps the full multi-backend system in one call.Write to all three backendsDemonstrates storing the same DataFrame in RDBMS, MongoDB (with database and collection kwargs), and the data lake (with a key path and date partition), showing how the same store_data interface works across all backends.Gotchas`**`Schema-on-read in a data lake does not mean schema-free`**`: dropping raw files into S3 without documenting the schema defers the pain, not eliminates it; analysts querying the lake later will infer conflicting schemas from different file vintages, producing silent join errors.`**`session.add_all followed by session.commit is not atomic across multiple batches`**`: if your RDBMS load iterates over chunks and the second chunk fails, the first chunk is already committed; wrap the entire load in a single transaction or use a staging table with a final swap.`**`Storing DataFrames as CSV in a data lake loses dtype information`**`: column types (especially dates and nullable integers) revert to object/string on every read; use Parquet (.to_parquet) to preserve schema and compress storage significantly.`**`Vertical scaling of a data warehouse hits a ceiling faster than expected`**`: the comparison chart lists warehouses as "vertical" scalable; most modern warehouse products (Snowflake, Redshift) do scale horizontally, but if you design for single-node vertical growth you may hit cost and size limits before switching architectures is easy.`**`MongoDB insert_many with to_dict('records') includes the pandas index as a field`**`: if the DataFrame has a non-default index (e.g., customer_id), the index column appears in each document; call reset_index() first or explicitly exclude the index with df.to_dict('records') after resetting.`**`Data versioning writes a new copy per store_data call`**`: if the DataStorage class versions every write unconditionally, frequent small loads will multiply storage consumption and version metadata rapidly; version at logical checkpoints (e.g., end-of-day load), not on every append.Remember: Choose your data storage solution based on your specific requirements and use cases!Next steps`[`Data integration`](data-integration.md)`, moving data between systems`[`ETL fundamentals`](etl-fundamentals.md)`, revisit load patterns`[`Module README`](./)
