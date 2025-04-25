# Introduction to Databases: From Data to Knowledge

## Understanding Databases

A database is an organized collection of structured information stored electronically. Key concepts include:

1. **Data Organization**
   - Structured vs Unstructured Data
   - Records and Fields
   - Tables and Relationships

2. **Data Integrity**
   - Accuracy
   - Consistency
   - Reliability
   - Completeness

3. **Data Access**
   - Concurrent Access
   - Security
   - Performance
   - Scalability

## Types of Databases

### 1. Relational Databases (RDBMS)

- Uses structured tables with rows and columns
- Enforces relationships between tables
- Examples: PostgreSQL, MySQL, Oracle

```sql
-- Example of relational structure
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date TIMESTAMP
);
```

### 2. NoSQL Databases

- Document Stores (MongoDB)
- Key-Value Stores (Redis)
- Column-Family Stores (Cassandra)
- Graph Databases (Neo4j)

### 3. Specialized Databases

- Time-Series Databases (InfluxDB)
- Search Engines (Elasticsearch)
- Vector Databases (Pinecone)
- Spatial Databases (PostGIS)

## Database Design Principles

### 1. Entity-Relationship Model

```sql
-- Example of implementing entities and relationships
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2)
);

CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE product_categories (
    product_id INTEGER REFERENCES products(product_id),
    category_id INTEGER REFERENCES categories(category_id),
    PRIMARY KEY (product_id, category_id)
);
```

### 2. Data Modeling

- Conceptual Model
- Logical Model
- Physical Model

### 3. Normalization Forms

1. **First Normal Form (1NF)**
   - Atomic values
   - No repeating groups

```sql
-- Bad: Non-1NF
CREATE TABLE orders_bad (
    order_id INTEGER,
    products TEXT -- "prod1,prod2,prod3"
);

-- Good: 1NF
CREATE TABLE orders_good (
    order_id INTEGER,
    product_id INTEGER
);
```

2. **Second Normal Form (2NF)**
   - Must be in 1NF
   - No partial dependencies

```sql
-- Bad: Non-2NF
CREATE TABLE order_items_bad (
    order_id INTEGER,
    product_id INTEGER,
    product_name VARCHAR(100), -- Depends only on product_id
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);

-- Good: 2NF
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(100)
);

CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);
```

3. **Third Normal Form (3NF)**
   - Must be in 2NF
   - No transitive dependencies

```sql
-- Bad: Non-3NF
CREATE TABLE employees_bad (
    employee_id INTEGER PRIMARY KEY,
    department_id INTEGER,
    department_name VARCHAR(50), -- Depends on department_id
    salary INTEGER
);

-- Good: 3NF
CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY,
    department_name VARCHAR(50)
);

CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    department_id INTEGER REFERENCES departments(department_id),
    salary INTEGER
);
```

## Database Management Systems (DBMS)

### 1. Core Functions

- Data Storage
- Data Retrieval
- Data Update
- Administration
- Security

### 2. Important Features

```sql
-- Transaction Management
BEGIN;
    -- Operations
COMMIT;

-- Access Control
GRANT SELECT, INSERT ON table_name TO user_name;

-- Backup and Recovery
CREATE EXTENSION pg_dump;
```

### 3. Performance Features

```sql
-- Indexing
CREATE INDEX idx_name ON table_name(column_name);

-- Query Planning
EXPLAIN ANALYZE SELECT * FROM table_name;

-- Caching
SET work_mem = '64MB';
```

## Basic Database Operations

### 1. Database Creation

```sql
-- Create database
CREATE DATABASE my_database;

-- Create schema
CREATE SCHEMA my_schema;

-- Set search path
SET search_path TO my_schema, public;
```

### 2. Table Management

```sql
-- Create table with constraints
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Alter table
ALTER TABLE users 
ADD COLUMN last_login TIMESTAMP,
ADD CONSTRAINT user_status CHECK (last_login IS NULL OR last_login <= CURRENT_TIMESTAMP);

-- Create view
CREATE VIEW active_users AS
SELECT * FROM users
WHERE last_login >= CURRENT_TIMESTAMP - INTERVAL '30 days';
```

### 3. Data Management

```sql
-- Insert data
INSERT INTO users (username, email)
VALUES ('john_doe', 'john@example.com');

-- Update data
UPDATE users 
SET last_login = CURRENT_TIMESTAMP
WHERE username = 'john_doe';

-- Delete data
DELETE FROM users
WHERE last_login < CURRENT_TIMESTAMP - INTERVAL '1 year';
```

## Additional Real-World Examples

### 1. E-commerce Analytics Platform

```sql
-- Track user behavior and product performance
CREATE TABLE user_events (
    event_id BIGSERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    event_type VARCHAR(50),  -- view, add_to_cart, purchase
    product_id INT REFERENCES products(product_id),
    session_id UUID,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    device_info JSONB,
    location POINT
);

-- Create materialized view for real-time analytics
CREATE MATERIALIZED VIEW product_engagement AS
SELECT 
    p.product_id,
    p.name,
    COUNT(DISTINCT CASE WHEN ue.event_type = 'view' THEN ue.user_id END) as unique_views,
    COUNT(DISTINCT CASE WHEN ue.event_type = 'add_to_cart' THEN ue.user_id END) as cart_adds,
    COUNT(DISTINCT CASE WHEN ue.event_type = 'purchase' THEN ue.user_id END) as purchasers,
    ROUND(
        COUNT(DISTINCT CASE WHEN ue.event_type = 'purchase' THEN ue.user_id END)::numeric /
        NULLIF(COUNT(DISTINCT CASE WHEN ue.event_type = 'view' THEN ue.user_id END), 0) * 100,
        2
    ) as conversion_rate
FROM products p
LEFT JOIN user_events ue ON p.product_id = ue.product_id
GROUP BY p.product_id, p.name;
```

### 2. Healthcare Management System

```sql
-- Patient records with privacy considerations
CREATE TABLE patients (
    patient_id SERIAL PRIMARY KEY,
    mrn VARCHAR(50) UNIQUE,  -- Medical Record Number
    first_name VARCHAR(50) ENCRYPTED,
    last_name VARCHAR(50) ENCRYPTED,
    date_of_birth DATE ENCRYPTED,
    contact_info JSONB ENCRYPTED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Medical history with versioning
CREATE TABLE medical_records (
    record_id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(patient_id),
    record_type VARCHAR(50),
    record_data JSONB ENCRYPTED,
    version INT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_by INT REFERENCES staff(staff_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Implement row-level security
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
CREATE POLICY patient_access_policy ON patients
    USING (created_by = CURRENT_USER OR 
           CURRENT_USER IN (SELECT user_id FROM staff WHERE role = 'doctor'));
```

## Performance Optimization Examples

### 1. Indexing Strategies

```sql
-- B-tree index for exact matches and ranges
CREATE INDEX idx_orders_date ON orders(order_date);

-- Hash index for equality comparisons
CREATE INDEX idx_users_email ON users USING HASH (email);

-- GiST index for geometric data
CREATE INDEX idx_locations ON stores USING GIST (location);

-- GIN index for full-text search
CREATE INDEX idx_products_search ON products USING GIN (to_tsvector('english', description));
```

### 2. Partitioning Examples

```sql
-- Range partitioning for time-series data
CREATE TABLE metrics (
    metric_id BIGSERIAL,
    timestamp TIMESTAMP,
    value DECIMAL(10,2),
    metadata JSONB
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE metrics_2023_01 PARTITION OF metrics
    FOR VALUES FROM ('2023-01-01') TO ('2023-02-01');
CREATE TABLE metrics_2023_02 PARTITION OF metrics
    FOR VALUES FROM ('2023-02-01') TO ('2023-03-01');

-- List partitioning for categorical data
CREATE TABLE sales (
    sale_id BIGSERIAL,
    region VARCHAR(50),
    amount DECIMAL(10,2)
) PARTITION BY LIST (region);

-- Create regional partitions
CREATE TABLE sales_north PARTITION OF sales
    FOR VALUES IN ('NORTH');
CREATE TABLE sales_south PARTITION OF sales
    FOR VALUES IN ('SOUTH');
```

### 3. Query Optimization

```sql
-- Use CTEs for better readability and performance
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', sale_date) as month,
        SUM(amount) as revenue
    FROM sales
    WHERE sale_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY DATE_TRUNC('month', sale_date)
),
sales_growth AS (
    SELECT 
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_month_revenue
    FROM monthly_sales
)
SELECT 
    month,
    revenue,
    ROUND(
        ((revenue - prev_month_revenue) / prev_month_revenue * 100)::numeric,
        2
    ) as growth_rate
FROM sales_growth;
```

## Common Pitfalls and Solutions

### 1. Connection Management

```sql
-- Bad: Not closing connections
db_conn = connect_to_db()
do_something(db_conn)
# Connection left open

-- Good: Use connection pooling
WITH connection_pool AS (
    SELECT * FROM dblink('connection_string')
    AS t(id INT, name TEXT)
)
SELECT * FROM connection_pool;
```

### 2. Transaction Management

```sql
-- Bad: No error handling
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- Good: Proper transaction handling
BEGIN;
    SAVEPOINT my_savepoint;
    
    UPDATE accounts 
    SET balance = balance - 100 
    WHERE id = 1;
    
    IF NOT FOUND THEN
        ROLLBACK TO my_savepoint;
        RAISE EXCEPTION 'Account not found';
    END IF;
    
    UPDATE accounts 
    SET balance = balance + 100 
    WHERE id = 2;
    
    IF NOT FOUND THEN
        ROLLBACK TO my_savepoint;
        RAISE EXCEPTION 'Account not found';
    END IF;
    
    COMMIT;
EXCEPTION WHEN OTHERS THEN
    ROLLBACK;
    RAISE;
```

## Interactive Examples with Sample Data

### 1. Customer Analysis

```sql
-- Create sample customer data
INSERT INTO customers (first_name, last_name, email, join_date)
SELECT 
    'Customer' || i as first_name,
    'Last' || i as last_name,
    'customer' || i || '@example.com' as email,
    CURRENT_DATE - (random() * 365)::integer as join_date
FROM generate_series(1, 1000) i;

-- Analyze customer cohorts
WITH cohorts AS (
    SELECT 
        DATE_TRUNC('month', join_date) as cohort_month,
        COUNT(*) as cohort_size
    FROM customers
    GROUP BY DATE_TRUNC('month', join_date)
)
SELECT 
    cohort_month,
    cohort_size,
    SUM(cohort_size) OVER (ORDER BY cohort_month) as cumulative_customers
FROM cohorts
ORDER BY cohort_month;
```

### 2. Product Performance

```sql
-- Generate sample sales data
INSERT INTO sales (product_id, sale_date, quantity, amount)
SELECT 
    (random() * 100)::integer as product_id,
    CURRENT_DATE - (random() * 90)::integer as sale_date,
    (random() * 10 + 1)::integer as quantity,
    (random() * 1000)::numeric(10,2) as amount
FROM generate_series(1, 10000);

-- Analyze product performance
WITH product_metrics AS (
    SELECT 
        product_id,
        COUNT(*) as sale_count,
        SUM(quantity) as units_sold,
        SUM(amount) as revenue,
        AVG(amount) as avg_sale_value
    FROM sales
    GROUP BY product_id
)
SELECT 
    product_id,
    sale_count,
    units_sold,
    ROUND(revenue::numeric, 2) as revenue,
    ROUND(avg_sale_value::numeric, 2) as avg_sale_value,
    NTILE(4) OVER (ORDER BY revenue DESC) as revenue_quartile
FROM product_metrics
ORDER BY revenue DESC;
```

Remember: "A well-designed database is the foundation of any successful application!"
