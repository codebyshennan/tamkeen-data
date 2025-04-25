# Mastering Basic SQL Operations: Your Data Query Journey

## Introduction to SQL Basics

SQL (Structured Query Language) is the standard language for managing and manipulating relational databases. Understanding basic SQL operations is crucial for:

- Data retrieval and analysis
- Database management
- Data integrity maintenance
- Application development

## CRUD Operations

### 1. CREATE: Adding Data

```sql
-- Create a new table
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert single row
INSERT INTO customers (first_name, last_name, email)
VALUES ('John', 'Doe', 'john.doe@email.com');

-- Insert multiple rows
INSERT INTO customers (first_name, last_name, email)
VALUES 
    ('Jane', 'Smith', 'jane.smith@email.com'),
    ('Bob', 'Johnson', 'bob.johnson@email.com');
```

### 2. READ: Querying Data

```sql
-- Select all columns
SELECT * FROM customers;

-- Select specific columns
SELECT first_name, last_name, email 
FROM customers;

-- Basic filtering
SELECT * FROM customers
WHERE last_name = 'Smith';

-- Pattern matching
SELECT * FROM customers
WHERE email LIKE '%@email.com';
```

### 3. UPDATE: Modifying Data

```sql
-- Update single record
UPDATE customers
SET email = 'new.email@email.com'
WHERE customer_id = 1;

-- Update multiple records
UPDATE customers
SET created_at = CURRENT_TIMESTAMP
WHERE created_at IS NULL;

-- Update with conditions
UPDATE customers
SET 
    first_name = INITCAP(first_name),
    last_name = INITCAP(last_name)
WHERE 
    first_name != INITCAP(first_name) OR
    last_name != INITCAP(last_name);
```

### 4. DELETE: Removing Data

```sql
-- Delete specific records
DELETE FROM customers
WHERE customer_id = 1;

-- Delete with conditions
DELETE FROM customers
WHERE created_at < CURRENT_DATE - INTERVAL '1 year';

-- Delete all records
TRUNCATE TABLE customers;
```

## Basic Query Structure

### 1. SELECT Statement Anatomy

```sql
SELECT 
    column1,
    column2,
    column3 AS alias,
    CONCAT(column4, ' ', column5) as derived_column
FROM table_name
WHERE condition
GROUP BY column1
HAVING group_condition
ORDER BY column3 DESC
LIMIT 10;
```

### 2. Filtering and Sorting

```sql
-- Basic WHERE clauses
SELECT * FROM products
WHERE 
    category = 'Electronics' AND
    price >= 100 AND
    stock_quantity > 0;

-- Multiple conditions
SELECT * FROM orders
WHERE 
    status IN ('pending', 'processing') AND
    order_date BETWEEN 
        CURRENT_DATE - INTERVAL '30 days' 
        AND CURRENT_DATE;

-- Pattern matching
SELECT * FROM customers
WHERE 
    email LIKE '%.com' AND
    first_name ILIKE 'j%';  -- Case-insensitive

-- Sorting results
SELECT 
    product_name,
    price,
    stock_quantity
FROM products
ORDER BY 
    price DESC,
    product_name ASC;
```

## Data Types and Constraints

### 1. Common Data Types

```sql
CREATE TABLE products (
    -- Numeric types
    product_id SERIAL PRIMARY KEY,
    price DECIMAL(10,2),
    weight INTEGER,
    
    -- String types
    name VARCHAR(100),
    description TEXT,
    
    -- Date/Time types
    created_at TIMESTAMP,
    sale_date DATE,
    
    -- Boolean type
    is_active BOOLEAN,
    
    -- Enumerated type
    status product_status
);
```

### 2. Constraints

```sql
CREATE TABLE orders (
    -- Primary Key
    order_id SERIAL PRIMARY KEY,
    
    -- Foreign Key
    customer_id INTEGER REFERENCES customers(customer_id),
    
    -- Not Null
    order_date TIMESTAMP NOT NULL,
    
    -- Unique
    tracking_number VARCHAR(50) UNIQUE,
    
    -- Check constraint
    total_amount DECIMAL(10,2) CHECK (total_amount >= 0),
    
    -- Default value
    status VARCHAR(20) DEFAULT 'pending'
);
```

## Table Relationships

### 1. One-to-Many Relationship

```sql
CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(category_id),
    name VARCHAR(100) NOT NULL
);
```

### 2. Many-to-Many Relationship

```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    order_date TIMESTAMP NOT NULL
);

CREATE TABLE order_items (
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    price_at_time DECIMAL(10,2) NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
```

## Basic Joins

### 1. INNER JOIN

```sql
-- Get all orders with customer information
SELECT 
    o.order_id,
    o.order_date,
    c.first_name,
    c.last_name,
    o.total_amount
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id;
```

### 2. LEFT JOIN

```sql
-- Get all customers and their orders (if any)
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    COUNT(o.order_id) as order_count,
    COALESCE(SUM(o.total_amount), 0) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name;
```

### 3. Multiple Joins

```sql
-- Get order details with product and customer information
SELECT 
    o.order_id,
    c.first_name || ' ' || c.last_name as customer_name,
    p.name as product_name,
    oi.quantity,
    oi.price_at_time,
    oi.quantity * oi.price_at_time as line_total
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
ORDER BY o.order_id, p.name;
```

## Additional Real-World Business Scenarios

### 1. E-commerce Order Analytics

```sql
-- Comprehensive order analysis with multiple metrics
WITH order_metrics AS (
    SELECT 
        DATE_TRUNC('day', order_date) as order_day,
        COUNT(*) as total_orders,
        COUNT(DISTINCT customer_id) as unique_customers,
        SUM(total_amount) as revenue,
        AVG(total_amount) as avg_order_value,
        COUNT(DISTINCT CASE 
            WHEN customer_id NOT IN (
                SELECT customer_id 
                FROM orders o2 
                WHERE o2.order_date < o.order_date
            ) THEN customer_id 
        END) as new_customers
    FROM orders o
    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY DATE_TRUNC('day', order_date)
)
SELECT 
    order_day,
    total_orders,
    unique_customers,
    ROUND(revenue::numeric, 2) as revenue,
    ROUND(avg_order_value::numeric, 2) as aov,
    new_customers,
    ROUND(
        (new_customers::float / NULLIF(unique_customers, 0) * 100)::numeric,
        2
    ) as new_customer_percentage,
    ROUND(
        (revenue::float / NULLIF(unique_customers, 0))::numeric,
        2
    ) as revenue_per_customer
FROM order_metrics
ORDER BY order_day DESC;
```

### 2. Customer Segmentation

```sql
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.email,
        COUNT(o.order_id) as order_count,
        SUM(o.total_amount) as total_spent,
        MAX(o.order_date) as last_order_date,
        MIN(o.order_date) as first_order_date,
        COUNT(DISTINCT DATE_TRUNC('month', o.order_date)) as active_months,
        AVG(o.total_amount) as avg_order_value
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.email
)
SELECT 
    email,
    order_count,
    ROUND(total_spent::numeric, 2) as total_spent,
    last_order_date,
    first_order_date,
    active_months,
    ROUND(avg_order_value::numeric, 2) as avg_order_value,
    CASE 
        WHEN order_count = 0 THEN 'Never Ordered'
        WHEN last_order_date >= CURRENT_DATE - INTERVAL '30 days' THEN 'Active'
        WHEN last_order_date >= CURRENT_DATE - INTERVAL '90 days' THEN 'At Risk'
        ELSE 'Churned'
    END as customer_status,
    CASE 
        WHEN total_spent >= 1000 AND order_count >= 10 THEN 'VIP'
        WHEN total_spent >= 500 OR order_count >= 5 THEN 'Regular'
        WHEN order_count > 0 THEN 'New'
        ELSE 'Inactive'
    END as customer_segment
FROM customer_metrics
ORDER BY total_spent DESC NULLS LAST;
```

### 3. Product Performance

```sql
WITH product_metrics AS (
    SELECT 
        p.product_id,
        p.name,
        p.category,
        p.price,
        COUNT(DISTINCT o.order_id) as order_count,
        SUM(oi.quantity) as units_sold,
        SUM(oi.quantity * oi.price_at_time) as revenue,
        COUNT(DISTINCT o.customer_id) as customer_count,
        AVG(r.rating) as avg_rating,
        COUNT(r.review_id) as review_count
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
    LEFT JOIN reviews r ON p.product_id = r.product_id
    GROUP BY p.product_id, p.name, p.category, p.price
)
SELECT 
    name,
    category,
    price,
    order_count,
    units_sold,
    ROUND(revenue::numeric, 2) as revenue,
    customer_count,
    ROUND(avg_rating::numeric, 2) as avg_rating,
    review_count,
    ROUND(
        (revenue / NULLIF(units_sold, 0))::numeric,
        2
    ) as avg_selling_price,
    ROUND(
        (units_sold::float / NULLIF(customer_count, 0))::numeric,
        2
    ) as units_per_customer
FROM product_metrics
ORDER BY revenue DESC NULLS LAST;
```

## Performance Optimization Examples

### 1. Index Usage

```sql
-- Create strategic indexes
CREATE INDEX idx_orders_customer_date 
ON orders(customer_id, order_date DESC);

CREATE INDEX idx_products_category_price 
ON products(category_id, price)
INCLUDE (name, stock_quantity);

-- Use indexes effectively
EXPLAIN ANALYZE
SELECT 
    c.name,
    COUNT(*) as order_count,
    SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE 
    o.order_date >= CURRENT_DATE - INTERVAL '30 days'
    AND o.total_amount > 100
GROUP BY c.customer_id, c.name;
```

### 2. Query Optimization

```sql
-- Bad: Inefficient subquery
SELECT *
FROM orders
WHERE customer_id IN (
    SELECT customer_id
    FROM customers
    WHERE status = 'active'
);

-- Good: Use JOIN
SELECT o.*
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE c.status = 'active';

-- Better: Use EXISTS
SELECT o.*
FROM orders o
WHERE EXISTS (
    SELECT 1
    FROM customers c
    WHERE c.customer_id = o.customer_id
    AND c.status = 'active'
);
```

### 3. Batch Processing

```sql
-- Process large datasets in batches
DO $$
DECLARE
    batch_size INT := 1000;
    total_processed INT := 0;
    batch_count INT := 0;
BEGIN
    LOOP
        WITH batch AS (
            SELECT order_id
            FROM orders
            WHERE processed = false
            ORDER BY order_date
            LIMIT batch_size
            FOR UPDATE SKIP LOCKED
        )
        UPDATE orders o
        SET processed = true
        FROM batch b
        WHERE o.order_id = b.order_id;
        
        GET DIAGNOSTICS batch_count = ROW_COUNT;
        
        EXIT WHEN batch_count = 0;
        
        total_processed := total_processed + batch_count;
        RAISE NOTICE 'Processed % orders', total_processed;
        
        COMMIT;
    END LOOP;
END $$;
```

## Common Pitfalls and Solutions

### 1. N+1 Query Problem

```sql
-- Bad: Separate query for each order
SELECT o.order_id, 
       (SELECT c.name FROM customers c WHERE c.id = o.customer_id) as customer_name
FROM orders o;

-- Good: Single JOIN query
SELECT o.order_id, c.name as customer_name
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;
```

### 2. Cartesian Products

```sql
-- Bad: Implicit cross join
SELECT * FROM orders, customers 
WHERE orders.customer_id = customers.customer_id;

-- Good: Explicit JOIN syntax
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;
```

### 3. NULL Handling

```sql
-- Bad: NULL comparison
SELECT * FROM products WHERE price = NULL;

-- Good: IS NULL operator
SELECT * FROM products WHERE price IS NULL;

-- Better: COALESCE for default values
SELECT 
    product_id,
    name,
    COALESCE(price, 0) as price,
    COALESCE(description, 'No description available') as description
FROM products;
```

## Best Practices Checklist

1. **Query Structure**
   - Use meaningful table aliases
   - Format queries for readability
   - Comment complex logic
   - Use CTEs for better organization

2. **Performance**
   - Create appropriate indexes
   - Filter early in the query
   - Avoid SELECT *
   - Use EXPLAIN ANALYZE

3. **Data Quality**
   - Handle NULL values appropriately
   - Validate input data
   - Use constraints
   - Implement error handling

4. **Maintenance**
   - Document queries
   - Use version control
   - Monitor performance
   - Regular optimization

Remember: "Clean, efficient queries lead to better performance and maintainability!"
