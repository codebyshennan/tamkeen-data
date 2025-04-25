# Advanced SQL Concepts: Beyond the Basics

## Introduction to Advanced SQL

SQL mastery goes beyond basic CRUD operations. Advanced SQL concepts enable you to:

- Write complex, performant queries
- Handle large-scale data processing
- Implement sophisticated business logic
- Optimize database operations

## Advanced SQL Functions

### 1. JSON Operations

```sql
-- JSON creation and manipulation
SELECT 
    order_id,
    jsonb_build_object(
        'customer', customer_name,
        'items', (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'product', product_name,
                    'quantity', quantity,
                    'price', price
                )
            )
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            WHERE oi.order_id = o.order_id
        )
    ) as order_details
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;

-- JSON querying
SELECT 
    order_id,
    order_details -> 'customer' as customer,
    jsonb_array_length(order_details -> 'items') as item_count,
    jsonb_path_query_array(
        order_details,
        '$.items[*].price'
    ) as prices
FROM order_details_json;
```

### 2. Full-Text Search

```sql
-- Create search vectors
CREATE INDEX idx_products_search ON products USING gin(
    to_tsvector('english', 
        coalesce(name,'') || ' ' || 
        coalesce(description,'') || ' ' || 
        coalesce(category,'')
    )
);

-- Perform search with ranking
SELECT 
    name,
    description,
    ts_rank(
        to_tsvector('english', 
            coalesce(name,'') || ' ' || 
            coalesce(description,'') || ' ' || 
            coalesce(category,'')
        ),
        plainto_tsquery('english', 'search term')
    ) as relevance
FROM products
WHERE to_tsvector('english', 
    coalesce(name,'') || ' ' || 
    coalesce(description,'') || ' ' || 
    coalesce(category,'')
) @@ plainto_tsquery('english', 'search term')
ORDER BY relevance DESC;
```

### 3. Recursive Queries

```sql
-- Employee hierarchy
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level employees
    SELECT 
        employee_id,
        name,
        manager_id,
        1 as level,
        ARRAY[name] as path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees with managers
    SELECT 
        e.employee_id,
        e.name,
        e.manager_id,
        eh.level + 1,
        eh.path || e.name
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT 
    level,
    lpad(' ', (level-1)*2) || name as employee,
    array_to_string(path, ' -> ') as hierarchy_path
FROM employee_hierarchy
ORDER BY path;
```

## Window Functions Deep Dive

### 1. Advanced Framing

```sql
SELECT 
    date,
    amount,
    -- Different frame specifications
    SUM(amount) OVER (
        ORDER BY date
        ROWS BETWEEN 
            UNBOUNDED PRECEDING 
            AND CURRENT ROW
    ) as cumulative_sum,
    
    AVG(amount) OVER (
        ORDER BY date
        ROWS BETWEEN 
            3 PRECEDING 
            AND 1 FOLLOWING
    ) as centered_average,
    
    SUM(amount) OVER (
        ORDER BY date
        RANGE BETWEEN 
            INTERVAL '1 month' PRECEDING 
            AND CURRENT ROW
    ) as rolling_monthly_sum
FROM transactions;
```

### 2. Multiple Window Functions

```sql
SELECT 
    category,
    product_name,
    price,
    -- Rankings within category
    RANK() OVER w1 as price_rank,
    DENSE_RANK() OVER w1 as dense_rank,
    ROW_NUMBER() OVER w1 as row_num,
    
    -- Statistics within category
    AVG(price) OVER w2 as avg_price,
    price - AVG(price) OVER w2 as price_diff,
    
    -- Percentiles within category
    NTILE(4) OVER w1 as price_quartile,
    PERCENT_RANK() OVER w1 as price_percentile
FROM products
WINDOW 
    w1 as (PARTITION BY category ORDER BY price DESC),
    w2 as (PARTITION BY category);
```

## Advanced Joins and Set Operations

### 1. Lateral Joins

```sql
SELECT 
    c.customer_name,
    recent_orders.order_id,
    recent_orders.order_date,
    recent_orders.amount
FROM customers c
CROSS JOIN LATERAL (
    SELECT 
        order_id,
        order_date,
        total_amount as amount
    FROM orders o
    WHERE o.customer_id = c.customer_id
    ORDER BY order_date DESC
    LIMIT 3
) recent_orders;
```

### 2. Set Operations with Ordering

```sql
-- Complex set operations
(
    SELECT 
        'Current' as period,
        category,
        SUM(amount) as total_sales
    FROM sales
    WHERE date >= CURRENT_DATE - INTERVAL '1 month'
    GROUP BY category
)
UNION ALL
(
    SELECT 
        'Previous' as period,
        category,
        SUM(amount) as total_sales
    FROM sales
    WHERE 
        date >= CURRENT_DATE - INTERVAL '2 months' AND
        date < CURRENT_DATE - INTERVAL '1 month'
    GROUP BY category
)
ORDER BY 
    category,
    period DESC;
```

## Error Handling and Transactions

### 1. Transaction Management

```sql
-- Complex transaction with savepoints
BEGIN;

SAVEPOINT order_start;

-- Create order
INSERT INTO orders (customer_id, order_date, status)
VALUES (123, CURRENT_TIMESTAMP, 'pending')
RETURNING order_id INTO v_order_id;

-- Check inventory and update stock
UPDATE products
SET stock_quantity = stock_quantity - order_quantity
WHERE product_id = v_product_id
AND stock_quantity >= order_quantity;

IF NOT FOUND THEN
    ROLLBACK TO order_start;
    RAISE EXCEPTION 'Insufficient stock for product %', v_product_id;
END IF;

-- Process payment
SAVEPOINT payment;

BEGIN
    -- Payment processing logic
    IF payment_failed THEN
        ROLLBACK TO payment;
        RAISE EXCEPTION 'Payment failed';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK TO payment;
        RAISE;
END;

COMMIT;
```

### 2. Error Handling

```sql
CREATE OR REPLACE FUNCTION process_order(
    p_customer_id INT,
    p_items JSONB
) RETURNS INT AS $$
DECLARE
    v_order_id INT;
    v_item JSONB;
    v_total DECIMAL(10,2) := 0;
BEGIN
    -- Input validation
    IF p_items IS NULL OR jsonb_array_length(p_items) = 0 THEN
        RAISE EXCEPTION 'Order must contain at least one item';
    END IF;
    
    -- Start transaction
    BEGIN
        -- Create order
        INSERT INTO orders (customer_id, order_date, status)
        VALUES (p_customer_id, CURRENT_TIMESTAMP, 'pending')
        RETURNING order_id INTO v_order_id;
        
        -- Process items
        FOR v_item IN SELECT * FROM jsonb_array_elements(p_items)
        LOOP
            -- Add order item
            BEGIN
                INSERT INTO order_items (
                    order_id, 
                    product_id,
                    quantity,
                    price
                )
                VALUES (
                    v_order_id,
                    (v_item->>'product_id')::INT,
                    (v_item->>'quantity')::INT,
                    (v_item->>'price')::DECIMAL
                );
            EXCEPTION
                WHEN foreign_key_violation THEN
                    RAISE EXCEPTION 'Invalid product ID: %',
                        (v_item->>'product_id');
                WHEN numeric_value_out_of_range THEN
                    RAISE EXCEPTION 'Invalid quantity or price for product %',
                        (v_item->>'product_id');
            END;
            
            -- Update total
            v_total := v_total + 
                ((v_item->>'quantity')::INT * (v_item->>'price')::DECIMAL);
        END LOOP;
        
        -- Update order total
        UPDATE orders 
        SET total_amount = v_total,
            status = 'confirmed'
        WHERE order_id = v_order_id;
        
        RETURN v_order_id;
    EXCEPTION
        WHEN OTHERS THEN
            RAISE EXCEPTION 'Order processing failed: %', SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql;
```

## Additional Real-World Scenarios

### 1. E-commerce Funnel Analysis

```sql
WITH user_journey AS (
    SELECT 
        u.user_id,
        u.email,
        COUNT(DISTINCT CASE WHEN e.event_type = 'view' THEN e.product_id END) as products_viewed,
        COUNT(DISTINCT CASE WHEN e.event_type = 'add_to_cart' THEN e.product_id END) as products_carted,
        COUNT(DISTINCT CASE WHEN e.event_type = 'purchase' THEN e.product_id END) as products_purchased,
        COUNT(DISTINCT CASE WHEN e.event_type = 'purchase' THEN e.session_id END) as purchase_sessions,
        COUNT(DISTINCT e.session_id) as total_sessions
    FROM users u
    LEFT JOIN events e ON u.user_id = e.user_id
    GROUP BY u.user_id, u.email
)
SELECT 
    ROUND(AVG(products_viewed)::numeric, 2) as avg_products_viewed,
    ROUND(AVG(products_carted)::numeric, 2) as avg_products_carted,
    ROUND(AVG(products_purchased)::numeric, 2) as avg_products_purchased,
    ROUND(
        100.0 * SUM(CASE WHEN products_carted > 0 THEN 1 END) / 
        NULLIF(SUM(CASE WHEN products_viewed > 0 THEN 1 END), 0),
        2
    ) as view_to_cart_rate,
    ROUND(
        100.0 * SUM(CASE WHEN products_purchased > 0 THEN 1 END) / 
        NULLIF(SUM(CASE WHEN products_carted > 0 THEN 1 END), 0),
        2
    ) as cart_to_purchase_rate
FROM user_journey;
```

### 2. Fraud Detection System

```sql
WITH transaction_metrics AS (
    SELECT 
        t.transaction_id,
        t.user_id,
        t.amount,
        t.created_at,
        t.status,
        -- Time since last transaction
        EXTRACT(EPOCH FROM (
            t.created_at - LAG(t.created_at) OVER (
                PARTITION BY t.user_id 
                ORDER BY t.created_at
            )
        ))/60 as minutes_since_last_txn,
        -- Amount compared to user's average
        amount / NULLIF(AVG(amount) OVER (
            PARTITION BY t.user_id
        ), 0) as amount_vs_avg,
        -- Number of transactions in last hour
        COUNT(*) OVER (
            PARTITION BY t.user_id 
            ORDER BY t.created_at 
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING 
            AND CURRENT ROW
        ) as txns_last_hour,
        -- Different locations in last 24 hours
        COUNT(DISTINCT location_id) OVER (
            PARTITION BY t.user_id 
            ORDER BY t.created_at 
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING 
            AND CURRENT ROW
        ) as locations_24h
    FROM transactions t
)
SELECT 
    transaction_id,
    user_id,
    amount,
    created_at,
    CASE 
        WHEN minutes_since_last_txn < 1 
        AND amount_vs_avg > 3 THEN 'High Risk: Rapid Large Transaction'
        WHEN txns_last_hour > 10 THEN 'High Risk: High Frequency'
        WHEN locations_24h > 3 THEN 'High Risk: Multiple Locations'
        WHEN amount_vs_avg > 5 THEN 'Medium Risk: Unusual Amount'
        WHEN minutes_since_last_txn < 5 THEN 'Medium Risk: Rapid Transactions'
        ELSE 'Low Risk'
    END as risk_assessment
FROM transaction_metrics
WHERE 
    minutes_since_last_txn < 5 
    OR amount_vs_avg > 3 
    OR txns_last_hour > 10 
    OR locations_24h > 3;
```

### 3. Inventory Optimization

```sql
WITH inventory_metrics AS (
    SELECT 
        p.product_id,
        p.name,
        p.category,
        p.stock_quantity,
        p.reorder_point,
        p.lead_time_days,
        -- Sales velocity
        SUM(oi.quantity) FILTER (
            WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
        ) as units_sold_30d,
        -- Stockout incidents
        COUNT(*) FILTER (
            WHERE p.stock_quantity = 0
        ) as stockout_count,
        -- Average daily sales
        COALESCE(
            SUM(oi.quantity) FILTER (
                WHERE o.order_date >= CURRENT_DATE - INTERVAL '90 days'
            )::float / 90,
            0
        ) as avg_daily_sales,
        -- Safety stock calculation
        SQRT(
            POWER(p.lead_time_days * STDDEV(oi.quantity), 2) +
            POWER(AVG(oi.quantity) * STDDEV(p.lead_time_days), 2)
        ) as safety_stock
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
    GROUP BY 
        p.product_id, p.name, p.category, 
        p.stock_quantity, p.reorder_point, p.lead_time_days
)
SELECT 
    name,
    category,
    stock_quantity,
    units_sold_30d,
    ROUND(avg_daily_sales::numeric, 2) as avg_daily_sales,
    ROUND(safety_stock::numeric, 2) as recommended_safety_stock,
    CASE 
        WHEN stock_quantity = 0 THEN 'Out of Stock'
        WHEN stock_quantity < safety_stock THEN 'Below Safety Stock'
        WHEN stock_quantity < reorder_point THEN 'Reorder Needed'
        ELSE 'Adequate Stock'
    END as stock_status,
    CEIL(
        CASE 
            WHEN avg_daily_sales > 0 
            THEN stock_quantity / avg_daily_sales
            ELSE NULL
        END
    ) as days_of_inventory,
    ROUND(
        GREATEST(
            reorder_point - stock_quantity,
            (avg_daily_sales * lead_time_days) - stock_quantity,
            0
        )::numeric,
        0
    ) as suggested_order_quantity
FROM inventory_metrics
ORDER BY 
    CASE 
        WHEN stock_quantity = 0 THEN 1
        WHEN stock_quantity < safety_stock THEN 2
        WHEN stock_quantity < reorder_point THEN 3
        ELSE 4
    END,
    avg_daily_sales DESC;
```

## Performance Optimization Tips

### 1. Query Plan Analysis

```sql
-- Analyze and explain complex queries
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    c.customer_name,
    COUNT(*) as order_count,
    SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE 
    o.order_date >= CURRENT_DATE - INTERVAL '1 year'
    AND o.total_amount > 100
GROUP BY c.customer_id, c.customer_name
HAVING COUNT(*) > 5
ORDER BY total_spent DESC;

-- Key metrics to monitor:
-- 1. Planning Time
-- 2. Execution Time
-- 3. Actual vs. Planned Rows
-- 4. Buffer Usage (shared_blks_hit vs. shared_blks_read)
```

### 2. Index Design Patterns

```sql
-- Composite indexes for range + equality
CREATE INDEX idx_orders_customer_date 
ON orders(customer_id, order_date DESC);

-- Partial indexes for specific queries
CREATE INDEX idx_high_value_orders 
ON orders(order_date)
WHERE total_amount > 1000;

-- Expression indexes for function calls
CREATE INDEX idx_order_date_truncated 
ON orders(DATE_TRUNC('month', order_date));

-- Include columns to avoid table lookups
CREATE INDEX idx_orders_customer_details 
ON orders(customer_id)
INCLUDE (order_date, total_amount, status);
```

### 3. Materialized Views with Refresh Strategies

```sql
-- Create materialized view
CREATE MATERIALIZED VIEW sales_summary AS
SELECT 
    DATE_TRUNC('day', order_date) as sale_date,
    category,
    SUM(total_amount) as revenue,
    COUNT(*) as order_count
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
GROUP BY 
    DATE_TRUNC('day', order_date),
    category;

-- Create indexes on materialized view
CREATE INDEX idx_sales_summary_date 
ON sales_summary(sale_date DESC);

CREATE INDEX idx_sales_summary_category 
ON sales_summary(category, sale_date DESC);

-- Refresh strategy
CREATE OR REPLACE FUNCTION refresh_sales_summary()
RETURNS trigger AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY sales_summary;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic refresh
CREATE TRIGGER refresh_sales_summary_trigger
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_sales_summary();
```

## Common Pitfalls and Solutions

### 1. N+1 Query Problem

```sql
-- Bad: Separate query for each order
SELECT 
    o.order_id,
    (
        SELECT c.name 
        FROM customers c 
        WHERE c.id = o.customer_id
    ) as customer_name,
    (
        SELECT COUNT(*) 
        FROM order_items oi 
        WHERE oi.order_id = o.order_id
    ) as item_count
FROM orders o;

-- Good: Use JOINs and window functions
SELECT 
    o.order_id,
    c.name as customer_name,
    COUNT(*) OVER (PARTITION BY o.order_id) as item_count
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN order_items oi ON o.order_id = oi.order_id;
```

### 2. Inefficient Date Handling

```sql
-- Bad: Function on column prevents index use
SELECT * 
FROM orders 
WHERE EXTRACT(YEAR FROM order_date) = 2023;

-- Good: Range condition uses index
SELECT * 
FROM orders 
WHERE order_date >= '2023-01-01' 
AND order_date < '2024-01-01';
```

### 3. Subquery Performance

```sql
-- Bad: Correlated subquery runs for each row
SELECT 
    product_name,
    (
        SELECT AVG(quantity)
        FROM order_items oi
        WHERE oi.product_id = p.product_id
    ) as avg_quantity
FROM products p;

-- Good: Use window functions or JOIN
SELECT 
    p.product_name,
    AVG(oi.quantity) OVER (
        PARTITION BY p.product_id
    ) as avg_quantity
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id;
```

Remember: "Performance optimization is an iterative process - measure, analyze, improve!"

## Best Practices and Guidelines

### 1. Query Writing

- Write self-documenting queries with clear aliases
- Use CTEs for better readability and maintenance
- Leverage appropriate indexes for performance
- Consider query plan and execution cost

### 2. Database Design

- Implement proper constraints and relationships
- Use appropriate data types
- Design with scalability in mind
- Regular maintenance and optimization

### 3. Development Process

- Version control your database changes
- Implement proper testing procedures
- Monitor query performance
- Regular code reviews and optimization

## Additional Resources

1. **Documentation**
   - [PostgreSQL Documentation](https://www.postgresql.org/docs/)
   - [SQL Performance Tuning](https://use-the-index-luke.com/)
   - [Modern SQL Guide](https://modern-sql.com/)

2. **Tools**
   - [pgAdmin](https://www.pgadmin.org/) for database management
   - [DBeaver](https://dbeaver.io/) for query development
   - [pg_stat_statements](https://www.postgresql.org/docs/current/pgstatstatements.html) for query analysis

Remember: "Complex queries should be like well-written essays - clear, structured, and purposeful!"
