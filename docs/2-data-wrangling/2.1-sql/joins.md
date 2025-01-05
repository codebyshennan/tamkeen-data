# Mastering SQL Joins: Connecting Your Data Universe 🌐

[Previous content remains the same until the end]

## Additional Real-World Scenarios 💼

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

### 2. Supply Chain Analysis
```sql
WITH supplier_performance AS (
    SELECT 
        s.supplier_id,
        s.supplier_name,
        COUNT(DISTINCT o.order_id) as orders_fulfilled,
        AVG(EXTRACT(EPOCH FROM (d.delivery_date - o.order_date))/86400) as avg_delivery_days,
        COUNT(DISTINCT CASE 
            WHEN d.delivery_date > o.expected_delivery 
            THEN o.order_id 
        END) as late_deliveries,
        SUM(o.total_amount) as total_purchase_value
    FROM suppliers s
    LEFT JOIN purchase_orders o ON s.supplier_id = o.supplier_id
    LEFT JOIN deliveries d ON o.order_id = d.order_id
    GROUP BY s.supplier_id, s.supplier_name
)
SELECT 
    supplier_name,
    orders_fulfilled,
    ROUND(avg_delivery_days::numeric, 1) as avg_delivery_days,
    ROUND(
        100.0 * late_deliveries / NULLIF(orders_fulfilled, 0),
        2
    ) as late_delivery_rate,
    ROUND(total_purchase_value::numeric, 2) as total_purchase_value,
    CASE 
        WHEN late_deliveries = 0 THEN 'Excellent'
        WHEN late_deliveries::float / orders_fulfilled <= 0.05 THEN 'Good'
        WHEN late_deliveries::float / orders_fulfilled <= 0.10 THEN 'Fair'
        ELSE 'Poor'
    END as performance_rating
FROM supplier_performance
ORDER BY orders_fulfilled DESC;
```

### 3. Customer Service Integration
```sql
WITH ticket_metrics AS (
    SELECT 
        t.ticket_id,
        t.customer_id,
        t.created_at,
        t.resolved_at,
        t.status,
        t.priority,
        o.order_id,
        o.order_date,
        p.product_id,
        p.product_name,
        EXTRACT(EPOCH FROM (t.resolved_at - t.created_at))/3600 as resolution_time_hours
    FROM support_tickets t
    LEFT JOIN orders o ON t.order_id = o.order_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
)
SELECT 
    priority,
    COUNT(*) as ticket_count,
    ROUND(AVG(resolution_time_hours)::numeric, 2) as avg_resolution_hours,
    ROUND(
        100.0 * COUNT(CASE WHEN status = 'resolved' THEN 1 END) / COUNT(*),
        2
    ) as resolution_rate,
    STRING_AGG(DISTINCT product_name, ', ' ORDER BY product_name) 
        FILTER (WHERE product_name IS NOT NULL) as affected_products
FROM ticket_metrics
GROUP BY priority
ORDER BY 
    CASE priority
        WHEN 'high' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'low' THEN 3
    END;
```

## Performance Optimization Examples 🚀

### 1. Hash Join vs. Merge Join
```sql
-- Force hash join for large tables with no useful indexes
SELECT /*+ HASHJOIN(o c) */
    c.customer_name,
    COUNT(*) as order_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.customer_name;

-- Force merge join for indexed columns
SELECT /*+ MERGEJOIN(o c) */
    c.customer_name,
    COUNT(*) as order_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.customer_name;
```

### 2. Partitioned Joins
```sql
-- Join with partitioned tables
CREATE TABLE orders (
    order_id INT,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2)
) PARTITION BY RANGE (order_date);

CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    price DECIMAL(10,2)
) PARTITION BY RANGE (order_id);

-- Create corresponding partitions
CREATE TABLE orders_2023_q1 PARTITION OF orders
    FOR VALUES FROM ('2023-01-01') TO ('2023-04-01');
CREATE TABLE order_items_2023_q1 PARTITION OF order_items
    FOR VALUES FROM (1000) TO (2000);

-- Query specific partitions
SELECT 
    o.order_id,
    SUM(oi.quantity * oi.price) as total_value
FROM orders_2023_q1 o
JOIN order_items_2023_q1 oi ON o.order_id = oi.order_id
GROUP BY o.order_id;
```

### 3. Materialized Views for Complex Joins
```sql
-- Create materialized view for frequently joined data
CREATE MATERIALIZED VIEW order_summary AS
SELECT 
    o.order_id,
    c.customer_name,
    o.order_date,
    COUNT(oi.product_id) as total_items,
    SUM(oi.quantity * oi.price) as total_value,
    STRING_AGG(p.product_name, ', ') as products
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
GROUP BY o.order_id, c.customer_name, o.order_date;

-- Create indexes on materialized view
CREATE INDEX idx_order_summary_date ON order_summary(order_date);
CREATE INDEX idx_order_summary_customer ON order_summary(customer_name);

-- Refresh strategy
CREATE OR REPLACE FUNCTION refresh_order_summary()
RETURNS trigger AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY order_summary;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_order_summary_trigger
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_order_summary();
```

## Interactive Examples with Sample Data 💡

### 1. Generate Sample Data
```sql
-- Create sample customers
INSERT INTO customers (customer_name, email, join_date)
SELECT 
    'Customer ' || i,
    'customer' || i || '@example.com',
    CURRENT_DATE - (random() * 365)::integer
FROM generate_series(1, 1000) i;

-- Create sample orders
INSERT INTO orders (customer_id, order_date, total_amount)
SELECT 
    (random() * 1000)::integer,
    CURRENT_DATE - (random() * 90)::integer,
    (random() * 1000)::numeric(10,2)
FROM generate_series(1, 5000);

-- Create sample products
INSERT INTO products (product_name, category_id, price)
SELECT 
    'Product ' || i,
    (random() * 10 + 1)::integer,
    (random() * 100 + 10)::numeric(10,2)
FROM generate_series(1, 100);
```

### 2. Analysis Queries
```sql
-- Customer purchase patterns
WITH customer_patterns AS (
    SELECT 
        c.customer_id,
        c.customer_name,
        COUNT(DISTINCT o.order_id) as order_count,
        COUNT(DISTINCT DATE_TRUNC('month', o.order_date)) as active_months,
        SUM(o.total_amount) as total_spent,
        AVG(o.total_amount) as avg_order_value
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.customer_name
)
SELECT 
    CASE 
        WHEN order_count = 0 THEN 'Never Ordered'
        WHEN order_count = 1 THEN 'One-Time'
        WHEN order_count > 1 AND active_months = 1 THEN 'Same Month Multiple'
        WHEN order_count > 1 THEN 'Returning'
    END as customer_type,
    COUNT(*) as customer_count,
    ROUND(AVG(order_count)::numeric, 2) as avg_orders,
    ROUND(AVG(total_spent)::numeric, 2) as avg_total_spent,
    ROUND(AVG(avg_order_value)::numeric, 2) as avg_order_value
FROM customer_patterns
GROUP BY 
    CASE 
        WHEN order_count = 0 THEN 'Never Ordered'
        WHEN order_count = 1 THEN 'One-Time'
        WHEN order_count > 1 AND active_months = 1 THEN 'Same Month Multiple'
        WHEN order_count > 1 THEN 'Returning'
    END
ORDER BY avg_total_spent DESC;
```

Remember: "Efficient joins are the key to unlocking insights from your data!" 💪
