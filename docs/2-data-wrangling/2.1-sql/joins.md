# Mastering SQL Joins: Connecting Your Data Universe

**After this lesson:** You can choose the right join type (**INNER**, **LEFT**, **RIGHT**, **FULL**) for a question, write multi-table queries with clear aliases, and avoid accidental Cartesian products.

## Helpful video

Quick tour of join types in SQL (inner, left, right, full).

<iframe width="560" height="315" src="https://www.youtube.com/embed/9yeOJ0ZMUYw" title="SQL Joins Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Overview

**Prerequisites:** [Basic SQL Operations](basic-operations.md) and [Aggregations](aggregations.md). You should recognize foreign keys from [Introduction to Databases](intro-databases.md).

> **Time needed:** About 60-90 minutes with practice queries.

## Why this matters

Almost every real question spans more than one table, customers and orders, students and enrollments, parts and suppliers. Choosing **INNER** vs **LEFT** join is choosing *which rows you are willing to drop* from the result; getting that wrong silently loses data or duplicates it.

## Introduction to SQL Joins

{% include mermaid-diagram.html src="2-data-wrangling/2.1-sql/diagrams/joins-1.mmd" %}

> **Figure (add screenshot or diagram):** Four Venn diagrams side by side, INNER (centre only), LEFT (left circle + centre), RIGHT (right circle + centre), FULL OUTER (both circles). Shade the returned region for each.

SQL joins combine rows from two or more tables based on related columns. They are essential for:

- Retrieving related data across tables
- Building comprehensive reports
- Analyzing relationships in data
- Creating meaningful insights

## Types of SQL Joins

### 1. INNER JOIN

Returns only matching rows from both tables.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Basic INNER JOIN
SELECT
    o.order_id,
    c.customer_name,
    o.order_date,
    o.total_amount
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id;

-- Multiple conditions
SELECT
    o.order_id,
    c.customer_name
FROM orders o
INNER JOIN customers c
    ON o.customer_id = c.customer_id
    AND o.store_id = c.preferred_store_id;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Basic INNER JOIN</span>
    </div>
    <div class="code-callout__body">
      <p>Joins <code>orders</code> to <code>customers</code> on <code>customer_id</code>. Only rows with a matching customer in both tables appear, orders with no customer record and customers with no orders are both excluded.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Multiple ON conditions</span>
    </div>
    <div class="code-callout__body">
      <p>Two predicates in the <code>ON</code> clause mean both must match: the order's customer and the store must be the customer's preferred store. This further restricts the result, only orders placed at the customer's preferred location are returned.</p>
    </div>
  </div>
</aside>
</div>

### 2. LEFT JOIN (LEFT OUTER JOIN)

Returns all rows from the left table and matching rows from the right table.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Basic LEFT JOIN
SELECT
    c.customer_name,
    COUNT(o.order_id) as order_count,
    COALESCE(SUM(o.total_amount), 0) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_name;

-- Finding missing relationships
SELECT
    c.customer_name,
    'No orders' as status
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">LEFT JOIN: all customers, even those with no orders</span>
    </div>
    <div class="code-callout__body">
      <p>Every customer from the left table appears in the result. Where there are no matching orders, <code>COUNT(o.order_id)</code> returns 0 and <code>COALESCE(SUM(…), 0)</code> substitutes 0 for the NULL total, customers who never ordered still appear with zeroes.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-16" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Finding customers with no orders</span>
    </div>
    <div class="code-callout__body">
      <p>After the LEFT JOIN, rows where <code>o.order_id IS NULL</code> are exactly the customers who have no matching order. This anti-join pattern is a reliable way to find "missing" relationships without subqueries.</p>
    </div>
  </div>
</aside>
</div>

### 3. RIGHT JOIN (RIGHT OUTER JOIN)

Returns all rows from the right table and matching rows from the left table.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Basic RIGHT JOIN
SELECT
    p.product_name,
    COALESCE(SUM(oi.quantity), 0) as total_ordered
FROM order_items oi
RIGHT JOIN products p ON oi.product_id = p.product_id
GROUP BY p.product_name;

-- Finding unused products
SELECT
    p.product_name,
    'Never ordered' as status
FROM order_items oi
RIGHT JOIN products p ON oi.product_id = p.product_id
WHERE oi.order_id IS NULL;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">RIGHT JOIN: all products, even those never ordered</span>
    </div>
    <div class="code-callout__body">
      <p>The right table (<code>products</code>) drives the result, every product appears regardless of whether it has any matching <code>order_items</code> rows. <code>COALESCE(SUM(oi.quantity), 0)</code> returns 0 for products with no orders instead of NULL.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Finding products that have never been ordered</span>
    </div>
    <div class="code-callout__body">
      <p>After the RIGHT JOIN, <code>WHERE oi.order_id IS NULL</code> isolates products with no matching order item, another anti-join pattern, this time preserving the right table's unmatched rows.</p>
    </div>
  </div>
</aside>
</div>

### 4. FULL JOIN (FULL OUTER JOIN)

Returns all rows when there's a match in either left or right table.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Basic FULL JOIN
SELECT
    c.customer_name,
    p.product_name,
    COUNT(o.order_id) as times_ordered
FROM customers c
FULL JOIN orders o ON c.customer_id = o.customer_id
FULL JOIN order_items oi ON o.order_id = oi.order_id
FULL JOIN products p ON oi.product_id = p.product_id
GROUP BY c.customer_name, p.product_name;

-- Finding all missing relationships
SELECT
    COALESCE(c.customer_name, 'No Customer') as customer,
    COALESCE(p.product_name, 'No Product') as product,
    'Missing Relationship' as status
FROM customers c
FULL JOIN orders o ON c.customer_id = o.customer_id
FULL JOIN order_items oi ON o.order_id = oi.order_id
FULL JOIN products p ON oi.product_id = p.product_id
WHERE o.order_id IS NULL;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">FULL JOIN: every row from every table</span>
    </div>
    <div class="code-callout__body">
      <p>Three chained FULL JOINs mean unmatched rows from any table still appear, a customer with no orders, an order with no items, and a product with no order items all show up with NULLs for the unjoined columns.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Finding all missing relationships</span>
    </div>
    <div class="code-callout__body">
      <p><code>COALESCE</code> substitutes readable labels for NULLs in the output. <code>WHERE o.order_id IS NULL</code> filters to only the orphaned rows, customers with no orders, or products never ordered, exposing data integrity gaps across the four tables.</p>
    </div>
  </div>
</aside>
</div>

### 5. CROSS JOIN

Returns Cartesian product of both tables.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Basic CROSS JOIN
SELECT
    p.product_name,
    c.category_name
FROM products p
CROSS JOIN categories c;

-- Generate date-product combinations
SELECT
    d.date,
    p.product_name
FROM generate_series(
    CURRENT_DATE,
    CURRENT_DATE + INTERVAL '7 days',
    INTERVAL '1 day'
) as d(date)
CROSS JOIN products p;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Basic CROSS JOIN: every product × category combination</span>
    </div>
    <div class="code-callout__body">
      <p>A CROSS JOIN has no <code>ON</code> condition, it produces every combination of rows from both tables. With 100 products and 10 categories this gives 1,000 rows. Useful for generating all possibilities (e.g., a pricing matrix), dangerous when accidental.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Generate date × product combinations</span>
    </div>
    <div class="code-callout__body">
      <p><code>generate_series</code> produces 8 consecutive dates; CROSS JOIN pairs each date with every product. This is a common pattern for pre-filling a calendar grid so that days with zero sales still appear as rows rather than gaps.</p>
    </div>
  </div>
</aside>
</div>

## Common Join Patterns

### 1. Multi-Table Joins

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Order details with customer and product info
SELECT
    o.order_id,
    c.customer_name,
    p.product_name,
    oi.quantity,
    oi.quantity * p.price as line_total
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Multi-table INNER JOIN chain</span>
    </div>
    <div class="code-callout__body">
      <p>Three consecutive JOINs thread through four tables: <code>orders → customers</code> for the buyer name, <code>orders → order_items</code> for the line rows, <code>order_items → products</code> for the price. Each JOIN adds columns; only rows present in every table are kept.</p>
    </div>
  </div>
</aside>
</div>

### 2. Self Joins

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Employee hierarchy
SELECT
    e.employee_name as employee,
    m.employee_name as manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;

-- Product recommendations
SELECT
    p1.product_name,
    p2.product_name as recommended_product,
    COUNT(*) as times_bought_together
FROM order_items oi1
JOIN order_items oi2
    ON oi1.order_id = oi2.order_id
    AND oi1.product_id < oi2.product_id
JOIN products p1 ON oi1.product_id = p1.product_id
JOIN products p2 ON oi2.product_id = p2.product_id
GROUP BY p1.product_name, p2.product_name
HAVING COUNT(*) > 5
ORDER BY times_bought_together DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Self join: employee → manager from the same table</span>
    </div>
    <div class="code-callout__body">
      <p>The same <code>employees</code> table is aliased twice. <code>e</code> is the employee row; <code>m</code> is the manager row found by matching <code>e.manager_id = m.employee_id</code>. LEFT JOIN keeps employees who have no manager (e.g., the CEO).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Self join on order_items: frequently bought together</span>
    </div>
    <div class="code-callout__body">
      <p><code>order_items</code> is joined to itself on the same <code>order_id</code> to find pairs of products appearing in the same order. <code>oi1.product_id &lt; oi2.product_id</code> prevents counting each pair twice and eliminates self-pairs. <code>HAVING COUNT(*) &gt; 5</code> keeps only popular combinations.</p>
    </div>
  </div>
</aside>
</div>

### 3. Conditional Joins

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Join based on date ranges
SELECT
    e.event_name,
    p.promotion_name
FROM events e
LEFT JOIN promotions p
    ON e.event_date BETWEEN p.start_date AND p.end_date;

-- Join with multiple conditions
SELECT
    o.order_id,
    d.driver_name
FROM orders o
LEFT JOIN drivers d
    ON d.zone_id = o.delivery_zone_id
    AND d.is_active = true
    AND d.current_orders < d.max_orders;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Range join: match events to overlapping promotions</span>
    </div>
    <div class="code-callout__body">
      <p>The <code>ON</code> clause uses <code>BETWEEN</code> instead of equality, an event matches a promotion if its date falls within the promotion's active window. LEFT JOIN keeps events even when no promotion was running.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Multi-condition join: only active drivers with capacity</span>
    </div>
    <div class="code-callout__body">
      <p>Three conditions in the <code>ON</code> clause act as a composite filter at join time: the driver must cover the order's delivery zone, be currently active, and have fewer orders than their maximum capacity. This keeps the <code>WHERE</code> clause clean and expresses driver eligibility as part of the join.</p>
    </div>
  </div>
</aside>
</div>

## Join Best Practices

### 1. Performance Optimization

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Use proper indexes
CREATE INDEX idx_orders_customer
ON orders(customer_id);

CREATE INDEX idx_order_items_composite
ON order_items(order_id, product_id);

-- Join order matters
SELECT /*+ LEADING(small_table medium_table large_table) */
    *
FROM small_table
JOIN medium_table ON small_table.id = medium_table.id
JOIN large_table ON medium_table.id = large_table.id;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Index join columns for fast lookups</span>
    </div>
    <div class="code-callout__body">
      <p>Creating indexes on the foreign key columns used in <code>ON</code> clauses (<code>customer_id</code>, and the composite <code>order_id, product_id</code>) lets the planner use an index nested-loop join instead of a full sequential scan of each table.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-13" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Hint: start from the smallest table</span>
    </div>
    <div class="code-callout__body">
      <p>The <code>/*+ LEADING(…) */</code> hint tells the planner which table to access first. Joining small → medium → large reduces intermediate row counts at each step and keeps memory usage low.</p>
    </div>
  </div>
</aside>
</div>

### 2. Common Mistakes to Avoid

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Avoid Cartesian products
-- Bad:
SELECT * FROM orders, customers;

-- Good:
SELECT * FROM orders
JOIN customers ON orders.customer_id = customers.customer_id;

-- Handle NULL values
SELECT
    c.customer_name,
    COALESCE(SUM(o.total_amount), 0) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_name;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Avoid implicit Cartesian products</span>
    </div>
    <div class="code-callout__body">
      <p>The comma-separated <code>FROM orders, customers</code> syntax produces a full Cartesian product, every order row paired with every customer row. Without a filter this is almost always a mistake. The explicit <code>JOIN … ON</code> form makes the intent clear and is harder to accidentally omit.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Handle NULLs from outer joins with COALESCE</span>
    </div>
    <div class="code-callout__body">
      <p>After a LEFT JOIN, unmatched right-table columns are NULL. Aggregating NULL values with <code>SUM</code> returns NULL, not 0. <code>COALESCE(SUM(o.total_amount), 0)</code> converts the NULL result to 0 so customers who have never ordered show a meaningful total.</p>
    </div>
  </div>
</aside>
</div>

### 3. Maintainability Tips

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Use meaningful aliases
SELECT
    cust.name,
    ord.order_date,
    prod.name as product_name
FROM customers cust
JOIN orders ord ON cust.customer_id = ord.customer_id
JOIN products prod ON ord.product_id = prod.product_id;

-- Break down complex joins
WITH customer_orders AS (
    SELECT
        customer_id,
        COUNT(*) as order_count
    FROM orders
    GROUP BY customer_id
),
customer_spending AS (
    SELECT
        customer_id,
        SUM(total_amount) as total_spent
    FROM orders
    GROUP BY customer_id
)
SELECT
    c.customer_name,
    co.order_count,
    cs.total_spent
FROM customers c
LEFT JOIN customer_orders co ON c.customer_id = co.customer_id
LEFT JOIN customer_spending cs ON c.customer_id = cs.customer_id;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Readable aliases: full words instead of single letters</span>
    </div>
    <div class="code-callout__body">
      <p>Using <code>cust</code>, <code>ord</code>, and <code>prod</code> instead of <code>c</code>, <code>o</code>, <code>p</code> makes queries self-documenting, readers can tell which table each column comes from without cross-referencing the FROM clause.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 1: per-customer order counts</span>
    </div>
    <div class="code-callout__body">
      <p>The first CTE pre-aggregates orders into one row per customer. Breaking a complex join into CTEs makes each piece independently readable and testable before composing them in the final SELECT.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-31" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 2 + outer join: combine aggregates per customer</span>
    </div>
    <div class="code-callout__body">
      <p>The second CTE sums total spend per customer. The final SELECT LEFT JOINs both CTEs onto <code>customers</code> so that customers with no orders still appear, with NULLs for count and total rather than being silently dropped.</p>
    </div>
  </div>
</aside>
</div>

## Additional Real-World Scenarios

### 1. E-commerce Funnel Analysis

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
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
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-13" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE: per-user event counts across the funnel</span>
    </div>
    <div class="code-callout__body">
      <p>LEFT JOIN <code>events</code> keeps users who never triggered any event. <code>COUNT(DISTINCT CASE WHEN event_type = 'view' THEN product_id END)</code> counts unique products at each funnel stage, view, cart, purchase, per user without multiple self-joins.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="15-28" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: aggregate funnel conversion rates</span>
    </div>
    <div class="code-callout__body">
      <p>Averages per-user counts across all users, then computes view-to-cart and cart-to-purchase rates as percentages. <code>NULLIF(…, 0)</code> in the denominator prevents division-by-zero when no users reached the prior stage.</p>
    </div>
  </div>
</aside>
</div>

### 2. Supply Chain Analysis

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
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
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-16" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE: join suppliers → orders → deliveries</span>
    </div>
    <div class="code-callout__body">
      <p>Two LEFT JOINs chain from supplier to purchase orders then to deliveries. LEFT JOIN keeps suppliers who have no orders or deliveries. <code>EXTRACT(EPOCH FROM …)/86400</code> converts the interval to fractional days for <code>AVG</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-33" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: compute late-delivery rate and rating</span>
    </div>
    <div class="code-callout__body">
      <p>Divides late deliveries by total orders-<code>NULLIF(orders_fulfilled, 0)</code> prevents division-by-zero for new suppliers. The <code>CASE</code> expression buckets each supplier into a performance tier (Excellent / Good / Fair / Poor) for easy reporting.</p>
    </div>
  </div>
</aside>
</div>

### 3. Customer Service Integration

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
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
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-18" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE: join support tickets → orders → products</span>
    </div>
    <div class="code-callout__body">
      <p>Three LEFT JOINs link each ticket to its related order, the order's items, and those items' products. LEFT JOIN preserves tickets that aren't linked to an order. <code>EXTRACT(EPOCH FROM …)/3600</code> converts the timestamp difference to hours for resolution time.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-36" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: aggregate metrics by priority</span>
    </div>
    <div class="code-callout__body">
      <p>Groups tickets by priority and calculates ticket count, average resolution time, resolution rate, and a comma-separated list of distinct affected products. <code>FILTER (WHERE product_name IS NOT NULL)</code> on <code>STRING_AGG</code> skips tickets not linked to any product.</p>
    </div>
  </div>
</aside>
</div>

## Performance Optimization Examples

### 1. Hash Join vs. Merge Join

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
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
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Hash join: best for large unsorted tables</span>
    </div>
    <div class="code-callout__body">
      <p>Hash joins build an in-memory hash table from the smaller table, then probe it for each row of the larger table. They work well when neither side has a useful index and are typically the planner's default for large ad-hoc joins.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Merge join: best for pre-sorted or indexed columns</span>
    </div>
    <div class="code-callout__body">
      <p>Merge joins require both sides sorted on the join key. When an index already provides that order the planner can avoid a sort step and stream through both sides in a single pass, very efficient for equality joins on indexed columns.</p>
    </div>
  </div>
</aside>
</div>

### 2. Partitioned Joins

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
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
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-14" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Declare partitioned parent tables</span>
    </div>
    <div class="code-callout__body">
      <p><code>PARTITION BY RANGE (order_date)</code> and <code>PARTITION BY RANGE (order_id)</code> create parent tables with no data of their own, they delegate rows to child partitions. Queries against the parent automatically target only the relevant partition(s).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-28" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Create child partitions and join them directly</span>
    </div>
    <div class="code-callout__body">
      <p>Each <code>PARTITION OF … FOR VALUES FROM … TO …</code> creates a child table holding rows in that range. Joining the Q1 child partitions directly instead of the parent tables lets the planner skip all other partitions entirely, partition pruning.</p>
    </div>
  </div>
</aside>
</div>

### 3. Materialized Views for Complex Joins

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
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
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-15" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Materialize a complex multi-join query as a stored result</span>
    </div>
    <div class="code-callout__body">
      <p><code>CREATE MATERIALIZED VIEW</code> executes the four-table JOIN once and stores the rows on disk. Subsequent reads hit the stored result instead of re-running the join, trading up-to-date data for dramatically faster reads.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Index the materialized view for fast filtering</span>
    </div>
    <div class="code-callout__body">
      <p>Indexes on <code>order_date</code> and <code>customer_name</code> make range scans and equality lookups on the materialized view just as fast as on a regular indexed table.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-33" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Trigger-based refresh: keep the view current on writes</span>
    </div>
    <div class="code-callout__body">
      <p>A statement-level trigger fires after any INSERT, UPDATE, or DELETE on <code>orders</code> and calls <code>REFRESH MATERIALIZED VIEW CONCURRENTLY</code>-which rebuilds without locking out readers. This keeps the view fresh without a manual cron job.</p>
    </div>
  </div>
</aside>
</div>

## Interactive Examples with Sample Data

### 1. Generate Sample Data

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
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
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Bulk-insert 1,000 customers with random join dates</span>
    </div>
    <div class="code-callout__body">
      <p><code>generate_series(1, 1000)</code> produces 1,000 integers. The SELECT uses each integer <code>i</code> to generate a name and email, and casts a random fraction of 365 to an integer to spread join dates across the past year.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-24" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Bulk-insert orders and products with random values</span>
    </div>
    <div class="code-callout__body">
      <p>5,000 orders are inserted with random customer IDs (1-1,000), random dates in the past 90 days, and random amounts. Then 100 products are inserted with random category IDs and prices. Together these three blocks seed a realistic dataset for join practice.</p>
    </div>
  </div>
</aside>
</div>

### 2. Analysis Queries

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
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
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE: per-customer order and spending summary</span>
    </div>
    <div class="code-callout__body">
      <p>LEFT JOIN keeps customers who have never ordered. <code>COUNT(DISTINCT DATE_TRUNC('month', …))</code> counts how many different calendar months the customer placed at least one order, a proxy for purchase consistency over time.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-27" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: classify each customer into a behaviour segment</span>
    </div>
    <div class="code-callout__body">
      <p>A <code>CASE</code> expression in both <code>SELECT</code> and <code>GROUP BY</code> labels each customer: never ordered, one-time, same-month repeat, or true returning. Grouping on the same expression aggregates counts and averages per segment for a high-level cohort summary.</p>
    </div>
  </div>
</aside>
</div>

Remember: "Efficient joins are the key to unlocking insights from your data!"

## Gotchas

- **Using INNER JOIN when you need LEFT JOIN silently drops rows**: If any customer has no orders, an INNER JOIN will exclude them from the result without an error. Decide which table is the "source of truth" first; if you want all rows from it, you need a LEFT JOIN, not the default INNER JOIN.
- **Adding a WHERE filter on the right-side table converts a LEFT JOIN into an INNER JOIN**: `LEFT JOIN orders o ON … WHERE o.status = 'shipped'` eliminates unmatched left rows because NULL does not equal `'shipped'`. Move the condition into the ON clause (`ON c.customer_id = o.customer_id AND o.status = 'shipped'`) to preserve left-side rows with no shipped orders.
- **Joining on a non-unique column causes row multiplication**: If `orders.customer_id` has duplicates (multiple orders per customer) and you join to another table that also has duplicates on the same key, each pair matches, producing more output rows than either source table. Always check cardinality with COUNT before joining on a column you haven't verified is unique on one side.
- **Forgetting to handle NULL after an outer join when aggregating**: After a LEFT JOIN, unmatched right-side columns are NULL. `SUM(o.total_amount)` returns NULL (not 0) for customers with no orders. Wrap aggregates with `COALESCE(SUM(…), 0)` to get meaningful values.
- **Using a comma in FROM instead of JOIN accidentally creates a Cartesian product**: `FROM orders, customers` (old-style) returns every order paired with every customer, M × N rows. The query runs without error and can silently produce billions of rows. Always use explicit `JOIN … ON` syntax.
- **Aliasing ambiguity when two joined tables have a column with the same name**: If both `orders` and `customers` have a `created_at` column, writing `SELECT created_at` raises an error or returns the wrong table's value depending on the database. Prefix every column reference with its table alias to avoid this.

## Next steps

- [Aggregations](aggregations.md), **GROUP BY** and summaries on joined result sets
- [Advanced SQL concepts](advanced-concepts.md), subqueries, CTEs, and window functions
- [SQL project](project.md), apply joins in a structured brief
- [Data wrangling (Module 2.2)](../2.2-data-wrangling/README.md), cleaning and shaping extracts
