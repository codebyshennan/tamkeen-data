# E-commerce Data Analysis Project: GlobalMart Analytics Platform

**After this lesson:** You produce a small set of documented SQL answers (segments, product performance, trends) that mirror real analyst work, using **JOIN**s, **CTEs**, and aggregates.

## Helpful video

High-level introduction to SQL and relational databases.

<iframe width="560" height="315" src="https://www.youtube.com/embed/27axs9dO7AE" title="What is SQL?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Overview

**Prerequisites:** Work through [Joins](joins.md) and [Aggregations](aggregations.md) first. Use the same database tooling as in the [module README](README.md).

> **Time needed:** Often 4-8 hours including exploration and write-up.

## Why this matters

This project is a capstone for Module 2.1: you combine **joins**, **aggregates**, **CTEs**, and conditional logic the way analysts do in practice, then document assumptions and results so someone else could reproduce your queries.

## Project Overview

This project implements a comprehensive analytics platform for GlobalMart, an e-commerce business. The platform provides insights into customer behavior, product performance, and business operations through SQL-based analysis.

The numbered sections below mirror a typical assignment brief: work through them in order, and treat the SQL as **reference patterns**-adapt table and column names to your own database.

## Analysis Components

### 1. Customer Analytics (30 points)

#### 1.1 Customer Segmentation (15 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
WITH customer_metrics AS (
    SELECT
        c.customer_id,
        c.join_date,
        COUNT(DISTINCT o.order_id) as total_orders,
        SUM(o.total_amount) as total_spent,
        AVG(o.total_amount) as avg_order_value,
        MAX(o.order_date) as last_order_date,
        COUNT(DISTINCT DATE_TRUNC('month', o.order_date)) as active_months,
        SUM(o.discount_amount) / NULLIF(SUM(o.total_amount), 0) * 100 as discount_rate
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY c.customer_id, c.join_date
),
customer_segments AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY total_spent DESC) as spending_quartile,
        NTILE(4) OVER (ORDER BY total_orders DESC) as frequency_quartile,
        CURRENT_DATE - last_order_date as days_since_last_order,
        total_spent / NULLIF(active_months, 0) as monthly_avg_spend
    FROM customer_metrics
)
SELECT
    customer_id,
    ROUND(total_spent::numeric, 2) as total_spent,
    total_orders,
    ROUND(avg_order_value::numeric, 2) as avg_order_value,
    active_months,
    ROUND(monthly_avg_spend::numeric, 2) as monthly_avg_spend,
    ROUND(discount_rate::numeric, 2) as discount_rate,
    days_since_last_order,
    CASE
        WHEN spending_quartile = 1 AND frequency_quartile = 1 THEN ' VIP'
        WHEN spending_quartile <= 2 AND frequency_quartile <= 2 THEN ' High Value'
        WHEN days_since_last_order <= 30 THEN ' Active'
        WHEN days_since_last_order <= 90 THEN ' At Risk'
        ELSE ' Churned'
    END as customer_segment,
    CASE
        WHEN discount_rate > 20 THEN ' Discount Sensitive'
        WHEN avg_order_value > 500 THEN ' Premium Buyer'
        WHEN total_orders > 12 THEN ' Regular Buyer'
        ELSE ' Standard'
    END as buying_pattern
FROM customer_segments
ORDER BY total_spent DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-14" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 1: aggregate 12-month order history per customer</span>
    </div>
    <div class="code-callout__body">
      <p>LEFT JOIN keeps customers who placed no orders in the period. The filter on <code>order_date &gt;= CURRENT_DATE - INTERVAL '12 months'</code> restricts to recent activity. <code>discount_rate</code> is the ratio of total discount to total spend, a proxy for price sensitivity.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 2: rank customers by spend and frequency quartiles</span>
    </div>
    <div class="code-callout__body">
      <p>Two <code>NTILE(4)</code> window functions independently rank customers into quartiles by spend and order frequency. <code>CURRENT_DATE - last_order_date</code> gives recency as an interval; <code>monthly_avg_spend</code> normalizes spend by active months to compare irregular buyers fairly.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-36" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: assign lifecycle segment label</span>
    </div>
    <div class="code-callout__body">
      <p>The first CASE maps the combination of spending and frequency quartiles (and recency) to lifecycle labels: VIP, High Value, Active, At Risk, or Churned. Results ordered by total spend descending surface the most valuable customers first.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="38-48" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Second CASE: assign buying behaviour pattern</span>
    </div>
    <div class="code-callout__body">
      <p>A separate CASE independently labels each customer's buying pattern based on discount sensitivity, average order size, and purchase frequency. This allows a customer to be both "VIP" (segment) and "Discount Sensitive" (pattern), two orthogonal dimensions.</p>
    </div>
  </div>
</aside>
</div>

#### 1.2 Customer Retention Analysis (15 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
WITH cohort_dates AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', join_date) as cohort_month,
        DATE_TRUNC('month', order_date) as order_month
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
),
cohort_size AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_id) as num_customers
    FROM cohort_dates
    GROUP BY cohort_month
),
retention_analysis AS (
    SELECT
        c.cohort_month,
        o.order_month,
        COUNT(DISTINCT c.customer_id) as active_customers,
        cs.num_customers as cohort_size,
        EXTRACT(MONTH FROM o.order_month - c.cohort_month) as months_since_join
    FROM cohort_dates c
    JOIN cohort_dates o ON c.customer_id = o.customer_id
    JOIN cohort_size cs ON cs.cohort_month = c.cohort_month
    GROUP BY c.cohort_month, o.order_month, cs.num_customers
)
SELECT
    cohort_month,
    cohort_size,
    months_since_join,
    active_customers,
    ROUND(
        (active_customers::float / cohort_size * 100)::numeric,
        2
    ) as retention_rate,
    CASE
        WHEN months_since_join = 0 THEN ' New'
        WHEN months_since_join <= 3 THEN ' Early'
        WHEN months_since_join <= 6 THEN ' Established'
        ELSE ' Loyal'
    END as cohort_stage
FROM retention_analysis
WHERE months_since_join <= 12
ORDER BY cohort_month DESC, months_since_join;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 1: pair each customer's cohort month with each order month</span>
    </div>
    <div class="code-callout__body">
      <p><code>DATE_TRUNC('month', join_date)</code> groups customers by the month they first joined, their cohort. The JOIN with <code>orders</code> produces one row per (customer, order), giving both the cohort month and the order month for cross-referencing.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-16" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 2: count original cohort size per month</span>
    </div>
    <div class="code-callout__body">
      <p>Groups cohort_dates by cohort month and counts distinct customers, this is the denominator for retention rate calculations. Every subsequent month's active customers are divided by this number.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="18-28" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 3: count active customers per cohort × order-month pair</span>
    </div>
    <div class="code-callout__body">
      <p>Self-joins cohort_dates (<code>c</code> for cohort month, <code>o</code> for order month) on <code>customer_id</code> to find which cohort members placed orders in each subsequent month. <code>EXTRACT(MONTH FROM order_month - cohort_month)</code> computes months since join.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="30-46" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: retention rate and cohort stage label</span>
    </div>
    <div class="code-callout__body">
      <p>Divides active customers by cohort size to get retention percentage. The CASE labels each cohort-month combination as New (month 0), Early (1-3), Established (4-6), or Loyal (7+). <code>WHERE months_since_join &lt;= 12</code> limits to the first year.</p>
    </div>
  </div>
</aside>
</div>

### 2. Product Performance (30 points)

#### 2.1 Product Analytics (15 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
WITH product_metrics AS (
    SELECT
        p.product_id,
        p.name as product_name,
        p.category,
        p.price,
        p.cost_price,
        COUNT(DISTINCT o.order_id) as total_orders,
        SUM(oi.quantity) as units_sold,
        SUM(oi.quantity * p.price) as gross_revenue,
        SUM(oi.quantity * (p.price - p.cost_price)) as gross_profit,
        AVG(r.rating) as avg_rating,
        COUNT(r.review_id) as review_count
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
    LEFT JOIN reviews r ON p.product_id = r.product_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY p.product_id, p.name, p.category, p.price, p.cost_price
),
product_rankings AS (
    SELECT
        *,
        ROUND((gross_revenue / NULLIF(units_sold, 0))::numeric, 2) as avg_selling_price,
        ROUND((gross_profit / NULLIF(gross_revenue, 0) * 100)::numeric, 2) as profit_margin,
        RANK() OVER (PARTITION BY category ORDER BY units_sold DESC) as category_rank,
        PERCENT_RANK() OVER (ORDER BY gross_revenue) as revenue_percentile
    FROM product_metrics
)
SELECT
    product_name,
    category,
    ROUND(price::numeric, 2) as list_price,
    units_sold,
    ROUND(gross_revenue::numeric, 2) as gross_revenue,
    ROUND(gross_profit::numeric, 2) as gross_profit,
    profit_margin,
    ROUND(avg_rating::numeric, 2) as avg_rating,
    review_count,
    category_rank,
    CASE
        WHEN revenue_percentile >= 0.9 THEN ' Top Performer'
        WHEN revenue_percentile >= 0.7 THEN ' High Performer'
        WHEN revenue_percentile >= 0.4 THEN ' Mid Performer'
        ELSE ' Under Performer'
    END as performance_tier,
    CASE
        WHEN profit_margin >= 50 THEN ' High Margin'
        WHEN profit_margin >= 25 THEN ' Good Margin'
        WHEN profit_margin >= 10 THEN ' Fair Margin'
        ELSE ' Low Margin'
    END as margin_category,
    CASE
        WHEN avg_rating >= 4.5 THEN ''
        WHEN avg_rating >= 4.0 THEN ''
        WHEN avg_rating >= 3.0 THEN ''
        WHEN avg_rating >= 2.0 THEN ''
        ELSE ''
    END as rating_display
FROM product_rankings
ORDER BY gross_revenue DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-20" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 1: join products → orders → reviews and aggregate 90-day metrics</span>
    </div>
    <div class="code-callout__body">
      <p>Three LEFT JOINs attach order items, orders, and reviews to each product. LEFT JOIN keeps products with no sales or reviews. Gross revenue is <code>quantity × price</code>; gross profit is <code>quantity × (price - cost)</code>. The WHERE filters to 90 days of activity.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="22-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 2: add derived metrics and rankings per category</span>
    </div>
    <div class="code-callout__body">
      <p><code>profit_margin</code> is gross profit as a percentage of revenue. <code>RANK() OVER (PARTITION BY category ORDER BY units_sold DESC)</code> ranks each product within its own category. <code>PERCENT_RANK()</code> gives the global revenue percentile across all products.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="31-47" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: present financials and rating</span>
    </div>
    <div class="code-callout__body">
      <p>Rounds all numeric columns for display. <code>revenue_percentile</code> thresholds (0.9, 0.7, 0.4) map to performance tier labels. The rating_display CASE converts the numeric average rating into a text representation.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="49-62" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Margin category label</span>
    </div>
    <div class="code-callout__body">
      <p>A separate CASE labels each product's profitability tier (High / Good / Fair / Low Margin) independently of its revenue performance tier. Together with the performance tier, this gives a 2×2 style view: a product can be high-revenue but low-margin (a pricing signal) or low-revenue but high-margin (a niche opportunity).</p>
    </div>
  </div>
</aside>
</div>

#### 2.2 Inventory Analysis (15 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
WITH inventory_metrics AS (
    SELECT
        p.product_id,
        p.name as product_name,
        p.category,
        p.stock_quantity,
        p.reorder_level,
        p.cost_price,
        SUM(oi.quantity) as units_sold_30d,
        COUNT(DISTINCT o.order_id) as order_count_30d,
        SUM(oi.quantity) / 30.0 as daily_demand
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY p.product_id, p.name, p.category, p.stock_quantity,
             p.reorder_level, p.cost_price
),
inventory_analysis AS (
    SELECT
        *,
        ROUND(stock_quantity / NULLIF(daily_demand, 0)) as days_of_inventory,
        stock_quantity * cost_price as inventory_value,
        CASE
            WHEN stock_quantity = 0 THEN 0
            ELSE ROUND(units_sold_30d::float / stock_quantity * 100, 2)
        END as inventory_turnover
    FROM inventory_metrics
)
SELECT
    product_name,
    category,
    stock_quantity,
    reorder_level,
    ROUND(daily_demand::numeric, 2) as daily_demand,
    days_of_inventory,
    ROUND(inventory_value::numeric, 2) as inventory_value,
    inventory_turnover,
    CASE
        WHEN stock_quantity = 0 THEN ' Out of Stock'
        WHEN stock_quantity <= reorder_level THEN ' Reorder Needed'
        WHEN days_of_inventory >= 90 THEN ' Overstocked'
        WHEN days_of_inventory >= 30 THEN ' Healthy Stock'
        ELSE ' Low Stock'
    END as stock_status,
    CASE
        WHEN inventory_turnover >= 50 THEN ' High Turnover'
        WHEN inventory_turnover >= 25 THEN ' Good Turnover'
        WHEN inventory_turnover >= 10 THEN ' Moderate Turnover'
        ELSE ' Slow Turnover'
    END as turnover_rate,
    CASE
        WHEN stock_quantity = 0 THEN 'Urgent Reorder'
        WHEN stock_quantity <= reorder_level THEN 'Place Order'
        WHEN days_of_inventory >= 90 THEN 'Consider Promotion'
        ELSE 'Monitor Stock'
    END as recommended_action
FROM inventory_analysis
ORDER BY inventory_value DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-17" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 1: join products → order items → orders and aggregate 30-day demand</span>
    </div>
    <div class="code-callout__body">
      <p>LEFT JOINs keep products with no sales. <code>SUM(oi.quantity) / 30.0</code> computes average daily demand from the last 30 days. The WHERE limits to recent orders; products with no orders in the period will have NULL demand.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-27" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 2: derive days-of-inventory, value, and turnover rate</span>
    </div>
    <div class="code-callout__body">
      <p><code>stock_quantity / NULLIF(daily_demand, 0)</code> estimates how many days of stock remain. <code>inventory_value = stock_quantity × cost_price</code> is the cash tied up in stock. <code>inventory_turnover</code> is units sold as a percentage of current stock, high values signal fast-moving items.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="29-43" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: stock status and turnover rate labels</span>
    </div>
    <div class="code-callout__body">
      <p>The first CASE classifies stock level (Out of Stock → Reorder Needed → Overstocked → Healthy Stock → Low Stock). The second CASE labels turnover speed. Both are surfaced as readable columns for buyer dashboards.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="44-59" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Recommended action based on stock urgency</span>
    </div>
    <div class="code-callout__body">
      <p>A third CASE translates the stock status into an actionable instruction for the buyer: Urgent Reorder, Place Order, Consider Promotion, or Monitor Stock. Results are ordered by inventory value descending so high-value stockouts appear first.</p>
    </div>
  </div>
</aside>
</div>

### 3. Business Operations (40 points)

#### 3.1 Sales Performance (15 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
WITH daily_sales AS (
    SELECT
        DATE_TRUNC('day', o.order_date) as sale_date,
        COUNT(DISTINCT o.order_id) as num_orders,
        COUNT(DISTINCT o.customer_id) as num_customers,
        SUM(o.total_amount) as revenue,
        SUM(o.shipping_cost) as shipping_cost,
        SUM(o.discount_amount) as discounts,
        COUNT(DISTINCT CASE
            WHEN c.join_date = DATE_TRUNC('day', o.order_date)
            THEN c.customer_id
        END) as new_customers
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY DATE_TRUNC('day', o.order_date)
),
sales_metrics AS (
    SELECT
        *,
        revenue - shipping_cost - discounts as net_revenue,
        revenue / NULLIF(num_orders, 0) as avg_order_value,
        revenue / NULLIF(num_customers, 0) as revenue_per_customer,
        new_customers::float / NULLIF(num_customers, 0) * 100 as new_customer_percentage,
        LAG(revenue) OVER (ORDER BY sale_date) as prev_day_revenue,
        LAG(num_orders) OVER (ORDER BY sale_date) as prev_day_orders,
        LAG(num_customers) OVER (ORDER BY sale_date) as prev_day_customers
    FROM daily_sales
),
sales_analysis AS (
    SELECT
        *,
        ROUND(
            ((revenue - prev_day_revenue) /
             NULLIF(prev_day_revenue, 0) * 100)::numeric,
            2
        ) as revenue_growth,
        ROUND(
            ((num_orders - prev_day_orders)::float /
             NULLIF(prev_day_orders, 0) * 100)::numeric,
            2
        ) as order_growth,
        ROUND(
            ((num_customers - prev_day_customers)::float /
             NULLIF(prev_day_customers, 0) * 100)::numeric,
            2
        ) as customer_growth,
        AVG(revenue) OVER (
            ORDER BY sale_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as revenue_7day_avg,
        AVG(num_orders) OVER (
            ORDER BY sale_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as orders_7day_avg
    FROM sales_metrics
)
SELECT
    sale_date,
    num_orders,
    num_customers,
    new_customers,
    ROUND(new_customer_percentage::numeric, 2) as new_customer_pct,
    ROUND(revenue::numeric, 2) as revenue,
    ROUND(net_revenue::numeric, 2) as net_revenue,
    ROUND(avg_order_value::numeric, 2) as aov,
    ROUND(revenue_per_customer::numeric, 2) as revenue_per_customer,
    ROUND(revenue_7day_avg::numeric, 2) as revenue_7day_avg,
    ROUND(orders_7day_avg::numeric, 1) as orders_7day_avg,
    revenue_growth,
    order_growth,
    customer_growth,
    CASE
        WHEN revenue_growth >= 20 THEN ' High Growth'
        WHEN revenue_growth > 0 THEN ' Growing'
        WHEN revenue_growth > -20 THEN ' Declining'
        ELSE ' Sharp Decline'
    END as revenue_trend,
    CASE
        WHEN customer_growth >= 20 THEN ' Strong Acquisition'
        WHEN customer_growth > 0 THEN ' Growing Base'
        WHEN customer_growth > -20 THEN ' Customer Loss'
        ELSE ' High Churn'
    END as customer_trend
FROM sales_analysis
ORDER BY sale_date DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-17" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 1: aggregate daily order, revenue, and new-customer metrics</span>
    </div>
    <div class="code-callout__body">
      <p>Groups orders by day. <code>COUNT(DISTINCT CASE WHEN c.join_date = DATE_TRUNC('day', o.order_date) THEN c.customer_id END)</code> counts customers whose join date matches today's date, a proxy for new customers placing their first order on each day.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-30" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 2: derive per-day KPIs and prior-day values with LAG</span>
    </div>
    <div class="code-callout__body">
      <p>Computes net revenue (revenue minus shipping and discounts), AOV, and new-customer percentage. Three <code>LAG(…) OVER (ORDER BY sale_date)</code> calls capture the previous day's values for revenue, orders, and customers, used to calculate day-over-day growth in the next CTE.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="32-57" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 3: compute growth rates and 7-day rolling averages</span>
    </div>
    <div class="code-callout__body">
      <p>Three growth rate columns compare today to yesterday using <code>(current - prior) / NULLIF(prior, 0) * 100</code>. Two window functions with <code>ROWS BETWEEN 6 PRECEDING AND CURRENT ROW</code> smooth revenue and order counts into 7-day rolling averages.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="59-73" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: present all KPIs and rolling averages</span>
    </div>
    <div class="code-callout__body">
      <p>Selects and rounds all KPIs for display. Revenue and order growth rates feed the trend CASE labels in the next lines; AOV and revenue-per-customer give profitability context per day.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="75-87" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Revenue and customer trend labels</span>
    </div>
    <div class="code-callout__body">
      <p>Two CASE expressions translate numeric growth rates into readable trend labels (High Growth → Growing → Declining → Sharp Decline for revenue; Strong Acquisition → Growing Base → Customer Loss → High Churn for customers). Results are ordered most-recent-first.</p>
    </div>
  </div>
</aside>
</div>

#### 3.2 Marketing Campaign Analysis (15 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
WITH campaign_metrics AS (
    SELECT
        mc.campaign_id,
        mc.name as campaign_name,
        mc.start_date,
        mc.end_date,
        mc.budget,
        mc.target_segment,
        mc.channel,
        SUM(cp.impressions) as total_impressions,
        SUM(cp.clicks) as total_clicks,
        SUM(cp.conversions) as total_conversions,
        SUM(cp.spend) as total_spend,
        SUM(cp.revenue) as total_revenue,
        COUNT(DISTINCT DATE_TRUNC('day', cp.date)) as campaign_days
    FROM marketing_campaigns mc
    LEFT JOIN campaign_performance cp ON mc.campaign_id = cp.campaign_id
    GROUP BY
        mc.campaign_id, mc.name, mc.start_date, mc.end_date,
        mc.budget, mc.target_segment, mc.channel
),
campaign_kpis AS (
    SELECT
        *,
        CASE
            WHEN total_impressions > 0
            THEN ROUND((total_clicks::float / total_impressions * 100)::numeric, 2)
            ELSE 0
        END as ctr,
        CASE
            WHEN total_clicks > 0
            THEN ROUND((total_conversions::float / total_clicks * 100)::numeric, 2)
            ELSE 0
        END as conversion_rate,
        CASE
            WHEN total_conversions > 0
            THEN ROUND((total_revenue / total_conversions)::numeric, 2)
            ELSE 0
        END as revenue_per_conversion,
        CASE
            WHEN total_spend > 0
            THEN ROUND((total_revenue / total_spend)::numeric, 2)
            ELSE 0
        END as roas,
        ROUND((total_spend / NULLIF(total_clicks, 0))::numeric, 2) as cpc,
        ROUND((total_spend / NULLIF(total_conversions, 0))::numeric, 2) as cpa,
        total_revenue - total_spend as profit,
        CASE
            WHEN total_spend > 0
            THEN ROUND(((total_revenue - total_spend) / total_spend * 100)::numeric, 2)
            ELSE 0
        END as roi
    FROM campaign_metrics
)
SELECT
    campaign_name,
    channel,
    target_segment,
    start_date,
    end_date,
    campaign_days,
    ROUND(budget::numeric, 2) as budget,
    ROUND(total_spend::numeric, 2) as spend,
    ROUND((total_spend / budget * 100)::numeric, 2) as budget_utilization,
    total_impressions,
    total_clicks,
    total_conversions,
    ctr as click_through_rate,
    conversion_rate,
    ROUND(total_revenue::numeric, 2) as revenue,
    ROUND(profit::numeric, 2) as profit,
    roas as return_on_ad_spend,
    roi as return_on_investment,
    cpc as cost_per_click,
    cpa as cost_per_acquisition,
    CASE
        WHEN roi >= 100 THEN 'Exceptional'
        WHEN roi >= 50 THEN 'Strong'
        WHEN roi >= 0 THEN 'Acceptable'
        ELSE 'Poor'
    END as performance_category,
    CASE
        WHEN roi < 0 THEN 'Pause Campaign'
        WHEN cpa > revenue_per_conversion THEN 'Optimize Targeting'
        WHEN budget_utilization < 80 THEN 'Increase Budget'
        WHEN conversion_rate < 2 THEN 'Improve Landing Page'
        WHEN ctr < 1 THEN 'Revise Ad Creative'
        ELSE 'Maintain Strategy'
    END as recommended_action
FROM campaign_kpis
ORDER BY roi DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-21" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 1: aggregate impressions, clicks, conversions, and spend per campaign</span>
    </div>
    <div class="code-callout__body">
      <p>LEFT JOIN <code>campaign_performance</code> keeps campaigns even if they have no daily performance rows yet. SUM aggregates all daily records into campaign-level totals. <code>COUNT(DISTINCT DATE_TRUNC('day', cp.date))</code> counts how many days the campaign has data for.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-52" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 2: compute CTR, conversion rate, ROAS, CPC, CPA, and ROI</span>
    </div>
    <div class="code-callout__body">
      <p>Each KPI uses a guarded CASE to avoid division-by-zero. CTR = clicks / impressions; conversion rate = conversions / clicks; ROAS = revenue / spend; CPC = spend / clicks; CPA = spend / conversions; ROI = (revenue - spend) / spend × 100. All null-safe via <code>NULLIF</code> or inner <code>WHEN … &gt; 0</code> guards.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="54-72" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: present metrics and performance category</span>
    </div>
    <div class="code-callout__body">
      <p>Selects all KPIs from the second CTE and computes <code>budget_utilization</code> inline as spend / budget × 100. The performance_category CASE buckets campaigns by ROI into Exceptional / Strong / Acceptable / Poor.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="74-92" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Recommended action based on multi-signal diagnosis</span>
    </div>
    <div class="code-callout__body">
      <p>The action CASE checks multiple diagnostic signals in priority order: negative ROI → pause; CPA exceeds revenue per conversion → optimize targeting; low budget use → increase budget; low conversion rate → fix landing page; low CTR → revise creative. Results ordered by ROI descending.</p>
    </div>
  </div>
</aside>
</div>

#### 3.3 Supply Chain Efficiency (10 points)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
WITH supplier_metrics AS (
    SELECT
        s.supplier_id,
        s.company_name,
        s.country,
        s.lead_time_days,
        COUNT(DISTINCT p.product_id) as products_supplied,
        SUM(p.stock_quantity) as total_inventory,
        SUM(p.stock_quantity * p.cost_price) as inventory_value,
        COUNT(DISTINCT o.order_id) as fulfilled_orders,
        AVG(EXTRACT(EPOCH FROM (sh.actual_delivery - sh.ship_date)) / 86400) as avg_delivery_days,
        COUNT(DISTINCT CASE
            WHEN sh.actual_delivery > sh.estimated_delivery
            THEN sh.shipment_id
        END)::float / NULLIF(COUNT(DISTINCT sh.shipment_id), 0) * 100 as late_delivery_rate
    FROM suppliers s
    LEFT JOIN products p ON s.supplier_id = p.supplier_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
    LEFT JOIN shipments sh ON o.order_id = sh.order_id
    GROUP BY s.supplier_id, s.company_name, s.country, s.lead_time_days
),
supplier_performance AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY late_delivery_rate DESC) as reliability_quartile,
        NTILE(4) OVER (ORDER BY avg_delivery_days DESC) as speed_quartile,
        NTILE(4) OVER (ORDER BY inventory_value DESC) as value_quartile
    FROM supplier_metrics
)
SELECT
    company_name,
    country,
    lead_time_days,
    products_supplied,
    total_inventory,
    ROUND(inventory_value::numeric, 2) as inventory_value,
    fulfilled_orders,
    ROUND(avg_delivery_days::numeric, 1) as avg_delivery_days,
    ROUND(late_delivery_rate::numeric, 2) as late_delivery_rate,
    CASE
        WHEN reliability_quartile = 1 THEN 'High Risk'
        WHEN reliability_quartile = 2 THEN 'Medium Risk'
        WHEN reliability_quartile = 3 THEN 'Low Risk'
        ELSE 'Very Reliable'
    END as reliability_rating,
    CASE
        WHEN speed_quartile = 1 THEN 'Slow'
        WHEN speed_quartile = 2 THEN 'Moderate'
        WHEN speed_quartile = 3 THEN 'Fast'
        ELSE 'Very Fast'
    END as speed_rating,
    CASE
        WHEN value_quartile = 1 THEN 'Strategic'
        WHEN value_quartile = 2 THEN 'Major'
        WHEN value_quartile = 3 THEN 'Medium'
        ELSE 'Minor'
    END as value_rating,
    CASE
        WHEN late_delivery_rate > 20 OR avg_delivery_days > lead_time_days * 1.5
        THEN 'Review Partnership'
        WHEN late_delivery_rate > 10 OR avg_delivery_days > lead_time_days * 1.2
        THEN 'Needs Improvement'
        WHEN late_delivery_rate > 5 OR avg_delivery_days > lead_time_days
        THEN 'Monitor Closely'
        ELSE 'Good Standing'
    END as supplier_status
FROM supplier_performance
ORDER BY inventory_value DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-22" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 1: join suppliers → products → orders → shipments and aggregate</span>
    </div>
    <div class="code-callout__body">
      <p>Four LEFT JOINs chain from supplier through products, order items, orders, and shipments. <code>AVG(EXTRACT(EPOCH FROM …)/86400)</code> gives mean actual delivery time in days. The late-delivery rate uses <code>COUNT(DISTINCT CASE WHEN actual &gt; estimated THEN shipment_id END) / NULLIF(total, 0)</code>-a null-safe percentage.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-30" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE 2: rank suppliers into quartiles by reliability, speed, and value</span>
    </div>
    <div class="code-callout__body">
      <p>Three <code>NTILE(4)</code> window functions independently rank suppliers by late-delivery rate (reliability), average delivery days (speed), and inventory value (strategic importance). Quartile 1 = worst for rate/days, best for value.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="32-50" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer query: translate quartiles into reliability and speed ratings</span>
    </div>
    <div class="code-callout__body">
      <p>Two CASE expressions convert reliability and speed quartile numbers into human-readable labels (High Risk → Very Reliable; Slow → Very Fast). A third converts the value quartile to strategic tier (Strategic → Minor).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="52-69" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Supplier status: escalating action recommendation</span>
    </div>
    <div class="code-callout__body">
      <p>The final CASE checks late-delivery rate and actual vs. expected lead time together. Thresholds escalate from Good Standing → Monitor Closely → Needs Improvement → Review Partnership. Results ordered by inventory value so the most strategically important suppliers appear first.</p>
    </div>
  </div>
</aside>
</div>

## Implementation Guidelines

### 1. Project Setup

1. Create database and tables
2. Import sample data
3. Create necessary indexes
4. Set up monitoring queries

### 2. Analysis Workflow

1. Run customer analytics
2. Analyze product performance
3. Review business operations
4. Generate recommendations

### 3. Performance Optimization

1. Index strategy

   <div class="code-explainer" data-code-explainer>
   <div class="code-explainer__code">

   {% highlight sql %}
      -- Indexes for frequent joins
      CREATE INDEX idx_orders_customer ON orders(customer_id);
      CREATE INDEX idx_orders_date ON orders(order_date);

      -- Indexes for range queries
      CREATE INDEX idx_products_price ON products(price);
      CREATE INDEX idx_inventory_stock ON products(stock_quantity);

      -- Composite indexes for common query patterns
      CREATE INDEX idx_orders_customer_date
      ON orders(customer_id, order_date DESC);
   {% endhighlight %}
   </div>
   <aside class="code-explainer__callouts" aria-label="Code walkthrough">
     <div class="code-callout" data-lines="1-4" data-tint="1">
       <div class="code-callout__meta">
         <span class="code-callout__lines"></span>
         <span class="code-callout__title">Single-column indexes on FK and date columns</span>
       </div>
       <div class="code-callout__body">
         <p>Indexes on <code>customer_id</code> and <code>order_date</code> speed up the most common join condition and date range filters. Similar single-column indexes on <code>price</code> and <code>stock_quantity</code> cover range scans in inventory and product queries.</p>
       </div>
     </div>
     <div class="code-callout" data-lines="6-9" data-tint="2">
       <div class="code-callout__meta">
         <span class="code-callout__lines"></span>
         <span class="code-callout__title">Composite index for per-customer date-ordered queries</span>
       </div>
       <div class="code-callout__body">
         <p>The composite <code>(customer_id, order_date DESC)</code> index satisfies queries that filter by customer and sort by date in one index scan, no separate sort step needed. The leading equality column prunes by customer first, then the date sort is already in order.</p>
       </div>
     </div>
   </aside>
   </div>

2. Query optimization
   - Use CTEs for complex queries
   - Apply appropriate join types
   - Filter early in the query
   - Use covering indexes

3. Maintenance

   <div class="code-explainer" data-code-explainer>
   <div class="code-explainer__code">

   {% highlight sql %}
      -- Regular statistics update
      ANALYZE customers;
      ANALYZE orders;
      ANALYZE products;

      -- Monitor query performance
      SELECT * FROM pg_stat_statements
      ORDER BY total_time DESC
      LIMIT 10;
   {% endhighlight %}
   </div>
   <aside class="code-explainer__callouts" aria-label="Code walkthrough">
     <div class="code-callout" data-lines="1-5" data-tint="1">
       <div class="code-callout__meta">
         <span class="code-callout__lines"></span>
         <span class="code-callout__title">ANALYZE: refresh planner statistics</span>
       </div>
       <div class="code-callout__body">
         <p><code>ANALYZE</code> updates the internal statistics tables that the query planner uses to estimate row counts and choose join strategies. After bulk loads or schema changes, outdated statistics cause the planner to pick bad plans.</p>
       </div>
     </div>
     <div class="code-callout" data-lines="7-9" data-tint="2">
       <div class="code-callout__meta">
         <span class="code-callout__lines"></span>
         <span class="code-callout__title">pg_stat_statements: find the slowest queries</span>
       </div>
       <div class="code-callout__body">
         <p><code>pg_stat_statements</code> tracks execution statistics for every query the server runs. Ordering by <code>total_time DESC</code> and limiting to 10 surfaces the queries consuming the most cumulative time, the prime candidates for indexing or rewriting.</p>
       </div>
     </div>
   </aside>
   </div>

## Deliverables

1. SQL Scripts
   - Table creation
   - Data import
   - Analysis queries
   - Optimization code

2. Documentation
   - Schema design
   - Query explanations
   - Performance notes
   - Recommendations

3. Visualizations
   - Customer segments
   - Product performance
   - Sales trends
   - Campaign effectiveness

## Success Metrics

1. Query Performance
   - Execution time < 5 seconds
   - Efficient resource usage
   - Proper index utilization

2. Analysis Quality
   - Accurate insights
   - Actionable recommendations
   - Clear documentation

3. Business Impact
   - Improved customer retention
   - Optimized inventory
   - Increased sales
   - Better marketing ROI

## Gotchas

- **The LEFT JOIN in `customer_metrics` still drops customers who ordered outside the 12-month window**: The CTE uses `LEFT JOIN orders o ON … WHERE o.order_date >= CURRENT_DATE - INTERVAL '12 months'`. Because the date filter is in WHERE (not in the ON clause), customers whose only orders are older than 12 months are excluded entirely rather than showing zero activity. Move the date condition into the ON clause to preserve all customers.
- **NTILE assigns quartile 1 to the top spenders, not the bottom**: Because `NTILE(4) OVER (ORDER BY total_spent DESC)` sorts descending, quartile 1 is the highest-spending group. When labelling segments, "quartile 1 = VIP" is correct here, but reversing the ORDER BY direction in other uses would flip the meaning silently.
- **`EXTRACT(MONTH FROM age)` in the retention CTE only returns the month component, not total months elapsed**: If a customer joins in January 2023 and places an order in January 2024, `EXTRACT(MONTH FROM order_month - cohort_month)` returns 0 (same month number), not 12. Use `EXTRACT(YEAR FROM age) * 12 + EXTRACT(MONTH FROM age)` or `DATE_PART('month', AGE(order_month, cohort_month))` to get total months since join.
- **LEFT JOIN + WHERE on order date silently removes zero-sales products from the 90-day product analysis**: The `product_metrics` CTE LEFT JOINs to orders but then filters `WHERE o.order_date >= CURRENT_DATE - INTERVAL '90 days'`. Products with no recent orders have NULL for `o.order_date`, which fails the filter and is excluded. To keep unsold products, move the date filter into the ON clause or add `OR o.order_date IS NULL`.
- **NULLIF in discount-rate and profit-margin calculations silently returns NULL instead of zero**: `NULLIF(SUM(o.total_amount), 0)` prevents division by zero but returns NULL for customers with no spend, which then propagates through the ratio. Wrap the whole expression in `COALESCE(… , 0)` if a zero rate is more meaningful to downstream logic than NULL.
- **Hardcoded CASE thresholds become stale as the dataset grows**: Segment boundaries like `avg_order_value > 500` for "Premium Buyer" or `discount_rate > 20` for "Discount Sensitive" are calibrated to the current data distribution. After ingesting new data, re-examine these cutoffs with percentile queries rather than assuming the original numbers still separate segments meaningfully.

Remember: "Data-driven decisions lead to better business outcomes!"
