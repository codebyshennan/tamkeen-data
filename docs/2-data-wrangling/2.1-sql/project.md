# E-commerce Data Analysis Project: GlobalMart Analytics Platform 🚀

## Project Overview 📊

This project implements a comprehensive analytics platform for GlobalMart, an e-commerce business. The platform provides insights into customer behavior, product performance, and business operations through SQL-based analysis.

## Analysis Components 📈

### 1. Customer Analytics (30 points)

#### 1.1 Customer Segmentation (15 points)
```sql
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
        WHEN spending_quartile = 1 AND frequency_quartile = 1 THEN '💎 VIP'
        WHEN spending_quartile <= 2 AND frequency_quartile <= 2 THEN '🌟 High Value'
        WHEN days_since_last_order <= 30 THEN '✨ Active'
        WHEN days_since_last_order <= 90 THEN '😴 At Risk'
        ELSE '💔 Churned'
    END as customer_segment,
    CASE 
        WHEN discount_rate > 20 THEN '🎯 Discount Sensitive'
        WHEN avg_order_value > 500 THEN '💰 Premium Buyer'
        WHEN total_orders > 12 THEN '🔄 Regular Buyer'
        ELSE '👤 Standard'
    END as buying_pattern
FROM customer_segments
ORDER BY total_spent DESC;
```

#### 1.2 Customer Retention Analysis (15 points)
```sql
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
        WHEN months_since_join = 0 THEN '🆕 New'
        WHEN months_since_join <= 3 THEN '✨ Early'
        WHEN months_since_join <= 6 THEN '🌟 Established'
        ELSE '💫 Loyal'
    END as cohort_stage
FROM retention_analysis
WHERE months_since_join <= 12
ORDER BY cohort_month DESC, months_since_join;
```

### 2. Product Performance (30 points)

#### 2.1 Product Analytics (15 points)
```sql
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
        WHEN revenue_percentile >= 0.9 THEN '🏆 Top Performer'
        WHEN revenue_percentile >= 0.7 THEN '⭐ High Performer'
        WHEN revenue_percentile >= 0.4 THEN '📊 Mid Performer'
        ELSE '⚠️ Under Performer'
    END as performance_tier,
    CASE 
        WHEN profit_margin >= 50 THEN '💰 High Margin'
        WHEN profit_margin >= 25 THEN '💵 Good Margin'
        WHEN profit_margin >= 10 THEN '⚖️ Fair Margin'
        ELSE '⚠️ Low Margin'
    END as margin_category,
    CASE 
        WHEN avg_rating >= 4.5 THEN '⭐⭐⭐⭐⭐'
        WHEN avg_rating >= 4.0 THEN '⭐⭐⭐⭐'
        WHEN avg_rating >= 3.0 THEN '⭐⭐⭐'
        WHEN avg_rating >= 2.0 THEN '⭐⭐'
        ELSE '⭐'
    END as rating_display
FROM product_rankings
ORDER BY gross_revenue DESC;
```

#### 2.2 Inventory Analysis (15 points)
```sql
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
        WHEN stock_quantity = 0 THEN '🚨 Out of Stock'
        WHEN stock_quantity <= reorder_level THEN '⚠️ Reorder Needed'
        WHEN days_of_inventory >= 90 THEN '💤 Overstocked'
        WHEN days_of_inventory >= 30 THEN '✅ Healthy Stock'
        ELSE '📊 Low Stock'
    END as stock_status,
    CASE 
        WHEN inventory_turnover >= 50 THEN '🔄 High Turnover'
        WHEN inventory_turnover >= 25 THEN '📈 Good Turnover'
        WHEN inventory_turnover >= 10 THEN '📊 Moderate Turnover'
        ELSE '📉 Slow Turnover'
    END as turnover_rate,
    CASE 
        WHEN stock_quantity = 0 THEN 'Urgent Reorder'
        WHEN stock_quantity <= reorder_level THEN 'Place Order'
        WHEN days_of_inventory >= 90 THEN 'Consider Promotion'
        ELSE 'Monitor Stock'
    END as recommended_action
FROM inventory_analysis
ORDER BY inventory_value DESC;
```

### 3. Business Operations (40 points)

#### 3.1 Sales Performance (15 points)
```sql
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
        WHEN revenue_growth >= 20 THEN '🚀 High Growth'
        WHEN revenue_growth > 0 THEN '📈 Growing'
        WHEN revenue_growth > -20 THEN '📉 Declining'
        ELSE '⚠️ Sharp Decline'
    END as revenue_trend,
    CASE 
        WHEN customer_growth >= 20 THEN '🎯 Strong Acquisition'
        WHEN customer_growth > 0 THEN '👥 Growing Base'
        WHEN customer_growth > -20 THEN '⚠️ Customer Loss'
        ELSE '🚨 High Churn'
    END as customer_trend
FROM sales_analysis
ORDER BY sale_date DESC;
```

#### 3.2 Marketing Campaign Analysis (15 points)
```sql
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
```

#### 3.3 Supply Chain Efficiency (10 points)
```sql
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
```

## Implementation Guidelines 📋

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
   ```sql
   -- Indexes for frequent joins
   CREATE INDEX idx_orders_customer ON orders(customer_id);
   CREATE INDEX idx_orders_date ON orders(order_date);
   
   -- Indexes for range queries
   CREATE INDEX idx_products_price ON products(price);
   CREATE INDEX idx_inventory_stock ON products(stock_quantity);
   
   -- Composite indexes for common query patterns
   CREATE INDEX idx_orders_customer_date 
   ON orders(customer_id, order_date DESC);
   ```

2. Query optimization
   - Use CTEs for complex queries
   - Apply appropriate join types
   - Filter early in the query
   - Use covering indexes

3. Maintenance
   ```sql
   -- Regular statistics update
   ANALYZE customers;
   ANALYZE orders;
   ANALYZE products;
   
   -- Monitor query performance
   SELECT * FROM pg_stat_statements 
   ORDER BY total_time DESC 
   LIMIT 10;
   ```

## Deliverables 📊

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

## Success Metrics 📈

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

Remember: "Data-driven decisions lead to better business outcomes!" 🎯

