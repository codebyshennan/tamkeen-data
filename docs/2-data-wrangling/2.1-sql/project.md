# E-commerce Data Analysis Project: GlobalMart Analytics Platform 🚀

[Previous content remains the same until the sales_metrics CTE]

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
