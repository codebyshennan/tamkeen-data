# Business Intelligence with Tableau

**After this submodule:** you can use the lessons linked below and complete the exercises that match **3.3 Business Intelligence with Tableau** in your course schedule.

## Helpful video

Short Tableau Public install; pair with the written guides in this folder.

<iframe width="560" height="315" src="https://www.youtube.com/embed/lTNWfhmurUg" title="Tableau Public Tutorial Download and Setup" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

You have data. Your manager wants answers. Business Intelligence tools like Tableau, Power BI, and Looker Studio let you go from raw numbers to interactive dashboards that anyone can explore, without writing a single line of SQL. By the end of this submodule, you'll have built real dashboards in at least one of these tools.

## First 30 Minutes

New to BI tools? Do these five steps before reading anything else:

1. **Download and install Tableau Public** (free) from [public.tableau.com](https://public.tableau.com/en-us/s/download). Takes about 5 minutes.
2. **Download the [Sample Superstore dataset](assets/sample_superstore.xls)** and save it to your Desktop. On the Tableau start screen, click **Microsoft Excel** under "To a File" and open it.
3. **Drag `Sales` to the Rows shelf and `Category` to the Columns shelf.** You just made your first bar chart. Click "Show Me" on the top right to explore other chart types with the same fields.
4. **Add a filter**: drag `Region` to the Filters shelf, select a couple of regions, and watch the chart update instantly.
5. **Create a new Dashboard** (bottom tab menu → New Dashboard), drag your sheet onto the canvas, and share the URL. That's a real dashboard.

You've now touched every major concept: data source, worksheet, chart, filter, dashboard. The rest of the submodule goes deeper on each.

## Tool Comparison

All three tools below can take you from raw data to a shareable dashboard. Pick the one your workplace uses, or use Tableau for this course.

| Tool | Free tier? | Platform | Best for | Learning curve |
|---|---|---|---|---|
| **Tableau Public** | Yes (public dashboards only) | Windows, Mac | Visual exploration, storytelling, advanced analytics | Moderate |
| **Power BI Desktop** | Yes (free desktop app) | Windows only (Mac via browser) | Microsoft/Excel shops, enterprise reporting | Moderate |
| **Looker Studio** | Yes (fully free) | Web browser (any OS) | Google Analytics / Google Sheets users, quick sharing | Low |

> For this course, lessons and case studies focus on **Tableau**, with companion lessons for Looker Studio and Power BI.

## Overview

This submodule is **UI-first**: you will connect data and build views in **Tableau** (and related lessons may reference **Looker Studio** or **Power BI**) with drag-and-drop analytics instead of Python scripts.

> **Note:** Perception and chart-choice principles still apply; see [3.1 visualization principles](../3.1-intro-data-viz/visualization-principles.md) when you are unsure which visual to use.

> **Time needed:** Plan several hours across the case studies and practice labs if you have access to the tools.

**Submodule map**

| Lesson | What you'll learn |
|--------|------------------|
| [Tableau Basics](tableau-basics.md) | Connect data, build charts, filters, dashboards |
| [Tableau Case Study](tableau-case-study.md) | End-to-end Superstore analysis with calculated fields and LOD |
| [Advanced Analytics](advanced-analytics.md) | Table calculations, parameters, dashboard actions |
| [AI Tools Integration](ai-tools-integration.md) | Use Claude/ChatGPT to write formulas and debug faster |
| [Power BI Case Study](powerbi-case-study.md) | Build a report in Power BI with DAX and Power Query |
| [Looker Studio Case Study](looker-studio-case-study.md) | Build a dashboard in Looker Studio from a Google Sheet |
| [Assignment](/assignments/bi-dashboard-project.md) | Build your own dashboard answering 3 business questions |

## Prerequisites

- [3.1 Intro to data visualization](../3.1-intro-data-viz/README.md) and basic comfort with spreadsheets or SQL-style fields (dimensions vs measures).
- Tableau Desktop or a comparable lab environment (your instructor may specify Tableau Public or server access).

## Key Vocabulary

Before diving in, make sure these terms are clear. You'll see them constantly.

| Term | What it means | Example |
|---|---|---|
| **Dimension** | A categorical field, something you group or filter by | `Region`, `Product Category`, `Customer Name` |
| **Measure** | A numeric field you aggregate (sum, average, count) | `Sales`, `Profit`, `Quantity` |
| **Dashboard** | A canvas holding multiple charts that update together | A page showing sales KPIs + map + trend line |
| **Filter** | A control that limits which rows appear in a view | Show only orders from 2023, or only the West region |
| **Data Source** | The connection between Tableau and your data file or database | An Excel file, a PostgreSQL table, a Google Sheet |
| **Extract / Live Connection** | How Tableau reads your data: a static snapshot (Extract) vs always-current (Live) | Extract = fast offline; Live = always up to date |

## Why Tableau?

### 1. Intuitive Design

**Key Features**

| Feature | What it does |
|---------|-------------|
| Drag-and-Drop | Visual field mapping, instant chart creation, dynamic filtering |
| Visual Analytics | Automatic insights, statistical summaries, trend analysis |
| Rapid Development | Quick prototyping, instant feedback, easy iteration |

### 2. Enterprise Power

**Capabilities**

| Capability | Details |
|-----------|---------|
| Data Handling | Live connections, data extracts, incremental updates |
| Security | Row-level security, user authentication, data encryption |
| Scalability | Big data ready, server deployment, cloud integration |

## Core Concepts

### 1. Data Architecture

Every Tableau dashboard follows the same three-stage journey from raw numbers to insight:

| Stage | What happens | Example |
|-------|-------------|---------|
| **Data Source** | Tableau connects to your raw data, a file, a database, or a cloud service | An Excel spreadsheet, a PostgreSQL table, a Google Sheet |
| **Extract / Process** | Tableau reads and optimises the data for fast querying (live or as a snapshot) | Tableau builds a `.hyper` extract file from your 500 k-row CSV |
| **Visualization** | You drag fields onto shelves to build charts, filters, and dashboards | A bar chart showing profit by region, embedded in a team dashboard |

This pipeline runs every time you open a workbook or refresh an extract. Understanding it helps you choose the right connection type (see below) and diagnose slow dashboards.

#### Connection Types

| | **Live Connection** | **Extract** |
|---|---|---|
| **Data freshness** | Always current, Tableau queries the source on every interaction | Snapshot taken at extract time; stale until you refresh |
| **Performance** | Slower on large datasets; every filter triggers a new database query | Fast, Tableau reads from an optimised local `.hyper` file |
| **Works offline** | No, requires a live network path to the data source | Yes, the extract file is stored locally |
| **Storage needed** | None on the Tableau side | Yes, the `.hyper` file can be large for wide datasets |
| **Best for** | Real-time operational dashboards, small-to-medium data | Large datasets, shared published workbooks, scheduled refreshes |

> **Rule of thumb:** start with a Live connection while you explore the data, then switch to an Extract before publishing so your colleagues get fast load times.

> **Ask AI (Claude or ChatGPT)**
>
> "I'm connecting Tableau to a [describe your data source, e.g. 'MySQL database with 5 million rows updated hourly']. Should I use a live connection or an extract? What are the trade-offs for my specific situation?"

### 2. Visual Grammar

The Tableau workspace is where all the action happens. The left panel shows your fields (dimensions in blue, measures in green), the shelves at the top control the chart structure, and the Marks card controls color, size, and labels.

<!-- TODO: screenshot, Tableau workspace showing the Rows/Columns shelves, Marks card, and field list panel -->

Drag a field onto the canvas or onto a shelf to build a chart. Tableau's Show Me panel suggests chart types based on what you've dragged in.

<!-- TODO: screenshot, Dragging a measure onto the Rows shelf to build a bar chart in Tableau -->

#### Basic Charts

**Chart Selection**

| Goal | Chart types |
|------|------------|
| Comparison | Bar charts (categories), line charts (time), bullet charts (targets) |
| Distribution | Histograms (frequency), box plots (statistics), heat maps (density) |
| Composition | Pie charts (parts), tree maps (hierarchy), stacked bars (over time) |
| Relationship | Scatter plots (correlation), bubble charts (3 variables), connected scatter (paths) |

> **Ask AI (Claude or ChatGPT)**
>
> "I want to show [describe your goal, e.g. 'how monthly revenue has changed across 4 product categories over the past 2 years']. What chart type should I use in Tableau, and how do I set it up, which fields go on Rows, Columns, and Marks?"

#### Visual Best Practices

**Design Principles**

**Color Usage:**
- Sequential: Ordered data
- Diverging: Mid-point data
- Categorical: Distinct groups

**Layout:**
- Grid alignment
- Visual hierarchy
- White space

**Typography:**
- Clear hierarchy
- Consistent fonts
- Readable sizes

### 3. Calculations

#### Basic Formulas

```sql
-- Year-over-Year Growth
YOY_Growth =
([Sales] - LOOKUP([Sales], -1)) /
LOOKUP([Sales], -1)

-- Moving Average
Moving_Avg =
WINDOW_AVG([Value], -3, 0)

-- Running Total
Running_Sum =
RUNNING_SUM(SUM([Sales]))
```

> **Ask AI (Claude or ChatGPT)**
>
> "Write a Tableau calculated field that [describe what you need, e.g. 'calculates the 3-month moving average of profit, expressed as a percentage of the category total']. My data has these fields: [list your field names]."

<details>
<summary><strong>Advanced: Statistical Features</strong> (skip on first read)</summary>

These features require Tableau's Analytics model and are typically covered after you're comfortable with basic charts and dashboards.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Forecasting
FORECAST_INDICATOR(
    SUM([Sales]), 6, 'manual',
    0.95, 'multiplicative'
)

-- Clustering
KMEANS(
    [Dimension1], [Dimension2],
    3, 'euclidean'
)

-- Statistical Testing
T_TEST(
    [Group1], [Group2],
    'two-tail', 0.95
)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Forecasting</span>
    </div>
    <div class="code-callout__body">
      <p><code>FORECAST_INDICATOR</code> with <code>'multiplicative'</code> seasonality and 95% confidence interval forecasts 6 periods ahead from the aggregated sales measure.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">K-Means Clustering</span>
    </div>
    <div class="code-callout__body">
      <p><code>KMEANS</code> partitions records into 3 clusters using Euclidean distance on two dimensions, requires Tableau's Analytics model.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-17" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Statistical Test</span>
    </div>
    <div class="code-callout__body">
      <p><code>T_TEST</code> runs a two-tailed t-test between two groups at the 95% confidence level to test for significant differences in means.</p>
    </div>
  </div>
</aside>
</div>

</details>

### 4. Dashboard Design

#### Layout Patterns

Two layouts cover most dashboards you will build:

**1. Executive Dashboard**, for stakeholders who need quick answers at a glance

- **Top row:** 3-4 KPI tiles showing single numbers with a trend indicator (e.g. Total Sales: $1.2 M ▲ 8%)
- **Middle:** one large primary chart, a bar, map, or trend line that answers the main question
- **Bottom row:** supporting detail tables or secondary breakdowns that explain the primary chart

Use this when your audience opens the dashboard for 30 seconds and needs to leave with one takeaway.

**2. Analysis Dashboard**, for analysts who need to explore and drill down

- **Left sidebar:** dimension filters and quick-filter dropdowns (Region, Date range, Category)
- **Top-right:** overview chart showing the big picture
- **Bottom-right:** drill-down chart that responds to selections in the overview, clicking a region filters the detail below

Use this when your audience wants to ask follow-up questions: "OK, the West region is down, but which sub-category?"

> **Ask AI (Claude or ChatGPT)**
>
> "I'm designing a Tableau dashboard for [describe your audience and goal, e.g. 'a weekly sales review for a regional manager who wants to spot underperforming territories quickly']. Suggest a layout: which KPIs to show at the top, what the main chart should be, and what filters to include."

#### Interactive Elements

| Element | What it does | Example |
|---------|-------------|---------|
| Filter | Limits which rows show | Show only 2023 orders |
| Slicer / Quick Filter | Interactive dropdown or list | User picks a region |
| Action | Links charts together | Clicking a bar filters a map |
| Parameter | User-controlled input value | Top N selector, date range |

## What Your First Session Looks Like

Here's a realistic picture of your first 35 minutes with Tableau and Sample Superstore. You open the app, connect to the data source, and immediately see a table of fields on the left. You drag `Order Date` to Columns and `Sales` to Rows, Tableau aggregates the data and draws a line chart by year without you asking. You right-click the date and break it down by month; the chart fills in across the timeline. You drag `Category` to the Color mark and suddenly three lines appear, one per product category, color-coded automatically. You notice the Technology line dips in February every year and wonder why. You add a Region filter to narrow it down. Ten minutes later you drag everything onto a new Dashboard sheet, resize the panels, and share the URL. It is rough, but it answers a real question. That moment, going from raw rows to a visual insight, is what the rest of this submodule helps you do faster and better.

## Learning Path

This submodule runs over **two sessions**.

### Session 1: Tableau foundations

- Read [Tableau basics](tableau-basics.md), interface, connecting Sample Superstore, first charts, calculated fields, LOD
- Read [Advanced analytics](advanced-analytics.md), table calculations, LOD types, combo charts, parameters, actions
- Explore the [AI tools integration](ai-tools-integration.md) guide to use Claude or ChatGPT as a pair partner while building

### Session 2: Case studies and comparison

- Work through the [Tableau case study](tableau-case-study.md), end-to-end Superstore dashboard
- Compare tools: [Looker Studio case study](looker-studio-case-study.md) and [Power BI case study](powerbi-case-study.md)
- Complete the [module assignment](../assignments/module-assignment.md)

## Using AI Tools with Tableau

See [ai-tools-integration.md](ai-tools-integration.md) for a practical guide on using Claude Desktop or ChatGPT as an AI pair partner: writing calculated fields, debugging LOD expressions, generating sample data, and preparing data in Python before connecting to Tableau.

## Assignment

Use the [module assignment](../assignments/module-assignment.md) when you are ready for the graded work.

## Resources

### Documentation

* [Tableau Help](https://help.tableau.com)
* [Knowledge Base](https://kb.tableau.com)
* [Community Forums](https://community.tableau.com)
* [Video Library](https://www.tableau.com/learn/training)

### Learning Materials

* [Sample Workbooks](https://public.tableau.com/gallery)
* [Best Practices](https://www.tableau.com/learn/whitepapers)
* [Tips & Tricks](https://www.tableau.com/learn/tutorials)
* [Blog](https://www.tableau.com/blog)

Remember: The key to mastering Tableau is practice and experimentation. Start with simple visualizations, then gradually add complexity as you become more comfortable with the tool.
