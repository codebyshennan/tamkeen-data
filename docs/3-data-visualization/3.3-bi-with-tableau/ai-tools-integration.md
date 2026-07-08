# Using AI Tools with Tableau

**After this lesson:** you can use Claude or ChatGPT as an AI pair partner while building Tableau dashboards, getting concepts explained, writing calculated fields faster, debugging LOD expressions, and getting chart recommendations without leaving your workflow.

This course uses **Tableau Public**, which is free and requires no login. The setup for AI assistance is equally simple: open a chat tab next to Tableau and start asking questions.

---

## Quick start (no setup needed)

Open [claude.ai](https://claude.ai) or [chatgpt.com](https://chatgpt.com) in a browser tab alongside Tableau. That's the entire setup. Paste your question, field name, or error message into the chat and get an answer back.

> **One habit that makes a big difference:** always paste your exact field names from the Data pane. AI gives much better output when it knows whether your field is called `[Revenue]` or `[Sales]`.

---

## Key concepts to ask AI about

If you're new to Tableau, these are the five concepts that trip people up most. Ask Claude or ChatGPT to explain any of them in plain language, or use the ready-made prompts below each one.

### Dimensions vs. Measures

Blue pills in the Data pane are **dimensions** (categories you group by: `Region`, `Category`, `Customer Name`). Green pills are **measures** (numbers you aggregate: `Sales`, `Profit`, `Quantity`). Getting this wrong is the most common reason a chart looks broken.

> **Ask AI:** "In Tableau, what's the difference between a dimension and a measure? Give me an example of each using a retail sales dataset."

### The Marks card

The **Marks card** controls everything visual about your chart that isn't the X/Y axes: color, size, shape, label, and tooltip. Dragging a field onto Color is how you split one line into multiple lines, or one bar into a stacked bar.

> **Ask AI:** "In Tableau, I dragged a field to the Marks card Color slot but nothing changed. What could cause this? My data has Sales by Region and Category."

### Calculated fields

A **calculated field** is a new column you create inside Tableau using a formula, like a spreadsheet formula, but it works across aggregated data. You create one via **Analysis → Create Calculated Field**.

> **Ask AI:** "I want to create a Tableau calculated field that divides Profit by Sales to get a margin percentage. My fields are called [Profit] and [Sales]. Write the formula and explain what each part does."

### Filters and filter order

Tableau applies filters in a specific order: **Extract filters → Data source filters → Context filters → Dimension filters → Measure filters**. If a filter seems to be doing nothing, it may be in the wrong position in this order.

> **Ask AI:** "I added a Region filter in Tableau but my chart totals didn't change. Explain why filters might not work and what I should check first."

### Aggregation (SUM vs AVG vs COUNT)

When you drag a measure to a shelf, Tableau automatically aggregates it, usually as SUM. Right-click the pill to change it to AVG, COUNT, MIN, MAX, etc. Forgetting to check this is why totals often look far too large.

> **Ask AI:** "In Tableau, my Sales total looks 100× too large. Could aggregation be the reason? How do I check and fix it?"

---

## Practical workflows

Use these prompts whenever you hit a specific problem. Copy the prompt template, fill in your actual field names, and paste it into Claude or ChatGPT.

### Workflow 1: Write a calculated field

Describe what you need and paste the result into **Analysis → Create Calculated Field**.

**Prompt template:**

```
I'm working in Tableau with a table that has these fields: [list your fields].
Write a calculated field that [describe what you want it to do].
Show me the formula and explain each part.
```

**Example:**

```
I'm working in Tableau with an Orders table that has Sales, Profit, and Discount fields.
Write a calculated field that returns the net margin as a percentage, formatted for display.
Show me the formula and explain each part.
```

**Claude's output** (paste directly into the calculated field editor):

```
SUM([Profit]) / SUM([Sales])
```

Format the field as **Percentage** via right-click → Default Properties → Number Format. For a label-ready string:

```
STR(ROUND(SUM([Profit]) / SUM([Sales]) * 100, 1)) + "%"
```

---

### Workflow 2: Debug an LOD expression

LOD (Level of Detail) expressions compute values at a different granularity than the current view, useful but error-prone. Paste the broken expression and describe what it should do.

**Prompt template:**

```
This Tableau LOD expression is giving me wrong results: [paste expression]
What I want it to do: [describe expected behaviour]
My current view has these dimensions: [list dimensions]
What's wrong and how do I fix it?
```

**Example:**

```
This Tableau LOD expression returns wrong numbers when I add a Region filter:
{FIXED [Customer ID] : MIN([Order Date])}
I want it to show each customer's first order date, filtered by the current Region selection.
```

Claude will explain why `FIXED` ignores dimension filters and show you the fix, either adding a context filter or switching to `INCLUDE`:

```
{INCLUDE [Region] : MIN([Order Date])}
```

---

### Workflow 3: Generate a sample dataset

Use Claude to create practice data when you don't have a real dataset yet.

**Example prompt:**

```
Generate a CSV with 50 rows of fictional retail sales data suitable for Tableau practice.
Include: Order ID, Order Date (2023-2024), Region (East/West/North/South),
Category (Furniture/Technology/Office Supplies), Sales (numeric), Profit (numeric).
```

Save the output as `sample_sales.csv` and connect via **Text File** in Tableau's Connect pane.

---

### Workflow 4: Prepare data with Python before connecting

Tableau works best with clean, wide-format data. Ask Claude to write a pandas script to reshape or clean your source file before you connect.

**Example prompt:**

```
I have a CSV with columns: date, product, region, revenue.
Some revenue values are empty strings instead of NaN.
Write a pandas script that cleans the data and pivots it so each region
becomes its own column, ready to connect in Tableau.
```

Run the script, save the output, then connect Tableau to the cleaned file.

---

### Workflow 5: Choose the right chart type

Describe your data and question; AI recommends the chart type and shelf configuration.

**Example prompt:**

```
I want to show how profit margin has changed month-over-month for three product categories.
What chart type should I use in Tableau, and which fields go on Rows, Columns, and Marks?
```

**Sample AI response:**

> **Recommended chart type:** Multi-line chart (one line per category)
>
> **Shelf configuration:**
> - **Columns:** `Order Date` (set to continuous Month/Year)
> - **Rows:** Profit margin calculated field, `SUM([Profit]) / SUM([Sales])`
> - **Color:** `Category` (one line per category)
> - **Marks:** Line
>
> **To set it up:**
> 1. Create the calculated field `SUM([Profit]) / SUM([Sales])` and format it as Percentage.
> 2. Drag `Order Date` to Columns, right-click → **Month** (continuous, not discrete).
> 3. Drag the calculated field to Rows.
> 4. Drag `Category` to the Color mark card.
>
> **Verify:** You should see three coloured lines trending over time. If all three overlap exactly, check that `Category` landed on Color (not Rows or Columns).

---

## Debugging checklist

Before assuming AI gave you a bad formula, run through these:

- [ ] **Copied surrounding explanation text** → paste only the formula line(s) into the calculated field editor; strip plain-English sentences
- [ ] **Field names don't match your Data pane** → your prompt said `[Revenue]` but your field is `[Sales]`; re-prompt with exact names or find-and-replace before pasting
- [ ] **Result looks wrong on the view** → the formula may be correct but aggregate differently with multiple dimensions; drag the field onto a fresh sheet with one dimension and compare against a known total
- [ ] **`FIXED` LOD ignores your filter** → `FIXED` bypasses dimension filters by design; right-click the filter pill → **Add to Context**, then re-test
- [ ] **Claude is unavailable** → the [Tableau calculated fields reference](https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_create.htm) covers all functions offline

---

<details>
<summary><strong>Advanced: Claude Desktop + Tableau MCP (workplace / Tableau Server users only)</strong></summary>

> **This does not apply to Tableau Public.** Tableau Public does not expose the REST API required for MCP. If you're using Tableau Public in this course, use the copy-paste workflow above.

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) lets Claude Desktop connect directly to Tableau Server or Tableau Cloud workbooks and data sources, no copy-pasting field lists required. This is useful when you're working iteratively on a large workbook with many fields.

### Prerequisites

- Claude Desktop installed ([download](https://claude.ai/download))
- Access to **Tableau Server** or **Tableau Cloud** (not Tableau Public)
- Node.js v18+ installed ([nodejs.org](https://nodejs.org))

### Step 1: Create a Personal Access Token in Tableau

1. Sign in to Tableau Server or Tableau Cloud.
2. Click your profile icon → **My Account Settings**.
3. Scroll to **Personal Access Tokens** → **Create new token**.
4. Name it `claude-mcp` and copy the token secret immediately, you won't see it again.

### Step 2: Configure Claude Desktop

Open the Claude Desktop config file on macOS:

```bash
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

Add the Tableau block, replacing the placeholder values:

```json
{
  "mcpServers": {
    "tableau": {
      "command": "pnpm exec",
      "args": ["-y", "@salesforce/mcp-server-tableau"],
      "env": {
        "TABLEAU_SERVER_URL": "https://your-tableau-server.com",
        "TABLEAU_SITE_ID": "",
        "TABLEAU_TOKEN_NAME": "claude-mcp",
        "TABLEAU_TOKEN_VALUE": "your-token-secret-here"
      }
    }
  }
}
```

`TABLEAU_SITE_ID` is the value after `/site/` in your Tableau URL, leave blank for the default site.

### Step 3: Verify

Restart Claude Desktop. You should see a hammer icon (🔨) in the chat bar. Test with:

```
List my Tableau workbooks.
```

> **Note:** Tableau's MCP tooling is actively developed. If the package name above is not found, check the current name at the [Tableau Developer docs](https://developer.salesforce.com/docs/tableau).

</details>

---

## Resources

- [Tableau calculated fields reference](https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_create.htm)
- [Tableau LOD expressions](https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_lod.htm)
- [Claude Desktop download](https://claude.ai/download)
