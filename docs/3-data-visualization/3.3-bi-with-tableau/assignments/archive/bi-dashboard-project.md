# Build Your First BI Dashboard

**Time estimate:** 2–3 hours  
**Tool:** Your choice — Tableau Public, Power BI Desktop/Service, or Looker Studio

---

## The Scenario

You've just joined a retail company as a junior analyst. Your manager walks over on Monday morning with three questions:

> "Before our team meeting on Friday, can you put together something visual that shows me where our profit is coming from, whether sales are trending up, and which parts of the country we need to worry about?"

You have access to the **Superstore dataset** — the same sample data used in this submodule. Your job is to build a dashboard that answers all three questions in one place, with filters so your manager can explore the data herself.

---

## The 3 Business Questions

Your dashboard must answer all three:

1. **Profitability by category** — Which product categories are most profitable? Are any categories actually losing the company money?
2. **Sales trend over time** — How have sales changed over the past 4 years? Are there seasonal patterns (e.g. a holiday spike every Q4)?
3. **Regional performance** — Which regions or states are underperforming and may need attention?

---

## Deliverables

Submit all four of the following:

1. **A dashboard with at least 3 charts** — one chart per business question above. They should all live on the same dashboard view, not on separate pages.
2. **At least one interactive filter or slicer** — something your manager can click or drag to explore the data (e.g. filter by year, region, or category).
3. **A short insight summary** — 3 to 5 bullet points written in plain English explaining what you found. Imagine you're sending this to your manager in a Slack message alongside the dashboard link. Example format:
   - "Technology is our most profitable category, but Furniture — despite high sales — actually loses money in several states."
   - "Sales peak in November and December every year, with Q1 consistently being the slowest quarter."
4. **A published/shareable link** — Tableau Public URL, Power BI share link, or Looker Studio share link. Make sure the link is publicly accessible (not just visible to you).

---

## Evaluation Rubric

| Criterion | 1 — Needs work | 2 — Meets expectations | 3 — Exceeds expectations |
|---|---|---|---|
| **Completeness** | Missing charts or deliverables | All 3 questions answered, insight summary present, link works | All deliverables polished and clearly organised |
| **Insight quality** | Summary is vague or just restates what the chart shows | Summary draws a clear, specific conclusion from the data | Summary includes a recommendation or follow-up question for the manager |
| **Visual clarity** | Charts are hard to read, unlabelled, or cluttered | Charts have clear titles, axis labels, and an appropriate chart type for the data | Thoughtful use of color, consistent formatting, layout guides the reader's eye |
| **Interactivity** | No filters present | At least one working filter or slicer | Filter applies across multiple charts simultaneously (cross-filtering / dashboard action) |

---

## Starter Tips

Don't know where to begin? Start here.

- **Tableau Public** — Download [sample_superstore.xls](../assets/sample_superstore.xls), then on the start screen click **Microsoft Excel** under "To a File" and open it.
- **Power BI Desktop** — Use the same [sample_superstore.xls](../assets/sample_superstore.xls) file. In Power BI, use "Get Data → Excel" to connect.
- **Looker Studio** — Upload the Superstore CSV to Google Sheets first, then connect Looker Studio to it via "Google Sheets" as a data source. It takes about 2 minutes.
- **Stuck on chart type?** Go back to the [visualization principles lesson](../../3.1-intro-data-viz/visualization-principles.md): bar charts for category comparison, line charts for trends over time, maps or bar charts for regional comparison.
- **Stuck on a calculated field?** Ask an AI assistant — paste in your question and list your field names. For example: "Write a Tableau calculated field that flags any row where Profit is negative. My fields are `Profit` and `Sales`."

---

## Stretch Goals

Finished early? Try one or more of these:

- **Add a calculated field** — Create a "Profit Margin %" field (`SUM([Profit]) / SUM([Sales])`) and use it to colour your category chart from red (negative margin) to green (positive margin).
- **Add a parameter control** — Build a "Top N States by Sales" parameter that lets your manager choose to see the top 5, 10, or 15 states dynamically.
- **Tell a story** — In Tableau, use the "Story" tab (or equivalent in your tool) to walk through your three findings in sequence, adding a text annotation to each slide that explains the insight.

---

## Submission

Post your shareable link and insight summary in the course submission channel or share it with your instructor. Make sure the link works in an incognito / private browser window before you submit.

Good luck — this is the kind of thing you'll do in your first week at a real analytics job.
