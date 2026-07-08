# Quiz: Business Intelligence with Tableau

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

Try each question closed-book first. Click **Show hint** if you get stuck, hints point you at the relevant lesson section and how to think about the question, without naming the answer.

## Questions

1. In Tableau's Data pane, what is the visual difference that distinguishes a Dimension from a Measure?

- [ ] Dimensions appear in bold; Measures appear in italics
- [ ] Dimensions are shown in blue; Measures are shown in green
- [ ] Dimensions are on the left panel; Measures are on the right panel
- [ ] Dimensions are uppercase; Measures are lowercase

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Beginner's Mental Model" table.
- **Think:** The lesson maps each core concept to a color. Recall the two pill colors Tableau uses and which type each represents.

</details>

2. You drag `Order Date` to the Columns shelf and see only four data points on your line chart. What most likely caused this?

- [ ] The data source is not connected properly
- [ ] Tableau defaulted to aggregating by YEAR instead of month
- [ ] The filter shelf is hiding most of the data
- [ ] The Orders table is missing rows

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Common Mistakes, Date fields collapse to year only."
- **Think:** Tableau makes a default choice when a date field is dropped onto a shelf. The lesson's Common Mistakes section names exactly this behavior and the fix.

</details>

3. You drag `Postal Code` onto the view and it appears as a green Measure showing a sum. What is the correct fix?

- [ ] Delete the field and re-create it as a string
- [ ] Change the aggregation from SUM to COUNT
- [ ] Convert the field to a Dimension by dragging it or right-clicking → Convert to Dimension
- [ ] Apply a filter to remove numeric-looking postal codes

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Common Mistakes, Postal Code and Order ID end up as green Measures."
- **Think:** The issue is classification, not aggregation. Which panel does the field need to live in, and how do you move it there?

</details>

4. Which Tableau connection type takes a local snapshot of your data and allows offline access?

- [ ] Live connection
- [ ] Federated connection
- [ ] Extract
- [ ] Web data connector

<details>
<summary>Show hint</summary>

- **Where:** [README](../README.md), "Core Concepts, Connection Types" table.
- **Think:** The table contrasts two main types. One queries the source on every interaction; the other stores a local file. Which one is the snapshot?

</details>

5. A colleague writes the calculated field `[Profit] / [Sales]` to compute profit margin. What is the problem with this formula?

- [ ] Division is not supported in Tableau calculated fields
- [ ] The formula computes a row-level ratio and then averages those ratios, which is usually wrong
- [ ] Profit and Sales must be on the same axis before you can divide them
- [ ] The formula only works if the data source is an Extract

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Common Mistakes, Calculated field gives wrong profit margin."
- **Think:** The lesson explicitly contrasts this formula with the correct version. Focus on when aggregation happens relative to when the division happens.

</details>

6. You add a Region filter to a dashboard, but it only filters one of the three charts on the page. What setting do you need to change?

- [ ] Set the filter type from discrete to continuous
- [ ] Rebuild the other charts so they use the same data source
- [ ] Right-click the filter pill and choose Apply to Worksheets → All Using This Data Source
- [ ] Enable LOD expressions on the filter field

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Common Mistakes, Filters on a dashboard don't apply to all sheets."
- **Think:** The lesson describes the exact right-click menu path that fixes this. Recall the two-step action the lesson recommends.

</details>

7. Which shelf in Tableau places a field on the Y-axis of a chart?

- [ ] Columns shelf
- [ ] Rows shelf
- [ ] Marks card
- [ ] Pages shelf

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Beginner's Mental Model" table, "Shelf" row.
- **Think:** The lesson defines Rows and Columns as axis mappings. Which axis is vertical?

</details>

8. A `FIXED` LOD expression calculates based on the dimension you specify. Which filter type must you promote for a dimension filter to affect a `FIXED` LOD result?

- [ ] Quick filter
- [ ] Extract filter
- [ ] Context filter
- [ ] Table calculation filter

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Gotchas, LOD expressions are filtered by context filters, not regular filters."
- **Think:** The Gotchas section names the specific filter level that overrides a `FIXED` LOD. There is only one correct term.

</details>

9. For a time-based trend visualization in Tableau, which chart type does the lesson recommend?

- [ ] Bar chart
- [ ] Pie chart
- [ ] Line chart
- [ ] Scatter plot

<details>
<summary>Show hint</summary>

- **Where:** [README](../README.md), "Core Concepts, Basic Charts, Chart Selection" table, Comparison row.
- **Think:** The table maps each visualization goal to chart types. "Trends over time" maps directly to one type.

</details>

10. You publish a workbook to Tableau Public. What security risk does the lesson warn about?

- [ ] Other users can delete your workbook
- [ ] Tableau Public compresses your data and loses precision
- [ ] Anyone who views the workbook can download all rows in the underlying extract
- [ ] Calculated fields are stripped from published workbooks

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Gotchas, Published workbooks to Tableau Public expose all data in the extract."
- **Think:** The Gotcha specifically addresses what is downloadable by anyone, not just the author. Think about data confidentiality.

</details>

11. Which of the following best describes the Marks card in Tableau?

- [ ] A shelf that defines the aggregation method for all measures
- [ ] A panel that controls visual properties such as color, size, shape, and label for marks on the view
- [ ] The toolbar button used to export the chart as an image
- [ ] The area where you type calculated field formulas

<details>
<summary>Show hint</summary>

- **Where:** [Tableau basics](../tableau-basics.md), "Beginner's Mental Model" table, "Marks Card" row.
- **Think:** The table gives a plain-English "What it is" for each concept. Find the Marks Card entry and match it to the options above.

</details>

12. You want to compare actual sales to a budget target on the same bar chart. Which Tableau chart type does the lesson recommend for "showing progress toward a goal"?

- [ ] Pie chart
- [ ] Scatter plot
- [ ] Bullet chart
- [ ] Heatmap

<details>
<summary>Show hint</summary>

- **Where:** [README](../README.md), "Core Concepts, Basic Charts, Chart Selection" table, Comparison row.
- **Think:** The table lists three chart types for the Comparison goal. One is specifically described as being for "targets." Match the goal to the chart type.

</details>
