# BI Tool Screenshots

This folder contains placeholder images for Business Intelligence tool screenshots.

## Status

These are **placeholder images** that need to be replaced with actual screenshots from:
- Tableau Desktop
- Power BI Desktop
- Looker Studio (Google Data Studio)

## How to Capture Screenshots

### General Guidelines

1. **Resolution**: Use 1920x1080 or higher resolution
2. **Format**: PNG format, 150+ DPI
3. **Naming**: Keep the existing filename convention
4. **Size**: Aim for 10-50 KB per image after optimization

### Tableau Screenshots

1. Open Tableau Desktop
2. Load "Sample - Superstore" dataset
3. Navigate to the feature being documented
4. Use Windows Snipping Tool or macOS Screenshot (Cmd+Shift+4)
5. Crop to show relevant UI elements
6. Save as PNG with same filename

### Power BI Screenshots

1. Open Power BI Desktop
2. Import Sample Superstore data (available as Excel from Tableau)
3. Create visualizations as described
4. Capture with annotation highlights where helpful

### Looker Studio Screenshots

1. Go to lookerstudio.google.com
2. Create a new report or use a template
3. Connect to a sample dataset
4. Capture the relevant interface elements

## Screenshot Checklist

### Tableau (33 screenshots needed)
- [ ] tableau_start_page.png - Start page with connections
- [ ] tableau_workspace.png - Main workspace layout
- [ ] tableau_data_source.png - Data source tab
- [ ] tableau_blank_worksheet.png - Empty worksheet
- [ ] tableau_drag_drop.png - Dragging fields
- [ ] tableau_chart_formatting.png - Format options
- [ ] tableau_format_pane.png - Format pane
- [ ] tableau_filter_shelf.png - Filters applied
- [ ] tableau_parameter.png - Parameter dialog
- [ ] tableau_dashboard_actions.png - Actions config
- [ ] tableau_dashboard_workspace.png - Dashboard layout
- [ ] tableau_dashboard_interactivity.png - Interactive filters
- [ ] tableau_final_dashboard.png - Complete dashboard
- [ ] tableau_line_chart.png - Time series chart
- [ ] tableau_map.png - Geographic map
- [ ] tableau_bar_chart.png - Bar chart
- [ ] tableau_heatmap.png - Heat map
- [ ] tableau_scatter.png - Scatter plot
- [ ] tableau_calculated_field.png - Calculation editor
- [ ] tableau_table_calc.png - Table calculation
- [ ] tableau_lod.png - LOD expression
- [ ] tableau_prep.png - Tableau Prep
- [ ] tableau_flow.png - Data flow
- [ ] tableau_custom_viz.png - Custom visualization
- [ ] tableau_advanced_map.png - Advanced mapping
- [ ] tableau_content_mgmt.png - Server content
- [ ] tableau_collaboration.png - Sharing features
- [ ] tableau_extract.png - Extract settings
- [ ] tableau_performance.png - Performance tools
- [ ] tableau_security.png - Security config
- [ ] tableau_governance.png - Governance
- [ ] tableau_mobile.png - Mobile layout
- [ ] tableau_mobile_app.png - Mobile app

### Power BI (20 screenshots needed)
- [ ] powerbi_start_page.png - Home screen
- [ ] powerbi_workspace.png - Report view
- [ ] powerbi_data_model.png - Model view
- [ ] powerbi_first_chart.png - First visualization
- [ ] powerbi_line_chart.png - Line chart
- [ ] powerbi_map.png - Map visual
- [ ] powerbi_dashboard.png - Dashboard
- [ ] powerbi_dax.png - DAX formula
- [ ] powerbi_parameter.png - What-if parameter
- [ ] powerbi_publish.png - Publish dialog
- [ ] powerbi_query.png - Power Query
- [ ] powerbi_modeling.png - Relationships
- [ ] powerbi_custom_visuals.png - AppSource
- [ ] powerbi_advanced_charts.png - Advanced visuals
- [ ] powerbi_workspace_mgmt.png - Workspace
- [ ] powerbi_collaboration.png - Sharing
- [ ] powerbi_performance.png - Performance analyzer
- [ ] powerbi_refresh.png - Refresh settings
- [ ] powerbi_security.png - RLS
- [ ] powerbi_governance.png - Lineage

### Looker Studio (18 screenshots needed)
- [ ] looker_start_page.png - Start page
- [ ] looker_workspace.png - Report canvas
- [ ] looker_data_source.png - Data connection
- [ ] looker_first_chart.png - First chart
- [ ] looker_timeseries.png - Time series
- [ ] looker_map.png - Geo chart
- [ ] looker_dashboard.png - Dashboard layout
- [ ] looker_calculations.png - Calculated fields
- [ ] looker_controls.png - Filter controls
- [ ] looker_blending.png - Data blending
- [ ] looker_source_mgmt.png - Source management
- [ ] looker_advanced_charts.png - Advanced charts
- [ ] looker_interactivity.png - Interactions
- [ ] looker_collaboration.png - Collaboration
- [ ] looker_sharing.png - Share options
- [ ] looker_performance.png - Performance
- [ ] looker_optimization.png - Optimization
- [ ] looker_ai.png - AI features

## Regenerating Placeholders

If you need to regenerate placeholder images:

```bash
cd assets
python generate_placeholders.py
```

## Image Optimization

After capturing screenshots, optimize them:

```bash
# Using ImageOptim (macOS)
imageoptim *.png

# Using pngquant
pngquant --quality=65-80 *.png
```
