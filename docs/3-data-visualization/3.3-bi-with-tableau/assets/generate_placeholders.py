#!/usr/bin/env python3
"""
Generate placeholder images for BI tool screenshots.
These serve as visual indicators until real screenshots are captured.

Run: python generate_placeholders.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Screenshot definitions organized by tool
SCREENSHOTS = {
    'tableau': [
        ('tableau_start_page', 'Tableau Start Page', 'Show connection options and Sample Superstore'),
        ('tableau_workspace', 'Tableau Workspace', 'Show Data Pane, Shelves, Canvas, Show Me panel'),
        ('tableau_data_source', 'Data Source Tab', 'Show connected tables and relationships'),
        ('tableau_blank_worksheet', 'Blank Worksheet', 'Empty worksheet ready for visualization'),
        ('tableau_drag_drop', 'Drag and Drop Fields', 'Show dragging dimension to Rows shelf'),
        ('tableau_chart_formatting', 'Chart Formatting', 'Show format options and color selection'),
        ('tableau_format_pane', 'Format Pane', 'Show right-side format panel options'),
        ('tableau_filter_shelf', 'Filter Shelf', 'Show filter applied to visualization'),
        ('tableau_parameter', 'Parameter Creation', 'Show parameter dialog box'),
        ('tableau_dashboard_actions', 'Dashboard Actions', 'Show action configuration dialog'),
        ('tableau_dashboard_workspace', 'Dashboard Workspace', 'Show dashboard layout with sheets'),
        ('tableau_dashboard_interactivity', 'Dashboard Interactivity', 'Show filter interaction'),
        ('tableau_final_dashboard', 'Final Dashboard', 'Complete sales analysis dashboard'),
        ('tableau_line_chart', 'Line Chart', 'Time series with Sales and Profit'),
        ('tableau_map', 'Map View', 'Geographic visualization of sales'),
        ('tableau_bar_chart', 'Bar Chart', 'Category comparison bar chart'),
        ('tableau_heatmap', 'Heat Map', 'Matrix visualization with color encoding'),
        ('tableau_scatter', 'Scatter Plot', 'Profit vs Sales scatter plot'),
        ('tableau_calculated_field', 'Calculated Field', 'Show calculation editor'),
        ('tableau_table_calc', 'Table Calculation', 'Running total or percent difference'),
        ('tableau_lod', 'LOD Expression', 'Level of Detail calculation'),
        ('tableau_prep', 'Tableau Prep', 'Data preparation flow interface'),
        ('tableau_flow', 'Flow Management', 'Show data transformation steps'),
        ('tableau_custom_viz', 'Custom Visualization', 'Advanced chart type example'),
        ('tableau_advanced_map', 'Advanced Mapping', 'Multi-layer map with tooltips'),
        ('tableau_content_mgmt', 'Content Management', 'Server content organization'),
        ('tableau_collaboration', 'Collaboration Features', 'Sharing and comments'),
        ('tableau_extract', 'Extract Configuration', 'Data extract settings'),
        ('tableau_performance', 'Performance Tools', 'Performance recorder'),
        ('tableau_security', 'Security Configuration', 'Permissions and RLS'),
        ('tableau_governance', 'Governance Tools', 'Data governance features'),
        ('tableau_mobile', 'Mobile Features', 'Mobile layout design'),
        ('tableau_mobile_app', 'Mobile App', 'Tableau Mobile interface'),
    ],
    'powerbi': [
        ('powerbi_start_page', 'Power BI Start Page', 'Home screen with recent files'),
        ('powerbi_workspace', 'Power BI Workspace', 'Report view with visualizations pane'),
        ('powerbi_data_model', 'Data Model View', 'Table relationships diagram'),
        ('powerbi_first_chart', 'First Chart Creation', 'Creating a bar chart'),
        ('powerbi_line_chart', 'Line Chart Setup', 'Time series visualization'),
        ('powerbi_map', 'Map Creation', 'Geographic visualization'),
        ('powerbi_dashboard', 'Dashboard Building', 'Pinning visuals to dashboard'),
        ('powerbi_dax', 'DAX Measure', 'DAX formula editor'),
        ('powerbi_parameter', 'Parameter Creation', 'What-if parameter'),
        ('powerbi_publish', 'Publish Options', 'Publishing to Power BI Service'),
        ('powerbi_query', 'Power Query', 'Query Editor interface'),
        ('powerbi_modeling', 'Data Modeling', 'Relationship view'),
        ('powerbi_custom_visuals', 'Custom Visuals', 'AppSource marketplace'),
        ('powerbi_advanced_charts', 'Advanced Charts', 'Complex visualization'),
        ('powerbi_workspace_mgmt', 'Workspace Management', 'Service workspace'),
        ('powerbi_collaboration', 'Collaboration', 'Sharing and permissions'),
        ('powerbi_performance', 'Performance', 'Analyzer tool'),
        ('powerbi_refresh', 'Refresh Config', 'Scheduled refresh settings'),
        ('powerbi_security', 'Row Level Security', 'RLS configuration'),
        ('powerbi_governance', 'Governance', 'Data lineage view'),
    ],
    'looker': [
        ('looker_start_page', 'Looker Studio Start', 'Template gallery and recent reports'),
        ('looker_workspace', 'Looker Workspace', 'Report canvas with panels'),
        ('looker_data_source', 'Data Source Config', 'Connection settings'),
        ('looker_first_chart', 'First Chart', 'Basic chart creation'),
        ('looker_timeseries', 'Time Series', 'Date-based visualization'),
        ('looker_map', 'Map Creation', 'Geo chart'),
        ('looker_dashboard', 'Dashboard Building', 'Multi-chart layout'),
        ('looker_calculations', 'Calculated Fields', 'Field formula editor'),
        ('looker_controls', 'Interactive Controls', 'Date range and filter controls'),
        ('looker_blending', 'Data Blending', 'Multiple data source blend'),
        ('looker_source_mgmt', 'Source Management', 'Data source list'),
        ('looker_advanced_charts', 'Advanced Charts', 'Combo and specialized charts'),
        ('looker_interactivity', 'Interactive Features', 'Drill-through configuration'),
        ('looker_collaboration', 'Collaboration', 'Share and schedule options'),
        ('looker_sharing', 'Sharing Options', 'Export and embed'),
        ('looker_performance', 'Performance', 'Report optimization'),
        ('looker_optimization', 'Optimization', 'Loading and caching settings'),
        ('looker_ai', 'AI Features', 'Insight suggestions'),
    ],
}

def create_placeholder_image(filename, title, description, tool_color):
    """Create a placeholder image with tool-specific styling."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Background
    ax.set_facecolor('#f8f9fa')

    # Tool-specific header bar
    header = patches.Rectangle((0, 0.85), 1, 0.15, transform=ax.transAxes,
                                 facecolor=tool_color, edgecolor='none')
    ax.add_patch(header)

    # Title in header
    ax.text(0.5, 0.92, title, transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='white',
            ha='center', va='center')

    # Placeholder content area
    content_bg = patches.Rectangle((0.05, 0.15), 0.9, 0.65, transform=ax.transAxes,
                                     facecolor='white', edgecolor='#dee2e6', linewidth=2)
    ax.add_patch(content_bg)

    # Screenshot icon (camera)
    ax.text(0.5, 0.55, '📸', transform=ax.transAxes,
            fontsize=48, ha='center', va='center')

    # Placeholder text
    ax.text(0.5, 0.40, 'Screenshot Placeholder', transform=ax.transAxes,
            fontsize=14, color='#6c757d', ha='center', va='center')

    # Description
    ax.text(0.5, 0.30, description, transform=ax.transAxes,
            fontsize=11, color='#495057', ha='center', va='center',
            style='italic', wrap=True)

    # Instruction
    ax.text(0.5, 0.08, 'Replace with actual screenshot from the application',
            transform=ax.transAxes, fontsize=9, color='#adb5bd',
            ha='center', va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig(f'{filename}.png', dpi=120, bbox_inches='tight',
                facecolor='#f8f9fa', edgecolor='none')
    plt.close()

# Tool colors
TOOL_COLORS = {
    'tableau': '#E97627',  # Tableau orange
    'powerbi': '#F2C811',  # Power BI yellow
    'looker': '#4285F4',   # Google blue
}

print("Generating BI tool placeholder images...")

for tool, screenshots in SCREENSHOTS.items():
    color = TOOL_COLORS[tool]
    print(f"\n{tool.upper()} ({len(screenshots)} images):")
    for filename, title, desc in screenshots:
        create_placeholder_image(filename, title, desc, color)
        print(f"  - {filename}.png")

print(f"\n{'='*50}")
print(f"Generated {sum(len(s) for s in SCREENSHOTS.values())} placeholder images")
print("="*50)
