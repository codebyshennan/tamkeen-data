#!/usr/bin/env python3
"""
Update markdown files to replace [Screenshot: ...] placeholders with image references.
"""

import re
import os

# Mapping from screenshot text to image file
SCREENSHOT_MAPPING = {
    # Tableau
    'Tableau Start Page': 'tableau_start_page',
    'Tableau Workspace': 'tableau_workspace',
    'Data Source Tab': 'tableau_data_source',
    'Data Source Page': 'tableau_data_source',
    'Blank Worksheet': 'tableau_blank_worksheet',
    'Drag and Drop Fields': 'tableau_drag_drop',
    'Chart Formatting': 'tableau_chart_formatting',
    'Format Pane': 'tableau_format_pane',
    'Filter Shelf': 'tableau_filter_shelf',
    'Parameter Creation': 'tableau_parameter',
    'Dashboard Actions': 'tableau_dashboard_actions',
    'Dashboard Workspace': 'tableau_dashboard_workspace',
    'Dashboard Interactivity': 'tableau_dashboard_interactivity',
    'Final Dashboard': 'tableau_final_dashboard',
    'Final Dashboard Preview': 'tableau_final_dashboard',
    'Line Chart': 'tableau_line_chart',
    'Line Chart Setup': 'tableau_line_chart',
    'Map View': 'tableau_map',
    'Map Creation': 'tableau_map',
    'Bar Chart': 'tableau_bar_chart',
    'First Chart Creation': 'tableau_first_chart',
    'Heat Map': 'tableau_heatmap',
    'Scatter Plot': 'tableau_scatter',
    'Calculation Editor': 'tableau_calculated_field',
    'Calculated Field Creation': 'tableau_calculated_field',
    'Table Calculation': 'tableau_table_calc',
    'LOD Editor': 'tableau_lod',
    'LOD in View': 'tableau_lod',
    'Tableau Prep Interface': 'tableau_prep',
    'Flow Management': 'tableau_flow',
    'Custom Visualizations': 'tableau_custom_viz',
    'Advanced Mapping': 'tableau_advanced_map',
    'Map with Indicators': 'tableau_advanced_map',
    'Content Management': 'tableau_content_mgmt',
    'Collaboration Features': 'tableau_collaboration',
    'Save and Share Options': 'tableau_collaboration',
    'Extract Configuration': 'tableau_extract',
    'Performance Tools': 'tableau_performance',
    'Security Configuration': 'tableau_security',
    'Governance Tools': 'tableau_governance',
    'Mobile Features': 'tableau_mobile',
    'Mobile App': 'tableau_mobile_app',
    'Sample Superstore Data': 'tableau_data_source',

    # Power BI
    'Power BI Start Page': 'powerbi_start_page',
    'Power BI Workspace': 'powerbi_workspace',
    'Data Model View': 'powerbi_data_model',
    'DAX Measure Creation': 'powerbi_dax',
    'Save and Publish Options': 'powerbi_publish',
    'Power Query Editor': 'powerbi_query',
    'Data Modeling': 'powerbi_modeling',
    'Custom Visuals': 'powerbi_custom_visuals',
    'Advanced Charts': 'powerbi_advanced_charts',
    'Workspace Management': 'powerbi_workspace_mgmt',
    'Refresh Configuration': 'powerbi_refresh',

    # Looker Studio
    'Looker Studio Start Page': 'looker_start_page',
    'Looker Studio Workspace': 'looker_workspace',
    'Data Source Configuration': 'looker_data_source',
    'Time Series Setup': 'looker_timeseries',
    'Dashboard Building': 'looker_dashboard',
    'Advanced Calculations': 'looker_calculations',
    'Interactive Controls': 'looker_controls',
    'Data Blending': 'looker_blending',
    'Data Source Management': 'looker_source_mgmt',
    'Interactive Features': 'looker_interactivity',
    'Sharing Options': 'looker_sharing',
    'Performance Monitoring': 'looker_performance',
    'Dashboard Optimization': 'looker_optimization',
    'AI Features': 'looker_ai',
}

def get_image_name(screenshot_text, filepath):
    """Get the image name based on screenshot text and file context."""
    # Direct mapping
    if screenshot_text in SCREENSHOT_MAPPING:
        return SCREENSHOT_MAPPING[screenshot_text]

    # Context-based mapping for generic names
    filename = os.path.basename(filepath).lower()

    if 'powerbi' in filename:
        prefix = 'powerbi'
    elif 'looker' in filename:
        prefix = 'looker'
    else:
        prefix = 'tableau'

    # Generic mappings with context
    generic_mappings = {
        'First Chart Creation': f'{prefix}_first_chart',
        'Line Chart Setup': f'{prefix}_line_chart',
        'Map Creation': f'{prefix}_map',
        'Dashboard Building': f'{prefix}_dashboard',
        'Parameter Creation': f'{prefix}_parameter',
        'Collaboration Features': f'{prefix}_collaboration',
        'Performance Tools': f'{prefix}_performance',
        'Security Configuration': f'{prefix}_security',
        'Governance Tools': f'{prefix}_governance',
        'Final Dashboard Preview': f'{prefix}_final_dashboard',
        'Sample Superstore Data': f'{prefix}_data_source',
        'Advanced Charts': f'{prefix}_advanced_charts',
    }

    if screenshot_text in generic_mappings:
        return generic_mappings[screenshot_text]

    # Generate from text
    slug = screenshot_text.lower().replace(' ', '_').replace('-', '_')
    slug = re.sub(r'[^a-z0-9_]', '', slug)
    return f'{prefix}_{slug}'

def update_markdown_file(filepath):
    """Update a single markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern: [Screenshot: Something]
    pattern = r'\[Screenshot:\s*([^\]]+)\]'

    def replace_screenshot(match):
        screenshot_text = match.group(1).strip()
        image_name = get_image_name(screenshot_text, filepath)
        return f'![{screenshot_text}](assets/{image_name}.png)'

    content = re.sub(pattern, replace_screenshot, content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# Files to update
files = [
    '../tableau-basics.md',
    '../tableau-case-study.md',
    '../powerbi-case-study.md',
    '../looker-studio-case-study.md',
]

print("Updating markdown files...")
for filepath in files:
    full_path = os.path.join(os.path.dirname(__file__), filepath)
    if os.path.exists(full_path):
        if update_markdown_file(full_path):
            print(f"  Updated: {filepath}")
        else:
            print(f"  No changes: {filepath}")
    else:
        print(f"  Not found: {filepath}")

print("\nDone!")
