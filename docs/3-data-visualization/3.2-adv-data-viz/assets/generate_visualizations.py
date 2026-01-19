#!/usr/bin/env python3
"""
Generate visualization examples for 3.2 Advanced Data Visualization module.
Run: python generate_visualizations.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set up professional defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

# Load sample datasets
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
flights = sns.load_dataset("flights")
penguins = sns.load_dataset("penguins").dropna()

print("Generating Seaborn visualizations...")

# =============================================================================
# SEABORN VISUALIZATIONS
# =============================================================================

# 1. Distribution Suite
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram with KDE
sns.histplot(data=tips, x="total_bill", kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title('Histogram with KDE', fontsize=12, fontweight='bold')

# Box plot
sns.boxplot(data=tips, y="total_bill", ax=axes[0, 1], color="lightcoral")
axes[0, 1].set_title('Box Plot', fontsize=12, fontweight='bold')

# Violin plot
sns.violinplot(data=tips, y="total_bill", ax=axes[1, 0], color="lightgreen")
axes[1, 0].set_title('Violin Plot', fontsize=12, fontweight='bold')

# ECDF
sns.ecdfplot(data=tips, x="total_bill", ax=axes[1, 1], color="coral")
axes[1, 1].set_title('Empirical CDF', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('seaborn_distribution_suite.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_distribution_suite.png")

# 2. Categorical Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Box plot by category
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title('Box Plot by Category', fontsize=12, fontweight='bold')

# Violin plot by category
sns.violinplot(data=tips, x="day", y="total_bill", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title('Violin Plot by Category', fontsize=12, fontweight='bold')

# Strip plot
sns.stripplot(data=tips, x="day", y="total_bill", ax=axes[1, 0],
              alpha=0.6, jitter=0.2, palette="Set2")
axes[1, 0].set_title('Strip Plot', fontsize=12, fontweight='bold')

# Swarm plot
sns.swarmplot(data=tips, x="day", y="total_bill", ax=axes[1, 1],
              palette="Set2", size=4)
axes[1, 1].set_title('Swarm Plot', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('seaborn_categorical_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_categorical_analysis.png")

# 3. Scatter Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Basic scatter with hue
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time",
                ax=axes[0, 0], palette="deep")
axes[0, 0].set_title('Scatter Plot with Hue', fontsize=12, fontweight='bold')

# Regression plot
sns.regplot(data=tips, x="total_bill", y="tip", ax=axes[0, 1],
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
axes[0, 1].set_title('Scatter with Regression', fontsize=12, fontweight='bold')

# Residual plot
sns.residplot(data=tips, x="total_bill", y="tip", ax=axes[1, 0],
              scatter_kws={'alpha': 0.5})
axes[1, 0].set_title('Residual Plot', fontsize=12, fontweight='bold')

# KDE plot (2D)
sns.kdeplot(data=tips, x="total_bill", y="tip", ax=axes[1, 1],
            fill=True, cmap="YlOrRd", levels=10)
axes[1, 1].set_title('2D KDE Plot', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('seaborn_scatter_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_scatter_analysis.png")

# 4. Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = tips.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('seaborn_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_correlation_heatmap.png")

# 5. Pair Plot
g = sns.pairplot(iris, hue="species", palette="husl",
                 diag_kind="kde", height=2.5)
g.fig.suptitle('Pair Plot - Iris Dataset', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('seaborn_pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_pairplot.png")

# 6. Facet Grid
g = sns.FacetGrid(tips, col="time", row="smoker", height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", hue="day", palette="Set2")
g.add_legend()
g.fig.suptitle('Facet Grid - Tips by Time and Smoker', y=1.02,
               fontsize=14, fontweight='bold')
plt.savefig('seaborn_facetgrid.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_facetgrid.png")

# 7. Clustermap
g = sns.clustermap(corr, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=0.5, figsize=(8, 8))
g.fig.suptitle('Cluster Map', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('seaborn_clustermap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_clustermap.png")

# 8. Bar Plot with Error Bars
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=tips, x="day", y="total_bill", hue="sex",
            palette="Set1", ax=ax, errorbar="sd")
ax.set_title('Bar Plot with Error Bars', fontsize=14, fontweight='bold')
ax.legend(title="Gender")
plt.tight_layout()
plt.savefig('seaborn_barplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_barplot.png")

# 9. Count Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=tips, x="day", hue="time", palette="pastel", ax=ax)
ax.set_title('Count Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('seaborn_countplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_countplot.png")

# 10. Joint Plot
g = sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg",
                  height=8, ratio=4, marginal_kws=dict(bins=25))
g.fig.suptitle('Joint Plot with Regression', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('seaborn_jointplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_jointplot.png")

# 11. Heatmap - Flights pivot
flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(flights_pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
ax.set_title('Heatmap - Flight Passengers', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('seaborn_flights_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_flights_heatmap.png")

# 12. Line Plot with confidence interval
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=flights, x="year", y="passengers", hue="month",
             palette="tab10", ax=ax)
ax.set_title('Line Plot with Multiple Series', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('seaborn_lineplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seaborn_lineplot.png")

# =============================================================================
# PLOTLY-STYLE STATIC VISUALIZATIONS (for documentation)
# Using matplotlib to create plotly-style outputs
# =============================================================================

print("\nGenerating Plotly-style visualizations...")

# Create sample data for Plotly-style plots
np.random.seed(42)
n_countries = 5
years = list(range(1952, 2008, 5))
continents = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']

# Simulated gapminder-style data
gapminder_sample = pd.DataFrame({
    'country': np.repeat(['USA', 'China', 'Brazil', 'Germany', 'Nigeria'], len(years)),
    'continent': np.repeat(['Americas', 'Asia', 'Americas', 'Europe', 'Africa'], len(years)),
    'year': years * 5,
    'lifeExp': np.random.uniform(50, 85, 5 * len(years)),
    'pop': np.random.uniform(10e6, 1e9, 5 * len(years)),
    'gdpPercap': np.random.uniform(1000, 50000, 5 * len(years))
})

# 13. Animated scatter (static frame representation)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for idx, year in enumerate([1952, 1972, 1992, 2002]):
    ax = axes[idx // 2, idx % 2]
    data_year = gapminder_sample[gapminder_sample['year'] == year]
    scatter = ax.scatter(data_year['gdpPercap'], data_year['lifeExp'],
                        s=data_year['pop']/1e7, alpha=0.7,
                        c=range(len(data_year)), cmap='Set1')
    ax.set_xlabel('GDP per Capita')
    ax.set_ylabel('Life Expectancy')
    ax.set_title(f'Year: {year}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 55000)
    ax.set_ylim(45, 90)
    ax.grid(True, alpha=0.3)
fig.suptitle('Interactive Scatter - Animation Frames', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plotly_animated_scatter_frames.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - plotly_animated_scatter_frames.png")

# 14. Distribution Dashboard
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes[0, 0].hist(tips['total_bill'], bins=30, color='steelblue',
                edgecolor='white', alpha=0.7)
axes[0, 0].set_title('Histogram', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Total Bill')
axes[0, 0].grid(True, alpha=0.3)

# Box Plot
bp = axes[0, 1].boxplot([tips[tips['day'] == d]['total_bill'] for d in tips['day'].unique()],
                        labels=tips['day'].unique(), patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 1].set_title('Box Plot by Day', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Violin Plot (using seaborn for better result)
sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[1, 0],
               palette='pastel', inner='box')
axes[1, 0].set_title('Violin Plot', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# QQ Plot
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(tips['total_bill'], dist='norm')
axes[1, 1].scatter(osm, osr, alpha=0.5, color='steelblue')
axes[1, 1].plot(osm, slope * osm + intercept, 'r-', linewidth=2)
axes[1, 1].set_title('QQ Plot', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Quantiles')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Distribution Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plotly_distribution_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - plotly_distribution_dashboard.png")

# 15. Time Series with Range Slider representation
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

# Main plot
for continent in continents[:3]:
    data = gapminder_sample[gapminder_sample['continent'] == continent]
    axes[0].plot(data['year'], data['lifeExp'], marker='o', label=continent, linewidth=2)
axes[0].set_title('Life Expectancy Over Time', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Life Expectancy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Range slider representation
for continent in continents[:3]:
    data = gapminder_sample[gapminder_sample['continent'] == continent]
    axes[1].plot(data['year'], data['lifeExp'], linewidth=1, alpha=0.5)
axes[1].set_xlabel('Year')
axes[1].fill_between([1980, 2000], [0, 0], [100, 100], alpha=0.2, color='blue')
axes[1].set_ylim(45, 90)
axes[1].set_title('Range Selector', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plotly_timeseries_rangeslider.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - plotly_timeseries_rangeslider.png")

# 16. 3D Scatter representation
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(tips['total_bill'], tips['tip'], tips['size'],
                     c=tips['total_bill'], cmap='viridis', s=50, alpha=0.7)
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
ax.set_zlabel('Party Size')
ax.set_title('3D Scatter Plot', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Total Bill', shrink=0.5)
plt.tight_layout()
plt.savefig('plotly_3d_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - plotly_3d_scatter.png")

# 17. Sunburst/Treemap representation (using nested pie)
fig, ax = plt.subplots(figsize=(10, 10))
day_totals = tips.groupby('day')['total_bill'].sum()
time_totals = tips.groupby(['day', 'time'])['total_bill'].sum()

# Outer ring - by day
colors_outer = plt.cm.Set2(np.linspace(0, 1, len(day_totals)))
ax.pie(day_totals, labels=day_totals.index, colors=colors_outer,
       radius=1, wedgeprops=dict(width=0.3, edgecolor='white'),
       labeldistance=1.1)

# Inner ring - by time
inner_vals = []
inner_colors = []
for day in day_totals.index:
    for time in ['Lunch', 'Dinner']:
        try:
            val = time_totals.loc[(day, time)]
            inner_vals.append(val)
            inner_colors.append(plt.cm.Pastel1(0.3 if time == 'Lunch' else 0.7))
        except KeyError:
            pass

ax.pie(inner_vals, colors=inner_colors, radius=0.7,
       wedgeprops=dict(width=0.3, edgecolor='white'))

ax.set_title('Hierarchical Pie Chart\n(Sunburst-style)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plotly_sunburst_style.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - plotly_sunburst_style.png")

# 18. Subplots Dashboard
fig = plt.figure(figsize=(14, 10))

# Scatter
ax1 = fig.add_subplot(2, 2, 1)
scatter = ax1.scatter(tips['total_bill'], tips['tip'],
                      c=tips['size'], cmap='viridis', alpha=0.7, s=50)
ax1.set_xlabel('Total Bill')
ax1.set_ylabel('Tip')
ax1.set_title('Scatter: Bill vs Tip', fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Party Size')

# Bar
ax2 = fig.add_subplot(2, 2, 2)
day_means = tips.groupby('day')['total_bill'].mean()
bars = ax2.bar(day_means.index, day_means.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_ylabel('Average Total Bill')
ax2.set_title('Average Bill by Day', fontweight='bold')

# Line
ax3 = fig.add_subplot(2, 2, 3)
for continent in continents[:3]:
    data = gapminder_sample[gapminder_sample['continent'] == continent]
    ax3.plot(data['year'], data['lifeExp'], marker='o', label=continent)
ax3.set_xlabel('Year')
ax3.set_ylabel('Life Expectancy')
ax3.set_title('Life Expectancy Trend', fontweight='bold')
ax3.legend()

# Pie
ax4 = fig.add_subplot(2, 2, 4)
day_counts = tips['day'].value_counts()
ax4.pie(day_counts, labels=day_counts.index, autopct='%1.1f%%',
        colors=plt.cm.Pastel1(np.linspace(0, 1, len(day_counts))))
ax4.set_title('Distribution by Day', fontweight='bold')

plt.suptitle('Interactive Dashboard Layout', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plotly_dashboard_layout.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - plotly_dashboard_layout.png")

# 19. Hover information example (annotated plot)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(tips['total_bill'], tips['tip'],
                     c=tips['size'], cmap='viridis', s=80, alpha=0.7)

# Add sample annotations to show hover concept
sample_indices = [0, 50, 100, 150]
for idx in sample_indices:
    row = tips.iloc[idx]
    ax.annotate(f'Bill: ${row["total_bill"]:.2f}\nTip: ${row["tip"]:.2f}\nSize: {row["size"]}',
                xy=(row['total_bill'], row['tip']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Total Bill ($)')
ax.set_ylabel('Tip ($)')
ax.set_title('Interactive Hover Information', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Party Size')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plotly_hover_example.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - plotly_hover_example.png")

# 20. Theme Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

themes = [
    ('whitegrid', 'white', 'Modern Theme'),
    ('darkgrid', '#2d2d2d', 'Dark Theme'),
    ('ticks', 'white', 'Minimal Theme')
]

for ax, (style, bg, title) in zip(axes, themes):
    with plt.style.context(f'seaborn-v0_8-{style}'):
        ax.scatter(tips['total_bill'][:50], tips['tip'][:50],
                   c=range(50), cmap='viridis', s=60, alpha=0.7)
        ax.set_facecolor(bg)
        ax.set_xlabel('Total Bill')
        ax.set_ylabel('Tip')
        ax.set_title(title, fontweight='bold')

plt.suptitle('Theme Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plotly_theme_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - plotly_theme_comparison.png")

print("\n" + "="*50)
print(f"Generated 20 visualization images successfully!")
print("="*50)
