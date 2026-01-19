#!/usr/bin/env python3
"""
Generate EDA visualization examples for 2.3 EDA module.
Run: python generate_visualizations.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up professional defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

# Create sample data
np.random.seed(42)

# Normal distribution
normal_data = np.random.normal(100, 15, 1000)

# Skewed distribution
skewed_data = np.random.exponential(2, 1000) * 20

# Bimodal distribution
bimodal_data = np.concatenate([np.random.normal(30, 5, 500),
                                np.random.normal(70, 8, 500)])

# Sample dataset (similar to tips)
n = 200
sample_df = pd.DataFrame({
    'sales': np.random.exponential(100, n) + 50,
    'profit': np.random.normal(20, 10, n),
    'quantity': np.random.poisson(5, n) + 1,
    'category': np.random.choice(['A', 'B', 'C', 'D'], n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'date': pd.date_range('2023-01-01', periods=n, freq='D')
})
sample_df['profit_margin'] = sample_df['profit'] / sample_df['sales'] * 100

print("Generating EDA visualizations...")

# =============================================================================
# DISTRIBUTION VISUALIZATIONS
# =============================================================================

# 1. Distribution Types Comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Normal
sns.histplot(normal_data, kde=True, ax=axes[0], color='steelblue')
axes[0].axvline(np.mean(normal_data), color='red', linestyle='--', label='Mean')
axes[0].axvline(np.median(normal_data), color='green', linestyle=':', label='Median')
axes[0].set_title('Normal Distribution', fontweight='bold')
axes[0].legend()

# Skewed
sns.histplot(skewed_data, kde=True, ax=axes[1], color='coral')
axes[1].axvline(np.mean(skewed_data), color='red', linestyle='--', label='Mean')
axes[1].axvline(np.median(skewed_data), color='green', linestyle=':', label='Median')
axes[1].set_title('Right-Skewed Distribution', fontweight='bold')
axes[1].legend()

# Bimodal
sns.histplot(bimodal_data, kde=True, ax=axes[2], color='purple')
axes[2].axvline(np.mean(bimodal_data), color='red', linestyle='--', label='Mean')
axes[2].axvline(np.median(bimodal_data), color='green', linestyle=':', label='Median')
axes[2].set_title('Bimodal Distribution', fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig('distribution_types.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - distribution_types.png")

# 2. Histogram with Statistics
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(sample_df['sales'], kde=True, ax=ax, color='steelblue', bins=30)

# Add statistics
mean_val = sample_df['sales'].mean()
median_val = sample_df['sales'].median()
std_val = sample_df['sales'].std()

ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.1f}')
ax.axvline(mean_val - std_val, color='orange', linestyle='-.', alpha=0.7, label=f'Std: {std_val:.1f}')
ax.axvline(mean_val + std_val, color='orange', linestyle='-.', alpha=0.7)

ax.fill_betweenx([0, ax.get_ylim()[1]], mean_val - std_val, mean_val + std_val,
                  alpha=0.2, color='orange', label='1 Std Dev')

ax.set_title('Sales Distribution with Statistics', fontsize=14, fontweight='bold')
ax.set_xlabel('Sales ($)')
ax.set_ylabel('Frequency')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('histogram_with_stats.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - histogram_with_stats.png")

# 3. Box Plot Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot by category
sns.boxplot(data=sample_df, x='category', y='sales', ax=axes[0], palette='Set2')
axes[0].set_title('Sales by Category', fontweight='bold')

# Box plot by region
sns.boxplot(data=sample_df, x='region', y='profit', ax=axes[1], palette='Set3')
axes[1].set_title('Profit by Region', fontweight='bold')

plt.tight_layout()
plt.savefig('boxplot_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - boxplot_comparison.png")

# 4. Violin Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=sample_df, x='category', y='sales', hue='region',
               split=False, inner='box', ax=ax, palette='pastel')
ax.set_title('Sales Distribution by Category and Region', fontsize=14, fontweight='bold')
ax.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('violin_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - violin_plot.png")

# 5. QQ Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# QQ for normal data
stats.probplot(normal_data, dist="norm", plot=axes[0])
axes[0].set_title('QQ Plot: Normal Data', fontweight='bold')
axes[0].get_lines()[1].set_color('red')

# QQ for skewed data
stats.probplot(skewed_data, dist="norm", plot=axes[1])
axes[1].set_title('QQ Plot: Skewed Data', fontweight='bold')
axes[1].get_lines()[1].set_color('red')

plt.tight_layout()
plt.savefig('qq_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - qq_plots.png")

# 6. Outlier Detection
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Add outliers to data
data_with_outliers = np.concatenate([normal_data, [200, 210, -50, 220]])

# Box plot showing outliers
bp = axes[0].boxplot(data_with_outliers, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
axes[0].scatter([1]*4, [200, 210, -50, 220], color='red', s=100, zorder=5, label='Outliers')
axes[0].set_title('Box Plot with Outliers', fontweight='bold')
axes[0].legend()

# Histogram with outlier region
sns.histplot(data_with_outliers, kde=True, ax=axes[1], color='steelblue')
q1, q3 = np.percentile(data_with_outliers, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
axes[1].axvline(lower, color='red', linestyle='--', label=f'Lower bound: {lower:.1f}')
axes[1].axvline(upper, color='red', linestyle='--', label=f'Upper bound: {upper:.1f}')
axes[1].axvspan(axes[1].get_xlim()[0], lower, alpha=0.2, color='red')
axes[1].axvspan(upper, axes[1].get_xlim()[1], alpha=0.2, color='red')
axes[1].set_title('Histogram with Outlier Regions', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - outlier_detection.png")

# =============================================================================
# RELATIONSHIP VISUALIZATIONS
# =============================================================================

# 7. Scatter Plot with Regression
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(sample_df['sales'], sample_df['profit'],
                     c=sample_df['quantity'], cmap='viridis',
                     alpha=0.6, s=50)
# Regression line
z = np.polyfit(sample_df['sales'], sample_df['profit'], 1)
p = np.poly1d(z)
x_line = np.linspace(sample_df['sales'].min(), sample_df['sales'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.2f}')

plt.colorbar(scatter, ax=ax, label='Quantity')
ax.set_xlabel('Sales ($)')
ax.set_ylabel('Profit ($)')
ax.set_title('Sales vs Profit Relationship', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('scatter_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_regression.png")

# 8. Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
corr_matrix = sample_df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
            center=0, fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - correlation_heatmap.png")

# 9. Pair Plot Sample
pair_df = sample_df[['sales', 'profit', 'quantity', 'category']].copy()
g = sns.pairplot(pair_df, hue='category', palette='husl',
                 diag_kind='kde', height=2.5)
g.fig.suptitle('Pair Plot: Multivariate Relationships', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - pairplot.png")

# 10. Grouped Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
grouped = sample_df.groupby(['category', 'region'])['sales'].mean().unstack()
grouped.plot(kind='bar', ax=ax, width=0.8, colormap='Set2')
ax.set_title('Average Sales by Category and Region', fontsize=14, fontweight='bold')
ax.set_xlabel('Category')
ax.set_ylabel('Average Sales ($)')
ax.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('grouped_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - grouped_bar.png")

# =============================================================================
# TIME SERIES VISUALIZATIONS
# =============================================================================

# Create time series data
dates = pd.date_range('2022-01-01', periods=365, freq='D')
np.random.seed(42)
trend = np.linspace(0, 50, 365)
seasonal = 20 * np.sin(np.linspace(0, 4*np.pi, 365))
noise = np.random.normal(0, 10, 365)
ts_data = pd.DataFrame({
    'date': dates,
    'value': 100 + trend + seasonal + noise
})
ts_data['month'] = ts_data['date'].dt.month
ts_data['weekday'] = ts_data['date'].dt.day_name()

# 11. Time Series Line Plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ts_data['date'], ts_data['value'], color='steelblue', linewidth=1)
# Add rolling average
rolling_avg = ts_data['value'].rolling(30).mean()
ax.plot(ts_data['date'], rolling_avg, color='red', linewidth=2, label='30-day MA')
ax.set_title('Time Series with Trend', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
plt.tight_layout()
plt.savefig('timeseries_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - timeseries_trend.png")

# 12. Seasonal Decomposition Visualization
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Original
axes[0].plot(ts_data['date'], ts_data['value'], color='steelblue')
axes[0].set_title('Original Time Series', fontweight='bold')
axes[0].set_ylabel('Value')

# Trend
axes[1].plot(ts_data['date'], 100 + trend, color='red')
axes[1].set_title('Trend Component', fontweight='bold')
axes[1].set_ylabel('Trend')

# Seasonal
axes[2].plot(ts_data['date'], seasonal, color='green')
axes[2].set_title('Seasonal Component', fontweight='bold')
axes[2].set_ylabel('Seasonal')

# Residual
axes[3].plot(ts_data['date'], noise, color='purple', alpha=0.7)
axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[3].set_title('Residual Component', fontweight='bold')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.savefig('seasonal_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - seasonal_decomposition.png")

# 13. Monthly Pattern Box Plot
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=ts_data, x='month', y='value', ax=ax, palette='coolwarm')
ax.set_title('Value Distribution by Month', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Value')
plt.tight_layout()
plt.savefig('monthly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - monthly_pattern.png")

# 14. Autocorrelation Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ACF
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ts_data['value'], ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold')
axes[0].set_xlim([0, 50])

# PACF (manual calculation)
lags = range(1, 31)
pacf_values = [ts_data['value'].autocorr(lag=lag) for lag in lags]
axes[1].bar(lags, pacf_values, color='steelblue', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].axhline(y=1.96/np.sqrt(len(ts_data)), color='red', linestyle='--', alpha=0.7)
axes[1].axhline(y=-1.96/np.sqrt(len(ts_data)), color='red', linestyle='--', alpha=0.7)
axes[1].set_title('Partial Autocorrelation', fontweight='bold')
axes[1].set_xlabel('Lag')

plt.tight_layout()
plt.savefig('autocorrelation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - autocorrelation.png")

# 15. EDA Summary Dashboard
fig = plt.figure(figsize=(14, 10))

# Distribution
ax1 = fig.add_subplot(2, 3, 1)
sns.histplot(sample_df['sales'], kde=True, ax=ax1, color='steelblue')
ax1.set_title('Sales Distribution', fontweight='bold')

# Box plot
ax2 = fig.add_subplot(2, 3, 2)
sns.boxplot(data=sample_df, x='category', y='sales', ax=ax2, palette='Set2')
ax2.set_title('Sales by Category', fontweight='bold')

# Scatter
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(sample_df['sales'], sample_df['profit'], alpha=0.5, c='steelblue')
ax3.set_xlabel('Sales')
ax3.set_ylabel('Profit')
ax3.set_title('Sales vs Profit', fontweight='bold')

# Time series
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(ts_data['date'][:90], ts_data['value'][:90], color='steelblue')
ax4.set_title('Recent Trend (90 days)', fontweight='bold')
ax4.tick_params(axis='x', rotation=45)

# Category counts
ax5 = fig.add_subplot(2, 3, 5)
sample_df['category'].value_counts().plot(kind='bar', ax=ax5, color='coral')
ax5.set_title('Category Distribution', fontweight='bold')
ax5.tick_params(axis='x', rotation=0)

# Correlation mini heatmap
ax6 = fig.add_subplot(2, 3, 6)
corr_mini = sample_df[['sales', 'profit', 'quantity']].corr()
sns.heatmap(corr_mini, annot=True, cmap='coolwarm', center=0, ax=ax6, fmt='.2f')
ax6.set_title('Correlations', fontweight='bold')

plt.suptitle('EDA Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - eda_dashboard.png")

print("\n" + "="*50)
print("Generated 15 EDA visualization images successfully!")
print("="*50)
