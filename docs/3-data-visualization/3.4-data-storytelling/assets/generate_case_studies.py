from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better readability
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Brand color constants
WALMART_COLORS = {
    "primary": "#004C91",  # Walmart Blue
    "secondary": "#FFC220",  # Walmart Yellow
    "accent": "#78BE20",  # Walmart Green
    "negative": "#E60012",  # Red for negative trends
    "neutral": "#7C7C7C",  # Gray for neutral data
}

SPOTIFY_COLORS = {
    "primary": "#1DB954",  # Spotify Green
    "secondary": "#191414",  # Spotify Black
    "accent": "#FFFFFF",  # White
    "warning": "#FFB437",  # Warning Yellow
    "error": "#FF4632",  # Error Red
}

AIRBNB_COLORS = {
    "primary": "#FF5A5F",  # Airbnb Coral
    "secondary": "#00A699",  # Airbnb Teal
    "accent": "#FC642D",  # Airbnb Orange
    "neutral": "#484848",  # Airbnb Gray
}

TESLA_COLORS = {
    "primary": "#E31937",  # Tesla Red
    "secondary": "#181B21",  # Tesla Black
    "accent": "#FFFFFF",  # White
    "neutral": "#393C41",  # Tesla Gray
}

NETFLIX_COLORS = {
    "primary": "#E50914",  # Netflix Red
    "secondary": "#221F1F",  # Netflix Black
    "accent": "#F5F5F1",  # Netflix Light Gray
    "neutral": "#831010",  # Darker Red
}


def create_bad_dashboard():
    """Create an example of a bad dashboard with Walmart data"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Store Performance Dashboard", fontsize=16)

    # Random data
    np.random.seed(42)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    sales = np.random.normal(100, 20, 6)
    costs = np.random.normal(80, 15, 6)
    profit = sales - costs

    # 1. Cluttered bar chart with inconsistent colors
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    for i, (metric, data) in enumerate(
        [
            ("Sales", sales),
            ("Costs", costs),
            ("Profit", profit),
            ("Returns", np.random.normal(20, 5, 6)),
            ("Shrinkage", np.random.normal(10, 2, 6)),
        ]
    ):
        axes[0, 0].bar(months, data, color=colors[i], alpha=0.7, label=metric)
    axes[0, 0].set_title("All Metrics (Cluttered)")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1, 1))

    # 2. Confusing pie chart with too many segments
    sizes = [15, 12, 10, 8, 7, 6, 5, 4, 3, 2]
    labels = [
        "Cat A",
        "Cat B",
        "Cat C",
        "Cat D",
        "Cat E",
        "Cat F",
        "Cat G",
        "Cat H",
        "Cat I",
        "Cat J",
    ]
    axes[0, 1].pie(
        sizes, labels=labels, colors=plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    )
    axes[0, 1].set_title("Product Categories (Too Many)")

    # 3. Overcrowded line chart
    for i in range(8):
        axes[1, 0].plot(
            months, np.random.normal(100, 20, 6), label=f"Metric {i+1}", marker="o"
        )
    axes[1, 0].set_title("Daily Metrics (Overwhelming)")
    axes[1, 0].legend(loc="upper left", bbox_to_anchor=(1, 1))

    # 4. Confusing scatter plot with no clear meaning
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    colors = np.random.rand(100)
    axes[1, 1].scatter(x, y, c=colors, cmap="rainbow")
    axes[1, 1].set_title("Customer Data (Unclear)")

    plt.tight_layout()
    plt.savefig("bad_dashboard.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_dashboard():
    """Create an example of a good dashboard with Walmart data"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Store Performance Dashboard", fontsize=16, color=WALMART_COLORS["primary"]
    )

    # Random data
    np.random.seed(42)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    sales = np.random.normal(100, 20, 6)
    costs = np.random.normal(80, 15, 6)
    profit = sales - costs

    # 1. Clear sales trend
    axes[0, 0].plot(
        months,
        sales,
        color=WALMART_COLORS["primary"],
        marker="o",
        linewidth=2,
        label="Sales",
    )
    axes[0, 0].fill_between(months, sales, alpha=0.2, color=WALMART_COLORS["primary"])
    axes[0, 0].set_title("Monthly Sales Trend", color=WALMART_COLORS["primary"])
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Profit margin with clear color coding
    profit_margin = (profit / sales) * 100
    colors = [
        WALMART_COLORS["accent"] if x >= 0 else WALMART_COLORS["negative"]
        for x in profit_margin
    ]
    axes[0, 1].bar(months, profit_margin, color=colors)
    axes[0, 1].set_title("Profit Margin %", color=WALMART_COLORS["primary"])
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Key metrics summary
    metrics = ["Sales", "Profit", "Customers", "Satisfaction"]
    values = [95, 85, 90, 88]
    axes[1, 0].bar(metrics, values, color=WALMART_COLORS["secondary"])
    axes[1, 0].set_title("Key Performance Metrics", color=WALMART_COLORS["primary"])
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Year-over-year comparison
    current = np.array([100, 105, 110, 108, 115, 120])
    previous = np.array([98, 100, 102, 105, 108, 110])
    axes[1, 1].plot(
        months, current, color=WALMART_COLORS["primary"], marker="o", label="This Year"
    )
    axes[1, 1].plot(
        months,
        previous,
        color=WALMART_COLORS["neutral"],
        linestyle="--",
        marker="s",
        label="Last Year",
    )
    axes[1, 1].set_title("Year-over-Year Sales", color=WALMART_COLORS["primary"])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("good_dashboard.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_journey():
    """Create an example of a bad Spotify user journey map"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Text-heavy, confusing representation
    steps = [
        "Sign Up",
        "Profile Creation",
        "Music Preferences",
        "First Search",
        "Playlist Creation",
        "Regular Usage",
    ]
    users = [100, 82, 65, 45, 30, 20]

    # Inconsistent colors and no clear flow
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    ax.bar(steps, users, color=colors)

    # Overwhelming text annotations
    for i, v in enumerate(users):
        ax.text(
            i,
            v + 5,
            f"""
        Step {i+1}
        Users: {v}
        Drop: {users[i-1]-v if i > 0 else 0}
        Actions: Multiple
        Time: Varies
        Issues: Several
        """,
            ha="center",
            va="bottom",
        )

    ax.set_title("User Onboarding Flow (Confusing)")
    ax.set_xticklabels(steps, rotation=45)

    plt.tight_layout()
    plt.savefig("bad_journey.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_journey():
    """Create an example of a good Spotify user journey map"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Clear steps with consistent branding
    steps = ["Sign Up", "Profile", "Preferences", "Search", "Playlist", "Regular"]
    users = [100, 85, 70, 55, 45, 40]

    # Use Spotify's brand colors
    bars = ax.bar(steps, users, color=SPOTIFY_COLORS["primary"])

    # Clear, actionable annotations
    for i, v in enumerate(users):
        drop = users[i - 1] - v if i > 0 else 0
        color = (
            SPOTIFY_COLORS["error"]
            if drop > 20
            else SPOTIFY_COLORS["warning"] if drop > 10 else SPOTIFY_COLORS["primary"]
        )

        ax.text(i, v + 2, f"{v}%", ha="center", color=color)
        if i > 0:
            ax.text(
                i, v / 2, f"-{drop}%", ha="center", color=SPOTIFY_COLORS["secondary"]
            )

    ax.set_title(
        "User Journey - Clear Progress & Drop-offs", color=SPOTIFY_COLORS["secondary"]
    )
    ax.set_ylabel("User Retention %")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("good_journey.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_campaign():
    """Create an example of a bad Airbnb campaign report"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a messy campaign report with raw data
    channels = ["Facebook", "Instagram", "Google", "Email", "Direct", "Affiliates"]
    metrics = [
        "Impressions",
        "Clicks",
        "Bookings",
        "Revenue",
        "ROI",
        "CPA",
        "CTR",
        "CVR",
    ]

    # Random data with no clear story
    np.random.seed(42)
    data = np.random.normal(100, 20, (len(channels), len(metrics)))

    # Overwhelming heatmap with no clear insights
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        xticklabels=metrics,
        yticklabels=channels,
        ax=ax,
    )

    ax.set_title("Marketing Campaign Performance (Raw Data)")
    plt.tight_layout()
    plt.savefig("bad_campaign.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_campaign():
    """Create an example of a good Airbnb campaign report"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Marketing Campaign Performance", fontsize=16, color=AIRBNB_COLORS["primary"]
    )

    # Consistent data across visualizations
    channels = ["Facebook", "Instagram", "Google", "Email", "Direct"]
    impressions = np.array([120, 100, 80, 60, 40]) * 1000
    bookings = np.array([1200, 1000, 800, 600, 400])
    revenue = bookings * np.array([150, 120, 180, 90, 110])
    roi = (revenue - bookings * 50) / (bookings * 50) * 100

    # 1. Channel Performance Overview
    ax1 = axes[0, 0]
    ax1.bar(channels, roi, color=AIRBNB_COLORS["primary"])
    ax1.set_title("ROI by Channel", color=AIRBNB_COLORS["primary"])
    ax1.set_ylabel("ROI %")
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(axis="x", rotation=45)

    # 2. Conversion Funnel
    ax2 = axes[0, 1]
    funnel_stages = ["Impressions", "Clicks", "Bookings"]
    funnel_values = [
        impressions.sum() / 1000,
        impressions.sum() * 0.05 / 1000,
        bookings.sum(),
    ]
    ax2.bar(funnel_stages, funnel_values, color=AIRBNB_COLORS["secondary"])
    ax2.set_title("Conversion Funnel", color=AIRBNB_COLORS["primary"])
    for i, v in enumerate(funnel_values):
        ax2.text(i, v, f"{v:,.0f}", ha="center", va="bottom")
    ax2.grid(True, alpha=0.2)

    # 3. Revenue Contribution
    ax3 = axes[1, 0]
    ax3.pie(
        revenue,
        labels=channels,
        colors=[
            AIRBNB_COLORS["primary"],
            AIRBNB_COLORS["secondary"],
            AIRBNB_COLORS["accent"],
            AIRBNB_COLORS["neutral"],
            "#A1A1A1",
        ],
        autopct="%1.1f%%",
    )
    ax3.set_title("Revenue Share", color=AIRBNB_COLORS["primary"])

    # 4. Efficiency Matrix
    ax4 = axes[1, 1]
    conversion_rate = bookings / (impressions * 0.05) * 100
    ax4.scatter(
        roi,
        conversion_rate,
        s=revenue / 1000,
        c=range(len(channels)),
        cmap="RdYlBu",
        alpha=0.6,
    )
    for i, channel in enumerate(channels):
        ax4.annotate(
            channel,
            (roi[i], conversion_rate[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax4.set_xlabel("ROI %")
    ax4.set_ylabel("Conversion Rate %")
    ax4.set_title("Channel Efficiency", color=AIRBNB_COLORS["primary"])
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("good_campaign.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_financial():
    """Create an example of a bad Tesla financial report"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Overwhelming financial data
    quarters = ["Q1'21", "Q2'21", "Q3'21", "Q4'21", "Q1'22", "Q2'22", "Q3'22", "Q4'22"]
    metrics = [
        "Revenue",
        "COGS",
        "Gross Profit",
        "OpEx",
        "EBITDA",
        "D&A",
        "EBIT",
        "Tax",
        "Net Income",
        "FCF",
        "CapEx",
        "Working Capital",
    ]

    # Random data with no clear story
    np.random.seed(42)
    data = np.random.normal(100, 20, (len(quarters), len(metrics)))

    # Overwhelming heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        xticklabels=metrics,
        yticklabels=quarters,
        ax=ax,
    )

    ax.set_title("Financial Performance (Raw Data)")
    plt.tight_layout()
    plt.savefig("bad_financial.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_financial():
    """Create an example of a good Tesla financial report"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Tesla Financial Performance", fontsize=16, color=TESLA_COLORS["primary"]
    )

    quarters = ["Q1'22", "Q2'22", "Q3'22", "Q4'22"]
    revenue = np.array([18.8, 16.9, 21.5, 24.3])
    margins = np.array([32.9, 27.9, 25.1, 23.8])
    production = np.array([305, 258.6, 365.9, 439.7])

    # 1. Revenue Trend
    ax1 = axes[0, 0]
    ax1.plot(quarters, revenue, color=TESLA_COLORS["primary"], marker="o", linewidth=2)
    ax1.fill_between(quarters, revenue, alpha=0.2, color=TESLA_COLORS["primary"])
    ax1.set_title("Quarterly Revenue ($B)", color=TESLA_COLORS["primary"])
    ax1.grid(True, alpha=0.2)
    for i, v in enumerate(revenue):
        ax1.text(i, v, f"${v}B", ha="center", va="bottom")

    # 2. Margin Analysis
    ax2 = axes[0, 1]
    ax2.bar(quarters, margins, color=TESLA_COLORS["primary"])
    ax2.set_title("Gross Margin %", color=TESLA_COLORS["primary"])
    ax2.grid(True, alpha=0.2)
    for i, v in enumerate(margins):
        ax2.text(i, v, f"{v}%", ha="center", va="bottom")

    # 3. Production Numbers
    ax3 = axes[1, 0]
    ax3.plot(
        quarters, production, color=TESLA_COLORS["secondary"], marker="s", linewidth=2
    )
    ax3.fill_between(quarters, production, alpha=0.2, color=TESLA_COLORS["secondary"])
    ax3.set_title("Vehicle Production (K)", color=TESLA_COLORS["primary"])
    ax3.grid(True, alpha=0.2)
    for i, v in enumerate(production):
        ax3.text(i, v, f"{v}K", ha="center", va="bottom")

    # 4. Key Metrics Summary
    ax4 = axes[1, 1]
    metrics = ["Revenue\nGrowth", "Margin\nRetention", "Production\nEfficiency"]
    values = [29.1, 95.2, 87.5]
    colors = [
        TESLA_COLORS["primary"] if v >= 85 else TESLA_COLORS["secondary"]
        for v in values
    ]
    ax4.bar(metrics, values, color=colors)
    ax4.set_title("Key Performance Indicators", color=TESLA_COLORS["primary"])
    ax4.grid(True, alpha=0.2)
    for i, v in enumerate(values):
        ax4.text(i, v, f"{v}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("good_financial.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_usage():
    """Create an example of bad Netflix usage analytics"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw data dump of usage metrics
    features = [
        "Homepage",
        "Search",
        "Recommendations",
        "My List",
        "Continue Watching",
        "New Releases",
        "Trending Now",
        "Categories",
    ]
    metrics = [
        "Views",
        "Time Spent",
        "Clicks",
        "Conversions",
        "Bounces",
        "Returns",
        "Shares",
        "Ratings",
    ]

    # Random data with no insights
    np.random.seed(42)
    data = np.random.normal(100, 20, (len(features), len(metrics)))

    # Overwhelming heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        xticklabels=metrics,
        yticklabels=features,
        ax=ax,
    )

    ax.set_title("Feature Usage Analytics (Raw Data)")
    plt.tight_layout()
    plt.savefig("bad_usage.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_usage():
    """Create an example of good Netflix usage analytics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Netflix User Engagement Analytics",
        fontsize=16,
        color=NETFLIX_COLORS["primary"],
    )

    features = ["Homepage", "Search", "Recommendations", "My List", "Continue Watching"]
    engagement = np.array([95, 45, 75, 60, 85])
    conversion = np.array([15, 35, 45, 55, 65])
    time_spent = np.array([2.5, 1.5, 4.5, 3.5, 5.5])

    # 1. Feature Engagement
    ax1 = axes[0, 0]
    ax1.bar(features, engagement, color=NETFLIX_COLORS["primary"])
    ax1.set_title("Feature Engagement %", color=NETFLIX_COLORS["primary"])
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.2)
    for i, v in enumerate(engagement):
        ax1.text(i, v, f"{v}%", ha="center", va="bottom")

    # 2. Conversion Impact
    ax2 = axes[0, 1]
    ax2.plot(
        features, conversion, color=NETFLIX_COLORS["primary"], marker="o", linewidth=2
    )
    ax2.fill_between(features, conversion, alpha=0.2, color=NETFLIX_COLORS["primary"])
    ax2.set_title("Watch Conversion %", color=NETFLIX_COLORS["primary"])
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.2)
    for i, v in enumerate(conversion):
        ax2.text(i, v, f"{v}%", ha="center", va="bottom")

    # 3. Time Spent Analysis
    ax3 = axes[1, 0]
    ax3.bar(features, time_spent, color=NETFLIX_COLORS["secondary"])
    ax3.set_title("Avg. Time Spent (min)", color=NETFLIX_COLORS["primary"])
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.2)
    for i, v in enumerate(time_spent):
        ax3.text(i, v, f"{v}min", ha="center", va="bottom")

    # 4. Engagement Matrix
    ax4 = axes[1, 1]
    ax4.scatter(
        engagement,
        conversion,
        s=time_spent * 100,
        c=range(len(features)),
        cmap="RdYlBu",
        alpha=0.6,
    )
    for i, feature in enumerate(features):
        ax4.annotate(
            feature,
            (engagement[i], conversion[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax4.set_xlabel("Engagement %")
    ax4.set_ylabel("Conversion %")
    ax4.set_title("Feature Impact", color=NETFLIX_COLORS["primary"])
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("good_usage.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_dashboard_breakdown():
    """Create detailed breakdown of bad dashboard elements"""
    # 1. Cluttered Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = [
        "Sales",
        "Costs",
        "Profit",
        "Inventory",
        "Staff",
        "Customer Count",
        "Returns",
        "Discounts",
        "Marketing",
        "Maintenance",
        "Utilities",
        "Rent",
    ]
    values = np.random.normal(100, 20, len(metrics))
    ax.bar(metrics, values, color="red")
    ax.set_title("Too Many Metrics - Bad Example")
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("bad_dashboard_metrics.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Inconsistent Colors
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Category A", "Category B", "Category C", "Category D"]
    values = np.random.normal(100, 20, len(categories))
    colors = ["red", "blue", "green", "purple"]
    ax.bar(categories, values, color=colors)
    ax.set_title("Inconsistent Color Scheme - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_dashboard_colors.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Poor Hierarchy
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["Sales", "Costs", "Profit", "Inventory"]
    values = np.random.normal(100, 20, len(metrics))
    for i, ax in enumerate(axes.flat):
        ax.bar(metrics, values, color="gray")
        ax.set_title(f"Equal Importance - Bad Example {i+1}")
    plt.tight_layout()
    plt.savefig("bad_dashboard_hierarchy.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_dashboard_breakdown():
    """Create detailed breakdown of good dashboard elements"""
    # 1. Focused Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = [
        "Daily Sales",
        "Customer Count",
        "Average Transaction",
        "Inventory Level",
    ]
    values = np.random.normal(100, 20, len(metrics))
    ax.bar(metrics, values, color=WALMART_COLORS["primary"])
    ax.set_title("Focused Key Metrics - Good Example", color=WALMART_COLORS["primary"])
    plt.tight_layout()
    plt.savefig("good_dashboard_metrics.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Consistent Colors
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Category A", "Category B", "Category C", "Category D"]
    values = np.random.normal(100, 20, len(categories))
    colors = [
        WALMART_COLORS["primary"],
        WALMART_COLORS["secondary"],
        WALMART_COLORS["accent"],
        WALMART_COLORS["neutral"],
    ]
    ax.bar(categories, values, color=colors)
    ax.set_title(
        "Consistent Color Scheme - Good Example", color=WALMART_COLORS["primary"]
    )
    plt.tight_layout()
    plt.savefig("good_dashboard_colors.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Clear Hierarchy
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["Sales", "Costs", "Profit", "Inventory"]
    values = np.random.normal(100, 20, len(metrics))
    sizes = [1.0, 0.8, 0.6, 0.4]
    for i, ax in enumerate(axes.flat):
        ax.bar(metrics, values, color=WALMART_COLORS["primary"], alpha=sizes[i])
        ax.set_title(
            f"Visual Hierarchy - Good Example {i+1}", color=WALMART_COLORS["primary"]
        )
    plt.tight_layout()
    plt.savefig("good_dashboard_hierarchy.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_journey_breakdown():
    """Create detailed breakdown of bad journey map elements"""
    # 1. Text-Heavy Explanation
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
    values = [100, 80, 60, 40, 20]
    ax.bar(steps, values, color="red")
    ax.text(
        0.5,
        0.5,
        "Long text explanation here...",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title("Text-Heavy Explanation - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_journey_text.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Missing Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
    values = [100, 80, 60, 40, 20]
    ax.bar(steps, values, color="blue")
    ax.set_title("Missing Key Metrics - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_journey_metrics.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. No Flow
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
    values = [100, 80, 60, 40, 20]
    ax.bar(steps, values, color="green")
    ax.set_title("No Visual Flow - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_journey_flow.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_journey_breakdown():
    """Create detailed breakdown of good journey map elements"""
    # 1. Visual Flow
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = ["Sign Up", "Profile", "Preferences", "Search", "Playlist", "Regular"]
    values = [100, 85, 70, 55, 45, 40]
    bars = ax.bar(steps, values, color=SPOTIFY_COLORS["primary"])
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{values[i]}%",
            ha="center",
            color=SPOTIFY_COLORS["secondary"],
        )
    ax.set_title(
        "Visual Flow with Metrics - Good Example", color=SPOTIFY_COLORS["secondary"]
    )
    plt.tight_layout()
    plt.savefig("good_journey_flow.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Clear Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = ["Sign Up", "Profile", "Preferences", "Search", "Playlist", "Regular"]
    values = [100, 85, 70, 55, 45, 40]
    bars = ax.bar(steps, values, color=SPOTIFY_COLORS["primary"])
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{values[i]}%",
            ha="center",
            color=SPOTIFY_COLORS["secondary"],
        )
        if i > 0:
            drop = values[i - 1] - values[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"-{drop}%",
                ha="center",
                color=SPOTIFY_COLORS["accent"],
            )
    ax.set_title(
        "Clear Metrics and Drop-offs - Good Example", color=SPOTIFY_COLORS["secondary"]
    )
    plt.tight_layout()
    plt.savefig("good_journey_metrics.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Actionable Insights
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = ["Sign Up", "Profile", "Preferences", "Search", "Playlist", "Regular"]
    values = [100, 85, 70, 55, 45, 40]
    bars = ax.bar(steps, values, color=SPOTIFY_COLORS["primary"])
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{values[i]}%",
            ha="center",
            color=SPOTIFY_COLORS["secondary"],
        )
        if i > 0 and values[i - 1] - values[i] > 20:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                "Critical Drop-off",
                ha="center",
                color=SPOTIFY_COLORS["error"],
            )
    ax.set_title(
        "Actionable Insights - Good Example", color=SPOTIFY_COLORS["secondary"]
    )
    plt.tight_layout()
    plt.savefig("good_journey_insights.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_campaign_breakdown():
    """Create detailed breakdown of bad campaign report elements"""
    # 1. Raw Data Dump
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = ["Email", "Social", "Search", "Display", "Direct"]
    metrics = ["Impressions", "Clicks", "Conversions", "Revenue"]
    data = np.random.normal(100, 20, (len(channels), len(metrics)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        xticklabels=metrics,
        yticklabels=channels,
        ax=ax,
    )
    ax.set_title("Raw Data Dump - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_campaign_data.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. No Story
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = ["Email", "Social", "Search", "Display", "Direct"]
    values = np.random.normal(100, 20, len(channels))
    ax.bar(channels, values, color="blue")
    ax.set_title("No Clear Story - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_campaign_story.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Missing Context
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = ["Email", "Social", "Search", "Display", "Direct"]
    values = np.random.normal(100, 20, len(channels))
    ax.bar(channels, values, color="green")
    ax.set_title("Missing Context - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_campaign_context.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_campaign_breakdown():
    """Create detailed breakdown of good campaign report elements"""
    # 1. Clear Narrative
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    channels = ["Facebook", "Instagram", "Google", "Email", "Direct"]
    impressions = np.array([120, 100, 80, 60, 40]) * 1000
    bookings = np.array([1200, 1000, 800, 600, 400])
    revenue = bookings * np.array([150, 120, 180, 90, 110])
    roi = (revenue - bookings * 50) / (bookings * 50) * 100

    # Channel performance
    axes[0, 0].bar(channels, roi, color=AIRBNB_COLORS["primary"])
    axes[0, 0].set_title("Channel Performance", color=AIRBNB_COLORS["primary"])
    axes[0, 0].set_ylabel("ROI %")

    # Conversion funnel
    funnel_stages = ["Impressions", "Clicks", "Bookings"]
    funnel_values = [
        impressions.sum() / 1000,
        impressions.sum() * 0.05 / 1000,
        bookings.sum(),
    ]
    axes[0, 1].bar(funnel_stages, funnel_values, color=AIRBNB_COLORS["secondary"])
    axes[0, 1].set_title("Conversion Funnel", color=AIRBNB_COLORS["primary"])

    # ROI by channel
    axes[1, 0].bar(channels, revenue, color=AIRBNB_COLORS["accent"])
    axes[1, 0].set_title("Revenue by Channel", color=AIRBNB_COLORS["primary"])
    axes[1, 0].set_ylabel("Revenue ($)")

    # Performance metrics
    conversion_rate = bookings / (impressions * 0.05) * 100
    axes[1, 1].scatter(
        roi,
        conversion_rate,
        s=revenue / 1000,
        c=range(len(channels)),
        cmap="RdYlBu",
        alpha=0.6,
    )
    axes[1, 1].set_title("Channel Efficiency", color=AIRBNB_COLORS["primary"])
    axes[1, 1].set_xlabel("ROI %")
    axes[1, 1].set_ylabel("Conversion Rate %")

    plt.tight_layout()
    plt.savefig("good_campaign_narrative.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Channel Comparisons
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = ["Facebook", "Instagram", "Google", "Email", "Direct"]
    values = np.random.normal(100, 20, len(channels))
    ax.bar(channels, values, color=AIRBNB_COLORS["primary"])
    ax.set_title("Channel Comparisons - Good Example", color=AIRBNB_COLORS["primary"])
    plt.tight_layout()
    plt.savefig("good_campaign_comparisons.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. ROI Focus
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = ["Facebook", "Instagram", "Google", "Email", "Direct"]
    values = np.random.normal(100, 20, len(channels))
    ax.bar(channels, values, color=AIRBNB_COLORS["secondary"])
    ax.set_title("ROI Focus - Good Example", color=AIRBNB_COLORS["primary"])
    plt.tight_layout()
    plt.savefig("good_campaign_roi.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_financial_breakdown():
    """Create detailed breakdown of bad financial report elements"""
    # 1. Too Many Numbers
    fig, ax = plt.subplots(figsize=(10, 6))
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    metrics = [
        "Revenue",
        "Costs",
        "Profit",
        "ROI",
        "EBITDA",
        "Cash Flow",
        "Working Capital",
        "Debt Ratio",
        "Current Ratio",
        "Quick Ratio",
    ]
    data = np.random.normal(100, 20, (len(quarters), len(metrics)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        xticklabels=metrics,
        yticklabels=quarters,
        ax=ax,
    )
    ax.set_title("Too Many Numbers - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_financial_numbers.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. No Visual Aids
    fig, ax = plt.subplots(figsize=(10, 6))
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    values = np.random.normal(100, 20, len(quarters))
    ax.bar(quarters, values, color="blue")
    ax.set_title("No Visual Aids - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_financial_visuals.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Missing Context
    fig, ax = plt.subplots(figsize=(10, 6))
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    values = np.random.normal(100, 20, len(quarters))
    ax.bar(quarters, values, color="green")
    ax.set_title("Missing Context - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_financial_context.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_financial_breakdown():
    """Create detailed breakdown of good financial report elements"""
    # 1. Key Metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    quarters = ["Q1'22", "Q2'22", "Q3'22", "Q4'22"]
    revenue = np.array([18.8, 16.9, 21.5, 24.3])
    margins = np.array([32.9, 27.9, 25.1, 23.8])
    production = np.array([305, 258.6, 365.9, 439.7])

    # Revenue trend
    axes[0, 0].plot(
        quarters, revenue, color=TESLA_COLORS["primary"], marker="o", linewidth=2
    )
    axes[0, 0].fill_between(quarters, revenue, alpha=0.2, color=TESLA_COLORS["primary"])
    axes[0, 0].set_title("Revenue Trend", color=TESLA_COLORS["primary"])
    axes[0, 0].set_ylabel("Revenue ($B)")

    # Profit margin
    axes[0, 1].bar(quarters, margins, color=TESLA_COLORS["primary"])
    axes[0, 1].set_title("Profit Margin", color=TESLA_COLORS["primary"])
    axes[0, 1].set_ylabel("Margin %")

    # Production trend
    axes[1, 0].plot(
        quarters, production, color=TESLA_COLORS["secondary"], marker="s", linewidth=2
    )
    axes[1, 0].fill_between(
        quarters, production, alpha=0.2, color=TESLA_COLORS["secondary"]
    )
    axes[1, 0].set_title("Production Trend", color=TESLA_COLORS["primary"])
    axes[1, 0].set_ylabel("Production (K)")

    # Key metrics
    metrics = ["Revenue\nGrowth", "Margin\nRetention", "Production\nEfficiency"]
    values = [29.1, 95.2, 87.5]
    colors = [
        TESLA_COLORS["primary"] if v >= 85 else TESLA_COLORS["secondary"]
        for v in values
    ]
    axes[1, 1].bar(metrics, values, color=colors)
    axes[1, 1].set_title("Key Performance Indicators", color=TESLA_COLORS["primary"])

    plt.tight_layout()
    plt.savefig("good_financial_metrics.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Visual Trends
    fig, ax = plt.subplots(figsize=(10, 6))
    quarters = ["Q1'22", "Q2'22", "Q3'22", "Q4'22"]
    values = np.random.normal(100, 20, len(quarters))
    ax.plot(quarters, values, color=TESLA_COLORS["primary"], marker="o", linewidth=2)
    ax.fill_between(quarters, values, alpha=0.2, color=TESLA_COLORS["primary"])
    ax.set_title("Visual Trends - Good Example", color=TESLA_COLORS["primary"])
    plt.tight_layout()
    plt.savefig("good_financial_trends.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Year-over-Year
    fig, ax = plt.subplots(figsize=(10, 6))
    quarters = ["Q1'22", "Q2'22", "Q3'22", "Q4'22"]
    current = np.array([100, 105, 110, 115])
    previous = np.array([98, 100, 102, 105])
    ax.plot(
        quarters, current, color=TESLA_COLORS["primary"], marker="o", label="This Year"
    )
    ax.plot(
        quarters,
        previous,
        color=TESLA_COLORS["secondary"],
        linestyle="--",
        marker="s",
        label="Previous Year",
    )
    ax.legend()
    ax.set_title(
        "Year-over-Year Comparison - Good Example", color=TESLA_COLORS["primary"]
    )
    plt.tight_layout()
    plt.savefig("good_financial_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bad_usage_breakdown():
    """Create detailed breakdown of bad usage analytics elements"""
    # 1. Raw Data Tables
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    metrics = [
        "Users",
        "Time",
        "Sessions",
        "Errors",
        "Bugs",
        "Support Tickets",
        "Complaints",
        "Feature Requests",
        "Abandonment",
        "Churn",
    ]
    data = np.random.normal(100, 20, (len(features), len(metrics)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        xticklabels=metrics,
        yticklabels=features,
        ax=ax,
    )
    ax.set_title("Raw Data Tables - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_usage_data.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. No Visualizations
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    values = np.random.normal(100, 20, len(features))
    ax.bar(features, values, color="blue")
    ax.set_title("No Visualizations - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_usage_visuals.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Missing Patterns
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    values = np.random.normal(100, 20, len(features))
    ax.bar(features, values, color="green")
    ax.set_title("Missing Patterns - Bad Example")
    plt.tight_layout()
    plt.savefig("bad_usage_patterns.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_good_usage_breakdown():
    """Create detailed breakdown of good usage analytics elements"""
    # 1. User Flow
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    features = ["Homepage", "Search", "Recommendations", "My List", "Continue Watching"]
    engagement = np.array([95, 45, 75, 60, 85])
    conversion = np.array([15, 35, 45, 55, 65])
    time_spent = np.array([2.5, 1.5, 4.5, 3.5, 5.5])

    # Feature usage
    axes[0, 0].bar(features, engagement, color=NETFLIX_COLORS["primary"])
    axes[0, 0].set_title("Users by Feature", color=NETFLIX_COLORS["primary"])
    axes[0, 0].set_ylabel("Users %")

    # Time spent
    axes[0, 1].bar(features, time_spent, color=NETFLIX_COLORS["secondary"])
    axes[0, 1].set_title("Time Spent", color=NETFLIX_COLORS["primary"])
    axes[0, 1].set_ylabel("Minutes")

    # Session frequency
    axes[1, 0].plot(
        features, conversion, color=NETFLIX_COLORS["primary"], marker="o", linewidth=2
    )
    axes[1, 0].fill_between(
        features, conversion, alpha=0.2, color=NETFLIX_COLORS["primary"]
    )
    axes[1, 0].set_title("Conversion Rate", color=NETFLIX_COLORS["primary"])
    axes[1, 0].set_ylabel("Conversion %")

    # Engagement matrix
    axes[1, 1].scatter(
        engagement,
        conversion,
        s=time_spent * 100,
        c=range(len(features)),
        cmap="RdYlBu",
        alpha=0.6,
    )
    axes[1, 1].set_title("Feature Impact", color=NETFLIX_COLORS["primary"])
    axes[1, 1].set_xlabel("Engagement %")
    axes[1, 1].set_ylabel("Conversion %")

    plt.tight_layout()
    plt.savefig("good_usage_flow.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Feature Usage
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ["Homepage", "Search", "Recommendations", "My List", "Continue Watching"]
    values = np.random.normal(100, 20, len(features))
    ax.bar(features, values, color=NETFLIX_COLORS["primary"])
    ax.set_title("Feature Usage - Good Example", color=NETFLIX_COLORS["primary"])
    plt.tight_layout()
    plt.savefig("good_usage_features.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Time Patterns
    fig, ax = plt.subplots(figsize=(10, 6))
    features = ["Homepage", "Search", "Recommendations", "My List", "Continue Watching"]
    values = np.random.normal(100, 20, len(features))
    ax.plot(features, values, color=NETFLIX_COLORS["primary"], marker="o", linewidth=2)
    ax.fill_between(features, values, alpha=0.2, color=NETFLIX_COLORS["primary"])
    ax.set_title("Time Patterns - Good Example", color=NETFLIX_COLORS["primary"])
    plt.tight_layout()
    plt.savefig("good_usage_patterns.png", bbox_inches="tight", dpi=300)
    plt.close()


def main():
    """Generate all case study visualizations"""
    # Create output directory if it doesn't exist
    output_dir = Path("docs/3-data-visualization/3.4-data-storytelling/assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate original visualizations
    create_bad_dashboard()
    create_good_dashboard()
    create_bad_journey()
    create_good_journey()
    create_bad_campaign()
    create_good_campaign()
    create_bad_financial()
    create_good_financial()
    create_bad_usage()
    create_good_usage()

    # Generate detailed breakdowns
    create_bad_dashboard_breakdown()
    create_good_dashboard_breakdown()
    create_bad_journey_breakdown()
    create_good_journey_breakdown()
    create_bad_campaign_breakdown()
    create_good_campaign_breakdown()
    create_bad_financial_breakdown()
    create_good_financial_breakdown()
    create_bad_usage_breakdown()
    create_good_usage_breakdown()


if __name__ == "__main__":
    main()
