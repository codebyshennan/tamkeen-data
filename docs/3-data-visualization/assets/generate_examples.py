import os

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use("fivethirtyeight")

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)


# Example 1: Line Chart - Daily Steps
def create_steps_chart():
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    steps = [8000, 7500, 9000, 8200, 8800, 6000, 5000]

    plt.figure(figsize=(10, 6))
    plt.plot(days, steps, marker="o", color="#2ecc71", linewidth=2)
    plt.title("My Daily Steps This Week", fontsize=14, pad=20)
    plt.ylabel("Steps", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("assets/daily_steps.png", dpi=300, bbox_inches="tight")
    plt.close()


# Example 2: Bar Chart - Ice Cream Preferences
def create_ice_cream_chart():
    flavors = ["Chocolate", "Vanilla", "Strawberry", "Mint"]
    preferences = [45, 30, 20, 15]

    plt.figure(figsize=(10, 6))
    plt.bar(flavors, preferences, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
    plt.title("Favorite Ice Cream Flavors", fontsize=14, pad=20)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig("assets/ice_cream_preferences.png", dpi=300, bbox_inches="tight")
    plt.close()


# Example 3: Pie Chart - Daily Activities
def create_daily_activities_chart():
    activities = ["Sleep", "Work", "Free Time", "Other"]
    hours = [8, 8, 5, 3]

    plt.figure(figsize=(10, 6))
    plt.pie(
        hours,
        labels=activities,
        autopct="%1.1f%%",
        colors=["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
    )
    plt.title("How I Spend My Day", fontsize=14, pad=20)
    plt.savefig("assets/daily_activities.png", dpi=300, bbox_inches="tight")
    plt.close()


# Example 4: Scatter Plot - Study Time vs. Grades
def create_study_grades_chart():
    study_time = np.random.normal(5, 1, 50)  # hours per day
    grades = 60 + 5 * study_time + np.random.normal(0, 5, 50)  # percentage

    plt.figure(figsize=(10, 6))
    plt.scatter(study_time, grades, alpha=0.6, color="#3498db")
    plt.title("Study Time vs. Grades", fontsize=14, pad=20)
    plt.xlabel("Study Time (hours/day)", fontsize=12)
    plt.ylabel("Grade (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("assets/study_grades.png", dpi=300, bbox_inches="tight")
    plt.close()


# Example 5: Before/After Transformation
def create_before_after_chart():
    # Generate some messy data
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 2, 100)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Before transformation
    ax1.scatter(x, y, alpha=0.6, color="#e74c3c")
    ax1.set_title("Before: Raw Data", fontsize=14)
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # After transformation (with trend line)
    ax2.scatter(x, y, alpha=0.6, color="#2ecc71")
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax2.plot(x, p(x), "r--", alpha=0.8)
    ax2.set_title("After: With Trend Line", fontsize=14)
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("Y", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("assets/before_after.png", dpi=300, bbox_inches="tight")
    plt.close()


# Generate all examples
if __name__ == "__main__":
    create_steps_chart()
    create_ice_cream_chart()
    create_daily_activities_chart()
    create_study_grades_chart()
    create_before_after_chart()
    print("Example visualizations have been generated in the assets directory!")
