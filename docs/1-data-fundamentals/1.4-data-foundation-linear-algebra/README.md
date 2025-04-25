# Data Foundation with NumPy

## Why Learn NumPy?

Ever wondered how data scientists handle millions of numbers efficiently? Or how they perform complex calculations at lightning speed? Welcome to NumPy - your superpower for handling data in Python!

{% stepper %}
{% step %}

### The Power of NumPy

Think of NumPy as a turbocharger for Python:

- 100x faster than regular Python lists
- Efficient memory usage
- Powerful mathematical operations
- Perfect for data science and AI
{% endstep %}

{% step %}

### Real-World Applications

NumPy is everywhere:

- Data Analysis
- Machine Learning
- Image Processing
- Financial Analysis
{% endstep %}
{% endstepper %}

## Understanding Data Types

Let's break down data types in a way that's easy to understand!

### 1. Nominal Data (Names/Categories)

{% stepper %}
{% step %}

### What is it?

Like putting labels on things:

- Colors (Red, Blue, Green)
- Cuisines (Italian, Chinese, Mexican)
- T-shirt sizes (S, M, L, XL)
{% endstep %}

{% step %}

### Key Features

- No order (Blue isn't "more than" Red)
- Just categories
- Each item fits one category
{% endstep %}

{% step %}

### What You Can Do

 Count how many in each category
 Find most common (mode)
 Can't calculate average
 Can't find median
{% endstep %}
{% endstepper %}

### 2. Ordinal Data (Ordered Categories)

{% stepper %}
{% step %}

### What is it?

Categories with a clear order:

- ️ Spice levels (Mild → Medium → Hot)
- Ratings (1 star → 5 stars)
- Education (High School → Bachelor's → Master's)
{% endstep %}

{% step %}

### Key Features

- Has order (Hot is more than Mild)
- Spacing might not be equal
- Still categories, but ranked
{% endstep %}

{% step %}

### What You Can Do

 Everything from Nominal
 Compare (greater/less than)
 Find median
 Can't calculate average
{% endstep %}
{% endstepper %}

### 3. Interval Data (Equal Steps)

{% stepper %}
{% step %}

### What is it?

Numbers with equal steps between values:

- Temperature (Celsius/Fahrenheit)
- Calendar years
- IQ scores
{% endstep %}

{% step %}

### Key Features

- Equal spacing between values
- No true zero point
- Can add and subtract
{% endstep %}

{% step %}

### What You Can Do

 Everything from Ordinal
 Calculate average
 Find differences
 Can't say "twice as much"
{% endstep %}
{% endstepper %}

### 4. Ratio Data (True Zero)

{% stepper %}
{% step %}

### What is it?

Numbers with a meaningful zero:

- Height
- Weight
- Money
- Time
{% endstep %}

{% step %}

### Key Features

- Has true zero (0 height means no height)
- Can compare ratios
- Most flexible type
{% endstep %}

{% step %}

### What You Can Do

 Everything from Interval
 Multiply and divide
 Say "twice as much"
 All math operations
{% endstep %}
{% endstepper %}

## Quick Reference Guide

{% stepper %}
{% step %}

### Data Type Summary

```
Level     Order?   Equal Steps?   True Zero?   Example
Nominal                                 Colors
Ordinal                                 Ratings
Interval                                Temperature
Ratio                                   Height
```

{% endstep %}

{% step %}

### Categorical vs. Continuous

1. **Categorical** (Discrete)
   - Distinct groups
   - Like boxes to sort things
   - Example: T-shirt sizes

2. **Continuous** (Measurement)
   - Any value in a range
   - Like a ruler
   - Example: Height in cm
{% endstep %}
{% endstepper %}

## What We'll Learn

Get ready to master NumPy through these exciting topics:

1. **Introduction to NumPy**
   - Fast calculations
   - Efficient arrays

2. **NumPy ndarray**
   - Multi-dimensional arrays
   - Data organization

3. **ndarray Basics**
   - Creating arrays
   - Basic operations

4. **Boolean Indexing**
   - Filtering data
   - Conditional selection

5. **ndarray Methods**
   - Useful functions
   - Data manipulation

6. **Linear Algebra**
   - Matrix operations
   - Mathematical tools

 **Pro Tip**: Understanding data types is crucial because it determines what operations you can perform on your data!
