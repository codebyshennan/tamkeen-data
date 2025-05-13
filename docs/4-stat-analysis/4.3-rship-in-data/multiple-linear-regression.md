# Multiple Linear Regression: Prediction with Multiple Factors

Welcome to the next level of prediction! We've explored how one factor can predict an outcome using simple linear regression. Now we're taking a big step forward by learning how to use **multiple factors at once** to make even better predictions.

## What is Multiple Linear Regression?

In real life, outcomes are rarely influenced by just one factor. Think about it:

- Your exam score isn't just determined by study time, but also by sleep quality, previous knowledge, and stress levels
- House prices aren't just based on square footage, but also location, number of bedrooms, age of the house, and more
- A plant's growth isn't just affected by water, but also by sunlight, soil quality, and temperature

**Multiple linear regression (MLR) is like having a team of predictors instead of a single predictor.** It allows us to use several pieces of information to make more accurate predictions.

### The Family Recipe Analogy

Think of simple linear regression like trying to bake cookies with just flour. You can make something, but it won't be great. 

Multiple linear regression is like using a complete recipe with flour, sugar, butter, eggs, and vanilla. Each ingredient contributes to the final product, and the recipe tells you exactly how much of each to use!

### The Math (Don't Worry, We'll Explain It Simply!)

The formula looks like this:

\[ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \epsilon \]

This might look intimidating, but let's break it down:

- **Y** is what we're trying to predict (like exam score)
- **X₁, X₂, etc.** are our predictors (like hours studied, hours slept, previous knowledge)
- **β₀** is our starting point (intercept)
- **β₁, β₂, etc.** tell us how much each predictor contributes to our prediction
- **ε** represents the error (because no prediction is perfect)

### Real-World Example: Predicting House Prices

Imagine we want to predict house prices:
- X₁ might be the house size in square feet
- X₂ might be the number of bedrooms
- X₃ might be the house's age
- Y would be the house price

Our equation might look like:
Price = $50,000 + ($100 × SquareFeet) + ($5,000 × Bedrooms) - ($1,000 × Age)

This tells us:
- A house with zero square feet, zero bedrooms, and zero years old would cost $50,000 (not realistic, just a starting point!)
- Each square foot adds $100 to the price
- Each bedroom adds $5,000 to the price
- Each year of age reduces the price by $1,000

## Before We Start: Important Assumptions

Just like a recipe only works under certain conditions, multiple linear regression works best when certain assumptions are met. Let's understand these in simple terms:

### 1. The Relationships Should Be Linear

Each predictor should have a straight-line relationship with what we're predicting.

**Garden Analogy**: If doubling the amount of water always roughly doubles plant growth, that's linear. If too much water starts drowning the plant, that's non-linear.

### 2. Each Observation Should Be Independent

One data point shouldn't influence another.

**Classroom Analogy**: Students should take tests independently. If they copy from each other, their scores are no longer independent, and our analysis won't be valid.

### 3. The Spread Should Be Consistent Throughout (Homoscedasticity)

The amount of "error" or "noise" in our predictions should be similar across all values.

**Weather Forecast Analogy**: A good forecasting system should be equally reliable whether predicting temperatures for summer or winter, not more accurate in one season.

### 4. The Errors Should Follow a Normal Distribution

The mistakes in our predictions should follow a bell curve pattern.

**Archery Analogy**: Most arrows land close to the bullseye, with fewer arrows landing farther away.

### 5. Predictors Shouldn't Be Too Similar (No Multicollinearity)

The factors you use should be relatively independent of each other.

**Recipe Analogy**: Adding both butter and margarine to a recipe might not give you independent effects since they serve similar purposes. Using butter and sugar would be better since they contribute differently.

## Building Your First Multiple Regression Model

Let's walk through a concrete example using Python. Don't worry if the code looks complex - focus on understanding the concepts!

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create example data for student exam scores
np.random.seed(42)  # For reproducible results
n_samples = 100  # 100 students

# Create factors that might affect exam scores
study_hours = np.random.normal(0, 1, n_samples)  # Hours spent studying
prev_gpa = np.random.normal(0, 1, n_samples)     # Previous GPA
sleep_hours = np.random.normal(0, 1, n_samples)  # Hours of sleep before exam

# Create exam scores based on these factors
# Notice each factor has a different weight (2, 3, and 1.5)
exam_scores = 2*study_hours + 3*prev_gpa + 1.5*sleep_hours + np.random.normal(0, 1, n_samples)

# Put everything in a nice table (DataFrame)
student_data = pd.DataFrame({
    'study_hours': study_hours,
    'prev_gpa': prev_gpa,
    'sleep_hours': sleep_hours,
    'exam_score': exam_scores
})

# Create and fit our model
X = student_data[['study_hours', 'prev_gpa', 'sleep_hours']]  # Predictors
y = student_data['exam_score']  # What we're predicting

model = LinearRegression()
model.fit(X, y)

# Print results
print("Contribution of each factor:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.2f} points")
print(f"\nStarting point (intercept): {model.intercept_:.2f}")
print(f"Model accuracy (R-squared): {model.score(X, y):.2f}")

# Check for predictor similarity (multicollinearity)
def check_predictor_similarity(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data

print("\nMulticollinearity Check (VIF values):")
print(check_predictor_similarity(X))
```
```
Contribution of each factor:
study_hours: 1.82 points
prev_gpa: 2.96 points
sleep_hours: 1.53 points

Starting point (intercept): 0.09
Model accuracy (R-squared): 0.94

Multicollinearity Check (VIF values):
      Variable       VIF
0  study_hours  1.053354
1     prev_gpa  1.019570
2  sleep_hours  1.034520
```

### Understanding the Results

Let's interpret what our model is telling us:

1. **Contribution of each factor**:
   - Each additional hour studied adds 1.82 points to the exam score
   - Each point of previous GPA adds 2.96 points
   - Each additional hour of sleep adds 1.53 points

2. **Starting point (0.09)**:
   - The baseline score is nearly zero (just a mathematical starting point)

3. **Model accuracy (0.94)**:
   - Our model explains 94% of the variation in exam scores (that's excellent!)
   - Only 6% is due to factors we haven't included or random chance

4. **Multicollinearity check**:
   - All VIF values are close to 1, which is great!
   - This means our predictors aren't too similar to each other
   - Rule of thumb: VIF values above 10 indicate problematic similarity

### Visualizing the Model

It's harder to visualize a multiple regression model because it exists in more than two dimensions. However, we can look at the relationship between predicted scores and actual scores:

![Multiple Regression Prediction](assets/multiple-regression-prediction.png)

We can also see how our exam score relates to each individual predictor while holding the others constant:

![Partial Regression Plots](assets/partial-regression-plots.png)

## Choosing the Right Predictors

One of the biggest challenges in multiple regression is deciding which factors to include in your model. There are three main approaches:

### 1. Knowledge-Based Selection

This is when you use your understanding of the subject to choose predictors.

**Example**: If predicting crop yield, you might include rainfall, temperature, soil quality, and fertilizer use because agricultural science tells us these factors matter.

### 2. Statistical Selection

You can let the numbers guide you by including only statistically significant predictors.

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Find the two strongest predictors
selector = SelectKBest(score_func=f_regression, k=2)
X_selected = selector.fit_transform(X, y)

# See which ones were selected
selected_features = X.columns[selector.get_support()].tolist()
print("\nStatistically strongest features:", selected_features)
```
```
Statistically strongest features: ['prev_gpa', 'sleep_hours']
```

### 3. Stepwise Selection

This is like building a team one player at a time - you add predictors one by one, keeping only those that improve the model.

```python
from sklearn.feature_selection import RFE

# Recursively eliminate features
selector = RFE(estimator=model, n_features_to_select=2)
selector = selector.fit(X, y)

# See which ones were kept
selected_features = X.columns[selector.support_].tolist()
print("\nFeatures selected by stepwise method:", selected_features)
```
```
Features selected by stepwise method: ['study_hours', 'prev_gpa']
```

Notice how different methods can select different predictors! This shows why it's important to combine statistical methods with subject knowledge.

## Checking If Your Model Is Valid

Just like we check a car before a long journey, we should check our model before relying on its predictions. Here are some key diagnostics:

```python
# Function for diagnostic plots
def check_model_validity(model, X, y):
    # Make predictions
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Residuals vs Fitted (checks linearity)
    axes[0,0].scatter(y_pred, residuals)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_xlabel('Predicted values')
    axes[0,0].set_ylabel('Errors (residuals)')
    axes[0,0].set_title('Residuals vs Fitted (should be random scatter)')
    
    # 2. Q-Q plot (checks normality)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Normal Q-Q (points should follow diagonal)')
    
    # 3. Scale-Location (checks equal variance)
    axes[1,0].scatter(y_pred, np.abs(residuals))
    axes[1,0].set_xlabel('Predicted values')
    axes[1,0].set_ylabel('|Errors|')
    axes[1,0].set_title('Scale-Location (spread should be even)')
    
    # 4. Correlation matrix (checks multicollinearity)
    corr = X.corr()
    sns.heatmap(corr, ax=axes[1,1], annot=True, cmap='coolwarm')
    axes[1,1].set_title('Correlation Matrix (should not have values near 1)')
    
    plt.tight_layout()
    plt.show()

# Check our model
check_model_validity(model, X, y)
```

![multiple-linear-regression_fig_4](assets/multiple-linear-regression_fig_4.png)

### What Good Diagnostic Plots Look Like:

1. **Residuals vs Fitted (top left)**:
   - Should look like a random cloud of points around the zero line
   - No patterns or curves should be visible
   
2. **Q-Q Plot (top right)**:
   - Points should follow the diagonal line closely
   - Significant deviations suggest non-normal errors
   
3. **Scale-Location (bottom left)**:
   - Should show a relatively even spread across all predicted values
   - A funnel shape suggests the errors aren't consistent
   
4. **Correlation Matrix (bottom right)**:
   - Shows the relationships between predictors
   - Values close to 1 or -1 indicate potential multicollinearity

## Real-World Applications

Multiple linear regression is an incredibly versatile tool used across many fields:

### Business & Marketing
- **Sales Forecasting**: Predicting sales based on advertising budget, price, competitor activity, and seasonality
- **Customer Value**: Estimating lifetime customer value based on demographics, purchase history, and engagement metrics
- **Pricing Strategy**: Determining optimal pricing by analyzing price sensitivity, competitor prices, and product features

### Healthcare
- **Patient Risk Assessment**: Predicting disease risk based on age, family history, lifestyle factors, and biomarkers
- **Hospital Resource Planning**: Estimating length of stay based on diagnosis, treatment, age, and comorbidities
- **Treatment Effectiveness**: Analyzing how different factors affect treatment outcomes

### Real Estate
- **Property Valuation**: Estimating house prices based on size, location, age, number of rooms, and nearby amenities
- **Investment Analysis**: Predicting return on investment based on property characteristics, location trends, and economic indicators
- **Rental Price Optimization**: Setting optimal rent prices based on unit features, location, and market demand

### Environmental Science
- **Climate Modeling**: Understanding how different factors contribute to temperature changes
- **Pollution Prediction**: Forecasting air quality based on traffic volume, industrial activity, and weather conditions
- **Resource Management**: Predicting water usage based on population, season, and weather patterns

## Common Challenges and How to Address Them

Even with a powerful tool like multiple regression, there are pitfalls to watch out for:

### 1. Overfitting: Including Too Many Predictors

**The Problem**: Adding too many predictors can make your model fit the training data perfectly but perform poorly on new data.

**The Analogy**: It's like memorizing exact answers to practice questions rather than understanding the underlying concepts. You'll do great on those specific questions but fail when given new ones.

**The Solution**: Use techniques like cross-validation (testing on data the model hasn't seen) and regularization (penalizing complex models).

### 2. Multicollinearity: When Predictors Are Too Similar

**The Problem**: When predictors are highly correlated, it becomes difficult to determine their individual effects.

**The Analogy**: Imagine trying to determine which twin ate the cookies when they're always together at the same time. It's impossible to separate their effects!

**The Solution**: Check VIF values (like we did earlier) and remove or combine highly correlated predictors.

### 3. Missing Important Variables

**The Problem**: Omitting key predictors can lead to biased results and poor predictions.

**The Analogy**: Trying to predict crop yield while considering rainfall and sunlight, but forgetting about soil quality.

**The Solution**: Use domain knowledge, literature review, and exploratory data analysis to identify potentially important predictors.

### 4. Extrapolation: Predicting Beyond Your Data Range

**The Problem**: Using your model to predict for values far outside your original data range can be unreliable.

**The Analogy**: Testing a car's performance at 30-60 mph, then assuming it will behave similarly at 200 mph.

**The Solution**: Be cautious with predictions outside the range of your training data and clearly communicate the limitations of your model.

## Hands-On Practice: Sales Prediction Exercise

Try working through this example to solidify your understanding:

```python
# Generate a realistic sales dataset
np.random.seed(42)
n = 100  # 100 observations

# Create predictors
advertising = np.random.uniform(10, 100, n)  # Advertising spend ($1000s)
price = np.random.uniform(50, 200, n)        # Product price ($)
competition = np.random.uniform(1, 10, n)    # Number of competitors

# Create sales (dependent variable)
# Note: Advertising has positive effect, price and competition have negative effects
sales = (3 * advertising - 2 * price - competition + 
        np.random.normal(0, 20, n))  # Sales in units

# Create DataFrame
data = pd.DataFrame({
    'advertising': advertising,
    'price': price,
    'competition': competition,
    'sales': sales
})

# Your Tasks:
# 1. Create scatter plots between each predictor and sales
# 2. Check for multicollinearity between predictors
# 3. Fit a multiple regression model
# 4. Interpret the coefficients (what do they mean for business decisions?)
# 5. Check model assumptions using diagnostic plots
# 6. Make predictions for a new scenario:
#    - $80,000 advertising budget
#    - $120 price
#    - 5 competitors
```

### What You Should Find:

- **Advertising** should have a positive coefficient (more advertising = more sales)
- **Price** should have a negative coefficient (higher price = fewer sales)
- **Competition** should have a negative coefficient (more competition = fewer sales)
- The model should capture these relationships well with an R-squared of around 0.85-0.95

## Key Points to Remember

1. Multiple linear regression lets us predict an outcome using several predictors at once
2. Each predictor gets its own coefficient showing its unique contribution
3. R-squared tells us how much of the variation our model explains
4. Always check model assumptions using diagnostic plots
5. Watch out for common issues: overfitting, multicollinearity, missing variables, and extrapolation
6. The best models combine statistical methods with subject-matter knowledge

## Where to Go From Here

Now that you understand multiple linear regression, you can explore:

1. **Polynomial Regression**: Adding squared or cubed terms to model curved relationships
2. **Interaction Terms**: When the effect of one variable depends on another
3. **Regularization Techniques**: Ridge, Lasso, and Elastic Net for handling many predictors
4. **Categorical Variables**: Using dummy variables to include non-numeric factors

## Helpful Resources for Learning More

- [StatQuest Videos](https://www.youtube.com/c/joshstarmer) - Excellent visual explanations of regression concepts
- [Khan Academy's Multiple Regression Course](https://www.khanacademy.org/math/statistics-probability/advanced-regression-inference-transformations)
- [An Introduction to Statistical Learning](https://www.statlearning.com/) - Free online textbook with accessible explanations
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html) - For when you're ready to implement more advanced models
- [Perplexity AI](https://www.perplexity.ai/) - For quick answers to specific questions
