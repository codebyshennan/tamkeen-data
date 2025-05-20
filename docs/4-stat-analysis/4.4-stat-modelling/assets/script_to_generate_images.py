"""
This script generates images for the Statistical Modeling module.
Run this script to create the images referenced in the .md files.
"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Create assets directory if it doesn't exist
if not os.path.exists('assets'):
    os.makedirs('assets')

# Set a common style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def generate_polynomial_regression_images():
    """Generate images for polynomial regression visualization."""
    # Generate sample data with non-linear pattern
    np.random.seed(42)
    x = np.linspace(-3, 3, 100)
    y_true = x**2
    y = y_true + np.random.normal(0, 1, size=len(x))
    
    # Create dataset
    X = x.reshape(-1, 1)
    
    # Plot data points and true function
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, label='Data points')
    plt.plot(x, y_true, 'r-', label='True function (y = x²)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple Quadratic Function with Noise')
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/overfitting_data.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Fit models of different complexity
    degrees = [1, 2, 15]  # Linear, quadratic, and high-degree polynomial
    labels = ['Linear (Underfitting)', 'Quadratic (Good fit)', 'Degree 15 (Overfitting)']
    colors = ['blue', 'green', 'purple']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train.ravel(), y_train, c='black', alpha=0.7, label='Training data')
    plt.plot(x, y_true, 'r-', alpha=0.5, label='True function')
    
    # Fit models
    x_plot = np.linspace(-3.5, 3.5, 100).reshape(-1, 1)
    for i, degree in enumerate(degrees):
        # Create polynomial features
        poly = PolynomialFeatures(degree)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        
        # Make predictions for plot line
        X_plot_poly = poly.transform(x_plot)
        y_plot = model.predict(X_plot_poly)
        
        # Calculate errors
        train_error = mean_squared_error(y_train, model.predict(X_poly_train))
        test_error = mean_squared_error(y_test, model.predict(X_poly_test))
        
        # Plot
        plt.plot(x_plot, y_plot, c=colors[i], 
                 label=f'{labels[i]}\nTrain MSE: {train_error:.2f}, Test MSE: {test_error:.2f}')
    
    plt.title('Overfitting Example: Different Polynomial Degrees')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-5, 15)
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/overfitting_example.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate regularization comparison
    plt.figure(figsize=(15, 6))
    
    # Ridge Regression
    plt.subplot(121)
    for alpha in [0, 0.1, 1, 10]:
        model = Ridge(alpha=alpha)
        model.fit(X_poly_train, y_train)
        y_plot = model.predict(X_plot_poly)
        plt.plot(x_plot, y_plot, label=f'α={alpha}')
    
    plt.scatter(X_train.ravel(), y_train, alpha=0.3, color='black')
    plt.title('Ridge Regression (L2)')
    plt.xlabel('Feature Value')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    
    # Lasso Regression
    plt.subplot(122)
    for alpha in [0, 0.01, 0.1, 1]:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_poly_train, y_train)
        y_plot = model.predict(X_plot_poly)
        plt.plot(x_plot, y_plot, label=f'α={alpha}')
    
    plt.scatter(X_train.ravel(), y_train, alpha=0.3, color='black')
    plt.title('Lasso Regression (L1)')
    plt.xlabel('Feature Value')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('assets/regularization_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize constraint spaces
    plt.figure(figsize=(12, 6))
    
    # Generate coefficient space
    beta1 = np.linspace(-2, 2, 100)
    beta2 = np.linspace(-2, 2, 100)
    B1, B2 = np.meshgrid(beta1, beta2)
    
    # Calculate constraint regions
    l1 = np.abs(B1) + np.abs(B2)  # L1 constraint: |β1| + |β2| ≤ c
    l2 = B1**2 + B2**2            # L2 constraint: β1² + β2² ≤ c
    
    # L1 Constraint (Diamond)
    plt.subplot(121)
    plt.contour(B1, B2, l1, levels=[1], colors='r', linewidths=2)
    
    # Add loss function contours (circular contours representing MSE)
    for r in [0.4, 0.8, 1.2, 1.6]:
        plt.contour(B1, B2, (B1-1)**2 + (B2-0.5)**2, levels=[r**2], 
                   colors='blue', alpha=0.5, linestyles='--')
    
    # Highlight the corner intersection point
    plt.plot([1], [0], 'ko', markersize=8)
    
    plt.title('L1 Constraint (Diamond)')
    plt.xlabel('Coefficient β₁')
    plt.ylabel('Coefficient β₂')
    plt.axis('equal')
    plt.grid(True)
    plt.annotate('Sparse Solution\n(β₂ = 0)', xy=(1, 0), xytext=(1, -1.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # L2 Constraint (Circle)
    plt.subplot(122)
    plt.contour(B1, B2, l2, levels=[1], colors='b', linewidths=2)
    
    # Add the same loss function contours
    for r in [0.4, 0.8, 1.2, 1.6]:
        plt.contour(B1, B2, (B1-1)**2 + (B2-0.5)**2, levels=[r**2], 
                   colors='blue', alpha=0.5, linestyles='--')
    
    # Highlight the non-sparse intersection point
    plt.plot([0.9], [0.45], 'ko', markersize=8)
    
    plt.title('L2 Constraint (Circle)')
    plt.xlabel('Coefficient β₁')
    plt.ylabel('Coefficient β₂')
    plt.axis('equal')
    plt.grid(True)
    plt.annotate('Non-sparse Solution\n(both β₁ and β₂ ≠ 0)', 
                xy=(0.9, 0.45), xytext=(0.2, -1.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig('assets/constraint_spaces.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated polynomial regression and regularization images")


def generate_logistic_regression_images():
    """Generate images for logistic regression."""
    # Generate sample data for student exam results
    np.random.seed(42)
    n_samples = 100
    study_hours = np.random.normal(5, 2, n_samples)
    aptitude_scores = np.random.normal(65, 15, n_samples)
    
    # Create relationship (higher study hours & aptitude → higher passing probability)
    passing_probability = 1 / (1 + np.exp(-(0.75 * (study_hours - 5) + 0.02 * (aptitude_scores - 65))))
    passed = np.random.binomial(1, passing_probability)
    
    # Create DataFrame
    exam_data = pd.DataFrame({
        'StudyHours': study_hours,
        'AptitudeScore': aptitude_scores,
        'Passed': passed
    })
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(exam_data.StudyHours[exam_data.Passed == 1], 
                exam_data.AptitudeScore[exam_data.Passed == 1], 
                c='green', marker='+', s=100, label='Passed')
    plt.scatter(exam_data.StudyHours[exam_data.Passed == 0], 
                exam_data.AptitudeScore[exam_data.Passed == 0], 
                c='red', marker='x', s=100, label='Failed')
    plt.xlabel('Study Hours')
    plt.ylabel('Aptitude Score')
    plt.title('Exam Results Based on Study Hours and Aptitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/binary_classification_example.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize the logistic function
    x = np.linspace(-6, 6, 100)
    y = 1 / (1 + np.exp(-x))
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    
    # Add annotations
    plt.annotate('Almost Certain 0', xy=(-4, 0.02), xytext=(-5, 0.15),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Decision Boundary\np = 0.5', xy=(0, 0.5), xytext=(-2.5, 0.7),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Almost Certain 1', xy=(4, 0.98), xytext=(3, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Add reference lines
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    
    plt.title('The Logistic (Sigmoid) Function')
    plt.xlabel('z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ')
    plt.ylabel('Probability p(z)')
    plt.grid(True)
    plt.text(-5.5, 0.95, 'p(z) = 1 / (1 + e^(-z))', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig('assets/logistic_curve_annotated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot coefficient effects
    plt.figure(figsize=(12, 8))
    
    for label, coef in {
        'Strong Positive (β=2)': 2,
        'Weak Positive (β=0.5)': 0.5,
        'Strong Negative (β=-2)': -2,
        'Weak Negative (β=-0.5)': -0.5
    }.items():
        y = 1 / (1 + np.exp(-coef * x))
        plt.plot(x, y, linewidth=2, label=label)
    
    plt.title('Effect of Different Coefficients on Probability Curve')
    plt.xlabel('Feature Value')
    plt.ylabel('Probability of Positive Class')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    plt.savefig('assets/coefficient_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create odds ratio visualization
    coefficients = np.array([2.1, 0.8, 0.0, -0.5, -1.7])
    odds_ratios = np.exp(coefficients)
    feature_names = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
    
    # Calculate confidence intervals (for illustration)
    std_errors = np.array([0.3, 0.2, 0.15, 0.25, 0.4])
    ci_lower = np.exp(coefficients - 1.96 * std_errors)
    ci_upper = np.exp(coefficients + 1.96 * std_errors)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Odds_Ratio': odds_ratios,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })
    df = df.sort_values('Odds_Ratio')
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(df.Odds_Ratio, range(len(df)), 
                 xerr=[df.Odds_Ratio - df.CI_Lower, df.CI_Upper - df.Odds_Ratio],
                 fmt='o', capsize=5)
    
    plt.axvline(x=1, color='r', linestyle='--', label='No Effect Line')
    plt.yticks(range(len(df)), df.Feature)
    plt.xscale('log')  # Log scale makes interpretation easier
    plt.xlabel('Odds Ratio (log scale)')
    plt.title('Odds Ratios with 95% Confidence Intervals')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.text(0.2, -0.5, 'Decreases Odds', color='blue', fontsize=12)
    plt.text(2, -0.5, 'Increases Odds', color='blue', fontsize=12)
    plt.savefig('assets/odds_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create decision boundary visualization
    X = exam_data[['StudyHours', 'AptitudeScore']]
    y = exam_data['Passed']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train_scaled, y_train)
    
    # Create decision boundary visualization
    h = 0.05  # Step size in mesh
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Get scaled features for the entire dataset
    X_scaled = scaler.transform(X)
    
    # Make predictions on the mesh grid
    Z = log_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot the contour
    plt.figure(figsize=(10, 8))
    
    contour = plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)
    plt.colorbar(contour, label='Probability of Passing')
    
    # Plot decision boundary (p=0.5)
    plt.contour(xx, yy, Z, levels=[0.5], linestyles='dashed', colors='k')
    
    # Plot data points
    classes = ['Failed', 'Passed']
    colors = ['blue', 'red']
    for i, cls in enumerate(np.unique(y)):
        plt.scatter(X_scaled[y == cls, 0], X_scaled[y == cls, 1], 
                   c=colors[i], label=classes[i], edgecolor='k')
    
    plt.xlabel('Study Hours (Standardized)')
    plt.ylabel('Aptitude Score (Standardized)')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/logistic_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated logistic regression images")


def generate_model_interpretation_images():
    """Generate images for model interpretation."""
    # Create example dataset for housing prices
    np.random.seed(42)
    n_samples = 200
    sqft = np.random.normal(1500, 300, n_samples)  # Square footage
    age = np.random.normal(15, 7, n_samples)       # Home age in years
    distance_downtown = np.random.normal(3, 1, n_samples)  # Miles from downtown
    num_rooms = np.random.normal(3, 1, n_samples)  # Number of rooms
    
    # Generate house prices with specific coefficient relationships
    price = (
        100000 +                     # Base price
        120 * sqft +                 # Price increases with size
        -2000 * age +                # Price decreases with age
        -15000 * distance_downtown + # Price decreases with distance from downtown
        25000 * num_rooms +          # Price increases with rooms
        np.random.normal(0, 20000, n_samples)  # Random noise
    )
    
    # Create DataFrame
    housing_df = pd.DataFrame({
        'sqft': sqft,
        'age': age,
        'distance_downtown': distance_downtown,
        'num_rooms': num_rooms,
        'price': price
    })
    
    # Split features and target
    X = housing_df.drop('price', axis=1)
    y = housing_df['price']
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Create coefficient visualization
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in feature_importance['Coefficient']]
    plt.barh(feature_importance['Feature'], feature_importance['Absolute_Coefficient'], color=colors)
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance in Linear Regression')
    plt.grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Positive Effect (Increases Price)'),
        Patch(facecolor='red', label='Negative Effect (Decreases Price)')
    ]
    plt.legend(handles=legend_elements)
    plt.savefig('assets/coefficient_interpretation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate standardized coefficients
    scaler = StandardScaler()
    X_std_scaled = scaler.fit_transform(X)
    
    # Train a model on standardized features
    std_model = LinearRegression()
    std_model.fit(X_std_scaled, y)
    
    # Create DataFrame for standardized coefficients
    std_feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Standardized_Coefficient': std_model.coef_,
        'Original_Coefficient': model.coef_
    })
    std_feature_importance['Abs_Std_Coefficient'] = np.abs(std_feature_importance['Standardized_Coefficient'])
    std_feature_importance = std_feature_importance.sort_values('Abs_Std_Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in std_feature_importance['Standardized_Coefficient']]
    plt.barh(std_feature_importance['Feature'], std_feature_importance['Abs_Std_Coefficient'], color=colors)
    plt.xlabel('Absolute Standardized Coefficient')
    plt.title('Standardized Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements)
    plt.savefig('assets/standardized_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate random forest feature importance
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    rf_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    })
    rf_importance = rf_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(rf_importance['Feature'], rf_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance from Random Forest')
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create categorical feature example
    n_samples = 300
    np.random.seed(42)
    
    # Create numerical features
    income = np.random.normal(60000, 15000, n_samples)
    age = np.random.normal(35, 10, n_samples)
    
    # Create categorical features
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education = np.random.choice(education_levels, n_samples)
    
    marital_status = ['Single', 'Married', 'Divorced']
    marital = np.random.choice(marital_status, n_samples)
    
    # Effects for education level
    education_effect = {
        'High School': 0,
        'Bachelor': 5000,
        'Master': 10000, 
        'PhD': 15000
    }
    
    # Effects for marital status
    marital_effect = {
        'Single': 0,
        'Married': 8000,
        'Divorced': -3000
    }
    
    # Calculate loan amount
    loan_amount = 20000 + 0.2 * income + 100 * age
    
    # Add categorical effects
    for i in range(n_samples):
        loan_amount[i] += education_effect[education[i]] + marital_effect[marital[i]]
    
    # Add noise
    loan_amount += np.random.normal(0, 5000, n_samples)
    
    # Create DataFrame
    loan_df = pd.DataFrame({
        'Income': income,
        'Age': age,
        'Education': education,
        'MaritalStatus': marital,
        'LoanAmount': loan_amount
    })
    
    # Create dummy variables
    loan_data = pd.get_dummies(loan_df, columns=['Education', 'MaritalStatus'], drop_first=True)
    
    # Train model and plot categorical effects
    X_cat = loan_data.drop('LoanAmount', axis=1)
    y_cat = loan_data['LoanAmount']
    
    cat_model = LinearRegression()
    cat_model.fit(X_cat, y_cat)
    
    # Get categorical features
    num_features = ['Income', 'Age']
    cat_features = [f for f in X_cat.columns if f not in num_features]
    
    # Create DataFrame with coefficients
    cat_coef = pd.DataFrame({
        'Feature': X_cat.columns,
        'Coefficient': cat_model.coef_
    })
    
    # Plot categorical coefficients
    plt.figure(figsize=(12, 8))
    colors = ['green' if c > 0 else 'red' for c in cat_coef[cat_coef['Feature'].isin(cat_features)]['Coefficient']]
    plt.barh(
        cat_coef[cat_coef['Feature'].isin(cat_features)]['Feature'],
        cat_coef[cat_coef['Feature'].isin(cat_features)]['Coefficient'],
        color=colors
    )
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel('Coefficient Value')
    plt.title('Effect of Categorical Features on Loan Amount')
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/categorical_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate decision tree visualization
    tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_model.fit(X, y)
    
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, feature_names=X.columns, filled=True, rounded=True, fontsize=12)
    plt.title('Decision Tree Model for House Price Prediction')
    plt.savefig('assets/decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate tree feature importance
    tree_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': tree_model.feature_importances_
    })
    tree_importance = tree_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(tree_importance['Feature'], tree_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Decision Tree Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/tree_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation vs causation example
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated features
    temperature = np.random.normal(75, 10, n_samples)  # Temperature in F
    
    # Create ice cream sales (causally related to temperature)
    ice_cream_sales = 100 + 10 * (temperature - 75) / 10 + np.random.normal(0, 5, n_samples)
    
    # Create shorts wearing (correlated with temperature but not causally related to sales)
    shorts_wearing = (0.8 * (temperature - 75) / 10 + np.random.normal(0, 0.2, n_samples) > 0).astype(int)
    
    # Plot relationships
    plt.figure(figsize=(15, 5))
    
    # Temperature vs Ice Cream Sales
    plt.subplot(131)
    plt.scatter(temperature, ice_cream_sales, alpha=0.5)
    plt.xlabel('Temperature (F)')
    plt.ylabel('Ice Cream Sales ($)')
    plt.title('Temperature → Ice Cream Sales\n(Causal Relationship)')
    plt.grid(True, alpha=0.3)
    
    # Temperature vs Shorts Wearing
    plt.subplot(132)
    plt.scatter(temperature, shorts_wearing, alpha=0.5)
    plt.xlabel('Temperature (F)')
    plt.ylabel('Wearing Shorts (1=Yes, 0=No)')
    plt.title('Temperature → Shorts Wearing\n(Causal Relationship)')
    plt.grid(True, alpha=0.3)
    
    # Shorts Wearing vs Ice Cream Sales
    plt.subplot(133)
    plt.violinplot([ice_cream_sales[shorts_wearing==0], ice_cream_sales[shorts_wearing==1]], 
                  [0, 1], points=100)
    plt.xticks([0, 1], ['Not Wearing Shorts', 'Wearing Shorts'])
    plt.ylabel('Ice Cream Sales ($)')
    plt.title('Shorts Wearing → Ice Cream Sales\n(Correlation without Causation)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/correlation_vs_causation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create feature interaction example
    np.random.seed(42)
    n_samples = 500
    
    # Create features with interaction
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    
    # Create target with interaction effect
    target = feature1 * feature2 + np.random.normal(0, 0.2, n_samples)
    
    # Create DataFrame
    interaction_df = pd.DataFrame({
        'Feature1': feature1,
        'Feature2': feature2,
        'Target': target
    })
    
    # Train models and visualize the interaction
    plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax = plt.subplot(121, projection='3d')
    x1_range = np.linspace(min(feature1), max(feature1), 50)
    x2_range = np.linspace(min(feature2), max(feature2), 50)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    
    # Calculate predictions using the interaction term
    z_pred = x1_grid * x2_grid
    
    surf = ax.plot_surface(x1_grid, x2_grid, z_pred, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Predicted Target')
    ax.set_title('Interaction Effect Surface')
    
    # Contour plot
    plt.subplot(122)
    contour = plt.contourf(x1_grid, x2_grid, z_pred, cmap='viridis', levels=20)
    plt.colorbar(contour, label='Predicted Target')
    plt.scatter(feature1, feature2, c=target, alpha=0.5, edgecolors='w', linewidths=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Interaction Effect Contour')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/feature_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated model interpretation images")


def generate_model_selection_images():
    """Generate images for model selection."""
    # Generate data with varying complexity
    np.random.seed(42)
    n_samples = 100
    
    # Generate training and testing data
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_true = np.sin(X.ravel())
    
    # Add noise to create training data
    y_train = y_true + np.random.normal(0, 0.1, n_samples)
    
    # Create training and validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.3, random_state=42)
    
    # Create range of polynomial degrees
    degrees = range(1, 15)
    
    # Track errors
    train_errors = []
    val_errors = []
    
    # Fit models of different complexity
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_val = poly.transform(X_val)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        
        # Calculate errors
        train_pred = model.predict(X_poly_train)
        val_pred = model.predict(X_poly_val)
        
        train_error = mean_squared_error(y_train, train_pred)
        val_error = mean_squared_error(y_val, val_pred)
        
        train_errors.append(train_error)
        val_errors.append(val_error)
    
    # Plot the error curves
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, 'o-', label='Training error')
    plt.plot(degrees, val_errors, 'o-', label='Validation error')
    
    # Mark the best model
    best_degree = degrees[np.argmin(val_errors)]
    plt.axvline(x=best_degree, color='red', linestyle='--', 
               label=f'Best degree: {best_degree}')
    
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Selection: Finding Optimal Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/model_selection_error_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize cross-validation
    plt.figure(figsize=(12, 8))
    
    # Create example of different CV folds
    n_samples = 20
    n_folds = 5
    
    # Generate sample indices
    indices = np.arange(n_samples)
    
    # Plot the folds
    for i, (train_idx, val_idx) in enumerate(KFold(n_splits=n_folds).split(indices)):
        plt.subplot(n_folds, 1, i+1)
        plt.scatter(indices[train_idx], [i]*len(train_idx), c='blue', marker='o', s=50, label='Training set')
        plt.scatter(indices[val_idx], [i]*len(val_idx), c='red', marker='x', s=50, label='Validation set')
        plt.yticks([])
        plt.xlabel('Sample index' if i == n_folds-1 else '')
        plt.title(f'Fold {i+1}' if i == 0 else '')
    
    plt.tight_layout()
    plt.figtext(0.5, 0.01, 'Cross-Validation: Each fold provides a different train/validation split', 
               ha='center', fontsize=12)
    plt.savefig('assets/cross_validation_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bias-variance tradeoff visualization
    complexity = np.linspace(0, 10, 100)
    
    # Simulate error components
    bias = 1 / (complexity + 1)
    variance = complexity / 10
    total_error = bias + variance
    
    # Find optimal complexity
    optimal = complexity[np.argmin(total_error)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(complexity, bias, 'b-', label='Bias', linewidth=2)
    plt.plot(complexity, variance, 'r-', label='Variance', linewidth=2)
    plt.plot(complexity, total_error, 'g-', label='Total Error', linewidth=2)
    plt.axvline(x=optimal, color='k', linestyle='--', 
               label=f'Optimal Complexity: {optimal:.1f}')
    
    plt.xlabel('Model Complexity')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated model selection images")


if __name__ == "__main__":
    """Run all image generation functions."""
    # Make sure assets directory exists
    if not os.path.exists('assets'):
        os.makedirs('assets')
        
    # Generate all images
    generate_polynomial_regression_images()
    generate_logistic_regression_images()
    generate_model_interpretation_images()
    generate_model_selection_images()
    
    print("All images generated successfully!")
