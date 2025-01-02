# Mathematical Foundation of Regularization 📐

Let's dive into the mathematical concepts that make regularization work! Understanding these foundations will help you make better decisions when implementing and tuning your models.

## Loss Function with Regularization 🎯

### General Form
The regularized loss function is:

$$L_{reg}(\beta) = L(\beta) + \lambda R(\beta)$$

where:
- $L(\beta)$ is the original loss function
- $R(\beta)$ is the regularization term
- $\lambda$ is the regularization strength

## Types of Regularization Terms 🔍

### 1. L1 Regularization (Lasso)
$$R(\beta) = \sum_{j=1}^p |\beta_j|$$

Properties:
- Non-differentiable at zero
- Promotes sparsity
- Solution path is nonlinear

### 2. L2 Regularization (Ridge)
$$R(\beta) = \sum_{j=1}^p \beta_j^2$$

Properties:
- Differentiable everywhere
- Shrinks coefficients proportionally
- Has closed-form solution

### 3. Elastic Net
$$R(\beta) = \alpha\sum_{j=1}^p |\beta_j| + (1-\alpha)\sum_{j=1}^p \beta_j^2$$

Properties:
- Combines L1 and L2 properties
- $\alpha$ controls the mix
- More stable than pure L1

## Optimization Theory 🎓

### Gradient Descent with L2
For ridge regression:

$$\beta^{(t+1)} = \beta^{(t)} - \eta(\nabla L(\beta^{(t)}) + 2\lambda\beta^{(t)})$$

where:
- $\eta$ is the learning rate
- $t$ is the iteration number

### Proximal Gradient for L1
For lasso regression:

$$\beta^{(t+1)} = \text{prox}_{\lambda\eta}(\beta^{(t)} - \eta\nabla L(\beta^{(t)}))$$

where $\text{prox}$ is the soft-thresholding operator:

$$\text{prox}_{\lambda}(x) = \text{sign}(x)\max(|x|-\lambda, 0)$$

## Geometric Interpretation 🌐

### L1 Constraint Region
- Shape: Diamond (in 2D)
- Promotes sparsity due to corners
- Intersects axes at $\pm\frac{1}{\lambda}$

### L2 Constraint Region
- Shape: Circle (in 2D)
- Smooth boundary
- Radius determined by $\frac{1}{\sqrt{\lambda}}$

### Elastic Net Region
- Shape: Rounded diamond
- Combines properties of L1 and L2
- Controlled by mixing parameter $\alpha$

## Statistical Properties 📊

### Bias-Variance Tradeoff
For regularized estimator $\hat{\beta}$:

$$\text{MSE}(\hat{\beta}) = \text{Bias}(\hat{\beta})^2 + \text{Var}(\hat{\beta})$$

- Regularization increases bias
- But reduces variance
- Optimal $\lambda$ balances this tradeoff

### Degrees of Freedom
For ridge regression:

$$\text{df}(\lambda) = \text{tr}(X(X^TX + \lambda I)^{-1}X^T)$$

For lasso, degrees of freedom ≈ number of non-zero coefficients

## Theoretical Guarantees 🎯

### Oracle Properties
Under certain conditions, Lasso achieves:
1. Consistent variable selection
2. Asymptotic normality
3. Oracle property (performs as if true model known)

### Convergence Rates
For well-behaved problems:
- Ridge: $O(1/t)$ convergence rate
- Lasso: $O(1/\sqrt{t})$ convergence rate
- Elastic Net: Between $O(1/t)$ and $O(1/\sqrt{t})$

## Cross-Validation Theory 📉

### K-Fold CV Error
For regularization parameter $\lambda$:

$$\text{CV}(\lambda) = \frac{1}{K}\sum_{k=1}^K \text{MSE}_k(\lambda)$$

where $\text{MSE}_k$ is the error on fold $k$

### One Standard Error Rule
Choose largest $\lambda$ such that:

$$\text{CV}(\lambda) \leq \min_{\lambda'}\text{CV}(\lambda') + \text{SE}(\min_{\lambda'}\text{CV}(\lambda'))$$

## Regularization Path 🛣️

### Solution Path Equations
For ridge:
$$\hat{\beta}(\lambda) = (X^TX + \lambda I)^{-1}X^Ty$$

For lasso:
$$\hat{\beta}(\lambda) = \text{sign}(\hat{\beta}_{OLS})\max(|\hat{\beta}_{OLS}| - \lambda, 0)$$

where $\hat{\beta}_{OLS}$ is the ordinary least squares solution

## Next Steps 🚀

Now that you understand the mathematics behind regularization, let's move on to [Implementation](3-implementation.md) to see how to put these concepts into practice!
