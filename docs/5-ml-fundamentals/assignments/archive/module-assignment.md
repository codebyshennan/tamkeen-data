# Module 5: Machine Learning Fundamentals Assignment

## Instructions
This assignment consists of two parts:
1. Multiple choice questions to test your understanding of key concepts
2. Implementation questions to demonstrate practical skills

## Part A: Concept Check (Multiple Choice)

### Section 1: Machine Learning Fundamentals
1. What is machine learning?
   a. Algorithms that learn from data
   b. Programming explicit rules
   c. Data storage methods
   d. Statistical formulas

2. What is feature engineering?
   a. Creating new features from existing data
   b. Collecting new data
   c. Training models
   d. Testing algorithms

3. What is the bias-variance tradeoff?
   a. Balance between model complexity and generalization
   b. Balance between training and testing data
   c. Balance between features and samples
   d. Balance between accuracy and speed

4. What is cross-validation used for?
   a. Assessing model performance on unseen data
   b. Creating new features
   c. Cleaning data
   d. Selecting algorithms

5. What is the difference between supervised and unsupervised learning?
   a. Supervised uses labeled data, unsupervised doesn't
   b. Supervised is faster than unsupervised
   c. Supervised uses more data than unsupervised
   d. Supervised is more accurate than unsupervised

### Section 2: Supervised Learning
6. What is Naive Bayes based on?
   a. Bayes' theorem with independence assumption
   b. Neural networks
   c. Decision trees
   d. Linear algebra

7. What is the main idea behind k-Nearest Neighbors?
   a. Similar instances belong to same class
   b. Probability calculations
   c. Tree-based decisions
   d. Linear separation

8. What type of boundary does SVM create?
   a. Maximum margin hyperplane
   b. Probability threshold
   c. Tree-based splits
   d. Nearest neighbor regions

9. What is Random Forest?
   a. Ensemble of decision trees
   b. Single optimized tree
   c. Neural network type
   d. Clustering algorithm

10. What is dropout in neural networks?
    a. Regularization technique
    b. Activation function
    c. Loss function
    d. Optimization algorithm

### Section 3: Unsupervised Learning
11. What is PCA used for?
    a. Dimensionality reduction
    b. Classification
    c. Regression
    d. Clustering

12. What is t-SNE best for?
    a. Visualization of high-dimensional data
    b. Feature selection
    c. Prediction
    d. Model evaluation

13. What is k-means clustering?
    a. Partitioning data into k groups
    b. Reducing dimensions to k
    c. Selecting k features
    d. Training k models

14. What is hierarchical clustering?
    a. Creating tree of nested clusters
    b. Selecting hierarchy of features
    c. Training models in hierarchy
    d. Organizing data in trees

15. What is DBSCAN?
    a. Density-based clustering
    b. Distance-based clustering
    c. Distribution-based clustering
    d. Dimension-based clustering

## Part B: Implementation Tasks

### Task 1: Text Classification with Naive Bayes
Implement a text classifier using Naive Bayes for spam detection:
1. Preprocess text data (tokenization, stop words, etc.)
2. Implement Naive Bayes classifier from scratch
3. Compare with scikit-learn's implementation
4. Evaluate using appropriate metrics

### Task 2: Customer Segmentation
Perform customer segmentation using multiple techniques:
1. Use PCA for dimensionality reduction
2. Apply k-means clustering
3. Visualize results using t-SNE
4. Compare with hierarchical clustering
5. Provide business insights from the analysis

### Task 3: Model Evaluation Framework
Create a comprehensive evaluation framework:
1. Implement cross-validation with stratification
2. Handle imbalanced classes using appropriate techniques
3. Create visualization functions for:
   - ROC curves
   - Confusion matrices
   - Learning curves
   - Feature importance plots
4. Generate detailed performance reports

### Task 4: Hyperparameter Optimization
Implement and compare different optimization strategies:
1. Grid search with cross-validation
2. Random search
3. Bayesian optimization
4. Analyze trade-offs between methods
5. Visualize optimization results

## Submission Requirements
1. Code must be well-documented with comments
2. Include visualizations where appropriate
3. Provide analysis and interpretation of results
4. Submit both .py files and Jupyter notebooks
5. Include requirements.txt for dependencies

## Grading Criteria
- Multiple Choice Questions: 30%
- Implementation Tasks: 60%
- Code Quality & Documentation: 10%

## References
- Include citations for any external resources used
- Document any assumptions made
- Note any limitations in the implementation
