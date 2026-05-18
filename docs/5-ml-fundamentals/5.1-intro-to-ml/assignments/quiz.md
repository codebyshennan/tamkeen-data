# Quiz: Introduction to Machine Learning

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

Try each question closed-book first. Click **Show hint** if you get stuck — hints point you at the relevant lesson section and how to think about the question, without naming the answer.

## Questions

1. In supervised learning, what distinguishes the training data from the inputs used in unsupervised learning?

- [ ] Supervised data is always larger
- [ ] Supervised data includes labeled examples paired with correct outputs
- [ ] Supervised data requires no preprocessing
- [ ] Supervised data is collected in real time

<details>
<summary>Show hint</summary>

- **Where:** [What is ML?](../what-is-ml.md) — "Types of Machine Learning → Supervised Learning".
- **Think:** The word "supervised" implies guidance — something tells the algorithm what the right answer is for each training example. Which option captures that idea?

</details>

2. Which of the following tasks is best described as an unsupervised learning problem?

- [ ] Predicting whether an email is spam based on labeled examples
- [ ] Grouping customers into segments based on purchasing behavior, with no predefined categories
- [ ] Training a robot with reward signals to navigate a maze
- [ ] Classifying images into cat or dog using annotated photos

<details>
<summary>Show hint</summary>

- **Where:** [What is ML?](../what-is-ml.md) — "Types of Machine Learning → Unsupervised Learning".
- **Think:** Unsupervised learning finds patterns without any labels. Which option describes discovery rather than prediction from known examples?

</details>

3. A game-playing AI learns by receiving points when it wins and losing points when it loses. Which learning paradigm does this describe?

- [ ] Supervised learning
- [ ] Unsupervised learning
- [ ] Reinforcement learning
- [ ] Semi-supervised learning

<details>
<summary>Show hint</summary>

- **Where:** [What is ML?](../what-is-ml.md) — "Types of Machine Learning → Reinforcement Learning".
- **Think:** This paradigm involves an agent, an environment, and a feedback signal that is not a pre-labeled dataset. Which term names that feedback loop?

</details>

4. What is the correct order of the machine learning workflow steps as described in the lesson?

- [ ] Model Training → Data Collection → Problem Definition → Model Evaluation → Deployment
- [ ] Problem Definition → Data Collection → Data Preparation → Model Selection → Model Training → Model Evaluation → Deployment
- [ ] Data Preparation → Problem Definition → Model Training → Data Collection → Deployment
- [ ] Data Collection → Model Training → Problem Definition → Model Evaluation → Data Preparation

<details>
<summary>Show hint</summary>

- **Where:** [ML Workflow](../ml-workflow.md) — "The Machine Learning Workflow Steps".
- **Think:** The workflow is linear: you must understand the problem before you collect, clean before you model, and evaluate before you deploy. Which sequence preserves that logical order?

</details>

5. During which workflow step would you compute a correlation heatmap and inspect the distribution of the target variable?

- [ ] Problem Definition
- [ ] Data Collection and Exploration
- [ ] Model Selection and Training
- [ ] Model Deployment

<details>
<summary>Show hint</summary>

- **Where:** [ML Workflow](../ml-workflow.md) — "2. Data Collection and Exploration → Exploratory Data Analysis".
- **Think:** Correlation heatmaps and target distributions are EDA activities. They happen after you have data but before you touch any model.

</details>

6. You train a model and observe that training accuracy is 98% but validation accuracy is 62%. What does this gap most likely indicate?

- [ ] Underfitting — the model is too simple
- [ ] A data collection error
- [ ] Overfitting — the model has memorized the training data
- [ ] A correct and expected outcome

<details>
<summary>Show hint</summary>

- **Where:** [Bias and Variance](../bias-variance.md) — "High Variance (Overfitting)".
- **Think:** High training score and low validation score means the model performs very differently on seen vs unseen data. Which failure mode is characterised by a large training–validation gap?

</details>

7. Which of the following correctly describes a model that suffers from high bias?

- [ ] It performs well on training data but poorly on validation data
- [ ] It has a large gap between training and cross-validation scores
- [ ] Both training and validation scores are low, with a small gap between them
- [ ] It changes predictions dramatically when trained on different subsets

<details>
<summary>Show hint</summary>

- **Where:** [Bias and Variance](../bias-variance.md) — "Interpreting Learning Curves → High Bias (Underfitting)".
- **Think:** High bias means the model is too simple and misses patterns consistently — even on its own training data. Which description matches that outcome?

</details>

8. Adding polynomial features to a linear model is a strategy for addressing which problem?

- [ ] High variance (overfitting)
- [ ] High bias (underfitting)
- [ ] Data leakage
- [ ] Class imbalance

<details>
<summary>Show hint</summary>

- **Where:** [Bias and Variance](../bias-variance.md) — "Dealing with High Bias → Increase Model Complexity".
- **Think:** Polynomial features let a model follow curved patterns it couldn't before — they make the hypothesis space richer. What failure mode does a richer hypothesis space fix?

</details>

9. You plot learning curves and see that both training and validation scores converge to a reasonably high value as you add more data. What does this indicate?

- [ ] The model is overfitting
- [ ] The model is underfitting
- [ ] The model has a good fit
- [ ] The model needs more features

<details>
<summary>Show hint</summary>

- **Where:** [Bias and Variance](../bias-variance.md) — "Interpreting Learning Curves → Good Fit".
- **Think:** Convergence of both curves at a high score is the signature of a well-calibrated model. None of the other options apply when both scores are high and close together.

</details>

10. What is the primary purpose of feature scaling (e.g., StandardScaler) before training a machine learning model?

- [ ] To remove missing values from the dataset
- [ ] To ensure all features are on a comparable numeric scale so no single feature dominates
- [ ] To encode categorical variables as integers
- [ ] To split the dataset into train and test sets

<details>
<summary>Show hint</summary>

- **Where:** [Feature Engineering](../feature-engineering.md) — "Scaling and Normalization → Why Scaling Matters".
- **Think:** Algorithms that rely on distances or gradient magnitudes behave poorly when one feature is measured in thousands and another in fractions. What does scaling fix?

</details>

11. A feature column contains city names such as "London", "Paris", and "Berlin". Which encoding strategy is most appropriate?

- [ ] StandardScaler
- [ ] Label encoding (assign London=0, Paris=1, Berlin=2)
- [ ] One-hot encoding with `pd.get_dummies`
- [ ] Log transformation

<details>
<summary>Show hint</summary>

- **Where:** [Feature Engineering](../feature-engineering.md) — "Handling Categorical Variables → When to Use Which Encoding Method".
- **Think:** City names have no inherent numeric order. Which encoding avoids implying a false ordering between categories?

</details>

12. Why must a `StandardScaler` be fitted on training data only, and then used to transform both training and test data (rather than being fitted on all data at once)?

- [ ] Because scikit-learn's API does not allow fitting on combined data
- [ ] To prevent test distribution information from leaking into the training process and inflating evaluation scores
- [ ] Because the scaler cannot handle the full dataset in memory
- [ ] To ensure the training set is always larger than the test set

<details>
<summary>Show hint</summary>

- **Where:** [Feature Engineering](../feature-engineering.md) — "Gotchas: Fitting the scaler on the full dataset before splitting".
- **Think:** The scaler computes statistics (mean, std) during `fit`. If you fit on all data, those statistics encode information from test rows. What does that do to your evaluation?

</details>
