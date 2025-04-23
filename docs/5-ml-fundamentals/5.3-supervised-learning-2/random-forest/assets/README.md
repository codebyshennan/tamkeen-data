# Random Forest Visualizations

This directory contains visualizations that help explain key concepts of Random Forests. The visualizations are generated using Python and can be recreated by running the `generate_visualizations.py` script.

## Visualizations

1. **Decision Tree Boundary** (`decision_tree_boundary.png`)
   - Shows how a single decision tree makes decisions
   - Demonstrates the piecewise linear nature of decision boundaries
   - Helps understand the basic building block of Random Forests

2. **Random Forest Boundary** (`random_forest_boundary.png`)
   - Shows how multiple trees work together to create a more complex decision boundary
   - Demonstrates the ensemble effect of Random Forests
   - Illustrates how the model can capture non-linear patterns

3. **Feature Importance** (`feature_importance.png`)
   - Shows the relative importance of different features in the model
   - Helps understand which features contribute most to predictions
   - Useful for feature selection and model interpretation

4. **Bias-Variance Tradeoff** (`bias_variance.png`)
   - Shows how model complexity affects predictions
   - Demonstrates the tradeoff between bias and variance
   - Helps understand the impact of tree depth on model performance

5. **Ensemble Prediction** (`ensemble_prediction.png`)
   - Shows how individual tree predictions combine to form the final prediction
   - Demonstrates the averaging effect of the ensemble
   - Illustrates how Random Forests reduce variance

## Generating Visualizations

To generate these visualizations:

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the visualization script:

   ```bash
   python generate_visualizations.py
   ```

The script will create all visualizations in the current directory.

## Usage in Documentation

These visualizations are used throughout the Random Forest documentation to illustrate key concepts. They are referenced in the markdown files using relative paths, for example:

```markdown
![Decision Tree Boundary](assets/decision_tree_boundary.png)
```

## Customization

You can modify the `generate_visualizations.py` script to:

- Change the style of the plots
- Adjust the parameters of the Random Forest models
- Create additional visualizations
- Modify the existing visualizations

## Dependencies

- numpy: For numerical computations
- matplotlib: For creating plots
- seaborn: For enhanced plotting styles
- scikit-learn: For Random Forest implementation
- pandas: For data manipulation
