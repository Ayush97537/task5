# task5

1. Data Preparation
- Loaded the heart disease dataset.
 
- Split data into training and test sets using `train_test_split`.
2. Train Decision Tree
 - Trained a **DecisionTreeClassifier** with controlled `max_depth` to avoid overfitting.
- Visualized the tree using `plot_tree()` from `sklearn.tree`.
  3. Train Random Forest
- Built a **RandomForestClassifier** with multiple decision trees.
- Compared its performance against the single decision tree.
  4. Feature Importance
- Used the `.feature_importances_` attribute to interpret which features are most influential.
- Visualized the importance scores with a bar plot.
 5. Model Evaluation
 
