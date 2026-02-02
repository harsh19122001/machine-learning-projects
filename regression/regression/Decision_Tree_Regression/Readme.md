🌳 Decision Tree Regression using Python

This project demonstrates the implementation of Decision Tree Regression to predict salary based on experience using Python.

The objective of this project is to understand how tree-based regression models work and how they capture non-linear patterns by splitting data into decision rules.

🛠️ Libraries Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn

📂 Dataset Description

The dataset (emp_sal.csv) contains employee salary information with the following columns:

Position

Level / Experience (Independent Variable)

Salary (Dependent Variable)

The data shows a non-linear salary progression, making it suitable for decision tree regression.

🔍 Project Workflow
1️⃣ Data Loading & Preparation

Loaded the dataset using Pandas

Selected:

Experience/Level as X

Salary as y

Split the dataset into 80% training and 20% testing sets

2️⃣ Model Training with Different Tree Depths

Trained Decision Tree Regressor models with max_depth values from 1 to 5

Calculated R² score for each depth

Observed how increasing depth impacts model performance

3️⃣ Final Decision Tree Model

Selected an optimal tree configuration

Trained the final DecisionTreeRegressor

Generated predictions on test data

4️⃣ Prediction

Predicted salary for 6.5 years of experience using the trained model

5️⃣ Visualization

Created a fine grid for smoother visualization

Plotted:

Training data points

Step-wise regression output from the decision tree

This visualization highlights how decision trees create piecewise constant predictions.

📊 Model Evaluation

Mean Squared Error (MSE)

R² Score

These metrics were used to evaluate how well the model fits unseen data.

🎯 What I Learned

How decision tree regression works internally

Effect of tree depth on bias and variance

Why decision trees produce step-like predictions

Difference between tree-based and curve-based regression models

Importance of model evaluation using MSE and R²

This project strengthened my understanding of tree-based machine learning algorithms.
