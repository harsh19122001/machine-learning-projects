🌲 Random Forest Regression using Python

This project demonstrates the implementation of Random Forest Regression to predict salary based on experience using Python.

The objective of this project is to understand how ensemble-based regression models combine multiple decision trees to improve prediction accuracy and reduce overfitting.

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

The dataset shows a non-linear salary progression, making it suitable for Random Forest Regression.

🔍 Project Workflow
1️⃣ Data Loading & Preparation

Loaded the dataset using Pandas

Selected:

Experience (Level) as X

Salary as y

Split data into 80% training and 20% testing sets

2️⃣ Model Training with Different Parameters

Trained Random Forest models with:

n_estimators = 10, 50, 100, 200

Evaluated each model using R² score

Observed how increasing the number of trees affects performance

3️⃣ Depth Analysis

Trained models with max_depth values from 1 to 5

Compared R² scores to analyze underfitting vs overfitting

Selected an optimal depth for the final model

4️⃣ Final Model & Prediction

Trained the optimized RandomForestRegressor

Predicted salary for 6.5 years of experience

5️⃣ Visualization

Created a fine input grid for smooth plotting

Plotted:

Actual training data points

Smooth Random Forest regression curve

This visualization shows how Random Forest captures complex patterns more smoothly than a single decision tree.

📊 Model Evaluation

Mean Squared Error (MSE)

R² Score

These metrics were used to evaluate the performance of the final Random Forest model on unseen data.

🎯 What I Learned

How Random Forest Regression works internally

Role of ensemble learning in reducing overfitting

Effect of number of trees and depth on model performance

Difference between Decision Tree and Random Forest regression

Importance of hyperparameter tuning and evaluation

This project strengthened my understanding of ensemble-based machine learning models.
