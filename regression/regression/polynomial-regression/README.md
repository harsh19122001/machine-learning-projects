📈 Polynomial Regression using Python

This project demonstrates the implementation of Polynomial Regression using Python to model a non-linear relationship between position level and salary.

The goal of this project is to understand:

when linear regression is not sufficient,

how polynomial regression works,

and how to select the best polynomial degree using error metrics.

🛠️ Libraries Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn

📂 Dataset Description

The dataset (emp_sal.csv) contains employee salary information with the following columns:

Position

Level (Independent Variable)

Salary (Dependent Variable)

The data shows a non-linear trend, making it suitable for polynomial regression.

🔍 Project Workflow
1️⃣ Data Loading & Preparation

Loaded the dataset using Pandas

Selected:

Level as independent variable (X)

Salary as dependent variable (y)

Split the dataset into 80% training and 20% testing sets

2️⃣ Linear Regression (Baseline Model)

Trained a Linear Regression model

Visualized:

scatter plot of training data

fitted linear regression line

Observed that linear regression does not fit the data well due to non-linearity

3️⃣ Polynomial Regression Implementation

Applied PolynomialFeatures with degrees 1 to 5

Transformed training and testing data for each degree

Trained a regression model for each polynomial degree

Calculated Mean Squared Error (MSE) for each degree

4️⃣ Model Evaluation & Degree Selection

Stored MSE values for each polynomial degree

Plotted MSE vs Polynomial Degree

Automatically selected the best degree based on the lowest MSE

Best polynomial degree selected: Degree = 5

5️⃣ Final Model & Visualization

Trained the final polynomial regression model using the best degree

Plotted:

polynomial regression curve

actual data points

Visualized how the polynomial model fits the non-linear data

6️⃣ Prediction

Predicted salary for a new position level:

Level = 6.5

Generated prediction using the final trained polynomial model

📊 Visualizations Included

Linear Regression fit graph

Polynomial Regression curve (best degree)

MSE vs Polynomial Degree plot

These visualizations help understand:

model behavior

error trends

effect of polynomial degree on performance

🎯 What I Learned

Difference between linear and non-linear relationships

Why linear regression may fail on complex data

How polynomial regression improves model fit

How to evaluate models using Mean Squared Error

How to select the best model using error analysis

Importance of visualization in regression problems

This project helped me strengthen my understanding of regression analysis and model evaluation.
