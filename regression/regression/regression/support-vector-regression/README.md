📈 Support Vector Regression (SVR) using Python

This project demonstrates the implementation of Support Vector Regression (SVR) to predict salary based on experience using Python.

The objective is to understand how SVR works and how it handles non-linear regression problems.

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

The data shows a non-linear trend, making it suitable for Support Vector Regression.

🔍 Project Workflow
1️⃣ Data Loading & Preparation

Loaded the dataset using Pandas

Selected:

Experience as X

Salary as y

Split the dataset into 80% training and 20% testing sets

2️⃣ Feature Scaling

Applied StandardScaler to both:

Independent variable (X)

Dependent variable (y)

Feature scaling was mandatory for SVR to perform correctly

3️⃣ SVR Model Training

Used SVR with RBF kernel

Set model parameters:

C = 100

gamma = 0.1

epsilon = 0.1

Trained the model using scaled training data

4️⃣ Prediction

Predicted salaries on test data

Inverse-transformed predicted values to original salary scale

Predicted salary for 6.5 years of experience

5️⃣ Visualization

Plotted actual data points

Visualized SVR regression curve using a fine grid

Compared real salary values with model predictions

📊 Visualizations Included

Salary vs Experience (SVR – Training Set)

Non-linear regression curve using RBF kernel

🎯 What I Learned

How Support Vector Regression works

Importance of feature scaling in SVR

Role of kernel functions in regression

How SVR differs from linear regression

How to visualize non-linear regression models

This project helped strengthen my understanding of advanced regression techniques.
