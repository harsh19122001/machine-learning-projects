🚀 XGBoost Regression using Python

This project demonstrates the implementation of XGBoost Regression to predict salary based on experience and skill level using Python.

The objective of this project is to understand how boosting-based ensemble models improve regression performance through sequential learning and optimization.

🛠️ Libraries Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn

XGBoost

📂 Dataset Description

The dataset (salary_2d_dataset.csv) contains:

Experience (Years)

Skill Level

Salary (Target Variable)

This dataset represents a multi-feature non-linear salary trend.

🔍 Project Workflow
1️⃣ Data Loading & Preparation

Loaded CSV dataset using Pandas

Selected:

Experience and Skill Level as X

Salary as y

Split data into 80% training and 20% testing

2️⃣ Hyperparameter Tuning
Tree Depth Analysis:

Tested max_depth from 2 to 6

Evaluated each using R² score

Learning Rate Analysis:

Tested learning rates:

0.3

0.1

0.05

Compared performance to identify optimal configuration.

3️⃣ Final XGBoost Model

Trained optimized XGBRegressor

Generated predictions on test data

4️⃣ Prediction

Predicted salary for:

14 years experience & skill level 3

5️⃣ Feature Importance Analysis

Extracted feature importance values

Visualized importance using bar chart

Observed that experience influenced salary more strongly than skill level

📊 Model Evaluation

Mean Squared Error (MSE)

R² Score

Training Score (Bias)

Testing Score (Variance)

These metrics were used to measure performance and generalization.

🎯 What I Learned

How XGBoost regression works internally

Importance of boosting in ensemble learning

How hyperparameters affect model accuracy

How to interpret feature importance

How to evaluate regression models effectively

This project strengthened my understanding of advanced ensemble regression techniques.
