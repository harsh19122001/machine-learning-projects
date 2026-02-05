⚡ LightGBM Regression for Salary Prediction

This project demonstrates the implementation of LightGBM Regression to predict salary based on experience and skill level using Python.

The objective is to understand how boosting-based ensemble models efficiently handle non-linear regression problems with multiple input features.

🛠️ Libraries Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn

LightGBM

📂 Dataset Description

The dataset (salary_2d_dataset.csv) contains:

Experience (Years)

Skill Level

Salary (Target Variable)

This dataset represents a multi-feature salary progression pattern.

🔍 Project Workflow
1️⃣ Data Loading & Preparation

Loaded dataset using Pandas

Selected:

Experience and Skill Level as X

Salary as y

Split dataset into 80% training and 20% testing

2️⃣ LightGBM Model Training

Implemented LGBMRegressor

Used:

n_estimators = 300

tuned max_depth

optimized learning_rate

Trained model on training data

3️⃣ Prediction

Generated predictions on test set

Predicted salary for:

21 years experience & skill level 3

4️⃣ Feature Importance Analysis

Extracted feature importance values from the trained model

Visualized importance using a bar chart

Observed that Experience contributed more strongly than Skill Level

📊 Model Evaluation

Mean Squared Error (MSE)

R² Score

These metrics were used to measure accuracy and generalization of the LightGBM model.

🎯 What I Learned

How LightGBM boosting works internally

Why gradient boosting performs well on structured datasets

How hyperparameters influence model performance

How feature importance improves model interpretability

How to evaluate ensemble regression models effectively

This project strengthened my understanding of advanced boosting-based machine learning algorithms.
