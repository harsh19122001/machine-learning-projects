🔍 Logistic Regression for Customer Purchase Prediction

This project demonstrates the implementation of Logistic Regression to solve a binary classification problem where the goal is to predict whether a customer will purchase a product based on their age and estimated salary.

The project covers the full workflow from data preprocessing to evaluation and future prediction.

🛠️ Libraries Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn

📂 Dataset Description

The dataset contains 400 records and 5 columns:

User ID

Gender

Age

Estimated Salary

Purchased (Target Variable)

Only Age and Estimated Salary were used as input features for prediction.

🔍 Project Workflow
1️⃣ Data Loading & Feature Selection

Loaded CSV dataset using Pandas

Selected:

Age and Estimated Salary as X

Purchased as y

2️⃣ Train-Test Split

Split data into:

80% Training

20% Testing

3️⃣ Feature Scaling

Applied StandardScaler to normalize input features

Prevented bias caused by different value ranges

4️⃣ Logistic Regression Model

Trained LogisticRegression() classifier

Fitted model on scaled training data

5️⃣ Prediction & Evaluation
Model Predictions:

Generated predictions on test dataset

Evaluation Metrics:

Confusion Matrix

Accuracy Score (≈ 92.5%)

Classification Report:

Precision

Recall

F1-score

6️⃣ Future Prediction

Loaded new unseen dataset

Applied same feature scaling

Generated predictions

Saved results into final.csv file

📊 Results

The model achieved:

High classification accuracy

Strong precision and recall values

Effective prediction on new future data

🎯 What I Learned

How Logistic Regression works for classification

Importance of preprocessing and scaling

How to evaluate classifiers properly

How to apply trained models to new data

How to save and analyze prediction results

This project strengthened my understanding of supervised machine learning classification techniques.
