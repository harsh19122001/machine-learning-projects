💼 Salary Prediction App using Simple Linear Regression

This project demonstrates a Simple Linear Regression model used to predict salary based on years of experience, along with a Streamlit web app as the frontend.

The goal of this project is to understand:

how a regression model is trained,

how it is evaluated,

how it can be saved using pickle,

and how it can be used in a simple web application.

🛠️ Technologies & Libraries Used
Backend (Model Training)

Python

NumPy

Pandas

Matplotlib

Scikit-learn

Pickle

Frontend (Web App)

Streamlit

NumPy

Pickle

📂 Dataset Used

Salary_Data.csv

Features:

Years of Experience (Independent variable)

Salary (Dependent variable)

🔍 Backend: Model Training Steps

The following steps were performed in the backend code:

Loaded the salary dataset using Pandas

Split data into:

Independent variable (Years of Experience)

Dependent variable (Salary)

Split dataset into training and testing sets (80–20)

Trained a Simple Linear Regression model

Made predictions on test data

Visualized:

Training set results

Test set results

Predicted salary for:

12 years of experience

20 years of experience

Evaluated model performance using:

R² score (Training & Testing)

Mean Squared Error (MSE)

Saved the trained model using pickle

📊 Model Evaluation

Training Score (R²) is used to measure how well the model fits training data

Testing Score (R²) helps check model generalization

MSE is calculated for both training and testing data

These metrics help understand bias and variance of the model.

🖥️ Frontend: Streamlit App

The Streamlit app allows users to:

Enter years of experience

Click a button to predict salary

View predicted salary instantly

Features:

User-friendly interface

Real-time prediction using the trained model

Model loaded using pickle

📁 Project Structure
simple-linear-regression-salary-prediction/
│
├── Salary_Data.csv
├── linear_regression_model.pkl
├── backend_model_training.py
├── app.py
└── README.md

🎯 What I Learned

How Simple Linear Regression works

How to visualize regression results

How to evaluate a regression model

How to save and load ML models using pickle

How to integrate a trained model with Streamlit

How backend and frontend work together in ML projects

This project helped me strengthen my machine learning fundamentals and understand how models can be deployed in simple applications.
