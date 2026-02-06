🏠 Housing Price Prediction using Multiple Machine Learning Models

This project is an end-to-end machine learning system that predicts house prices using multiple regression algorithms and provides an interactive Flask web application for real-time predictions.

The goal of this project is to:

compare different ML regression models,

evaluate their performance,

and deploy them into a working web interface.

🛠️ Technologies & Libraries Used
Backend (Model Training & Evaluation)

Python

Pandas

Scikit-learn

LightGBM

XGBoost

Pickle

Frontend (Deployment)

Flask

HTML Templates

📂 Dataset Used

USA Housing Dataset

Features include:

Average Area Income

Average House Age

Average Number of Rooms

Average Number of Bedrooms

Area Population

Target:

House Price

The Address column was removed during preprocessing.

🤖 Machine Learning Models Implemented

A total of 13 regression models were trained and compared:

Linear Regression

Ridge Regression

Lasso Regression

ElasticNet

Polynomial Regression

Robust Regression (Huber)

SGD Regressor

Artificial Neural Network (ANN)

Random Forest Regressor

Support Vector Regression (SVR)

K-Nearest Neighbors (KNN)

LightGBM Regressor

XGBoost Regressor

📊 Model Evaluation

Each model was evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score

All evaluation results were saved in:

model_evaluation_results.csv

💾 Model Saving

Each trained model was saved using pickle:

LinearRegression.pkl
RandomForest.pkl
XGBoost.pkl
LightGBM.pkl
...


This allows models to be reused in the Flask application without retraining.

🌐 Flask Web Application Features

The web app allows users to:

Select any trained ML model

Enter housing feature values

Get real-time price predictions

View model performance comparison table

📁 Project Structure
housing-price-ml-model-comparison/
│
├── models (.pkl files)
├── model_evaluation_results.csv
├── train_models.py
├── app.py
├── templates/
│   ├── index.html
│   ├── results.html
│   └── model.html
└── README.md

🎯 What I Learned

Training and comparing multiple ML models

Understanding regression performance metrics

Hyper-model experimentation

Saving and loading ML models

Building ML-powered web applications with Flask

End-to-end ML project workflow

This project strengthened my understanding of machine learning, model evaluation, and deployment.
