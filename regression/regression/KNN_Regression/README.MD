# KNN Regression – Salary vs Experience

## 📌 Project Description
This project implements a **K-Nearest Neighbors (KNN) Regression model** to predict **employee salary based on years of experience** using a real-world dataset.

The project focuses on understanding how **distance-based regression models** work and why **feature scaling is mandatory** for KNN algorithms.

---

## 📂 Dataset Details
The dataset includes the following columns:
- Position
- Level (Experience)
- Salary

**Independent Variable:** Experience  
**Dependent Variable:** Salary

---

## ⚙️ Steps Followed
1. Imported required libraries (NumPy, Pandas, Matplotlib)
2. Loaded the dataset using Pandas
3. Selected independent and dependent variables
4. Split data into training and test sets (80-20)
5. Applied **StandardScaler** for feature scaling
6. Trained KNN Regression models with different values of **K**
7. Evaluated models using **R² score**
8. Selected optimal K value
9. Evaluated final model using **MSE and RMSE**
10. Visualized **Salary vs Experience** using Matplotlib

---

## 📊 Model Evaluation
- **R² Score** used to compare performance for different K values
- **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** used for final evaluation
- Prediction made for **6.5 years of experience**

---

## 🧠 Why Feature Scaling is Required?
KNN is a distance-based algorithm. Feature scaling ensures that distance calculations are not biased due to different data ranges.

---

## 📈 Visualization
The project includes a graph showing:
- Training data points
- KNN regression curve
- Relationship between experience and salary

---

## 🛠️ Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## 🎯 Learning Outcomes
- Practical understanding of KNN Regression
- Importance of feature scaling
- Model evaluation using R², MSE, and RMSE
- Working with real-world datasets

---



