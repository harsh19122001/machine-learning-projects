# Import all necesary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pickle

# Load dataset
dataset = pd.read_csv(r"D:\python\machine_learning\2.Classification\SVM\logit classification.csv")

# Divide dataset into dependent variables and independent variables
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

# Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Hyperparameter tuning
param_grid = {
    'C' : [0.001,0.01,0.1,1,10,100],
    'kernel' : ['linear','rbf','poly'],
    'gamma' : ['scale','auto']   
}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator= SVC(probability = True), param_grid = param_grid, cv = 5, scoring = 'accuracy')
grid.fit(x_train,y_train)

print("Best Parameters:", grid.best_params_)

# Best Model
classifier = grid.best_estimator_

# Prediction
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve

#  Evaluation 
cm = confusion_matrix(y_test,y_pred)
print("Confusion matrix", cm)

accuracy = accuracy_score(y_test,y_pred)
print('Accurarcy',accuracy)

report = classification_report(y_test,y_pred)
print("classification report", report)


# ROC
y_prob = classifier.predict_proba(x_test)[:,1]

roc = roc_auc_score(y_test,y_prob)

print("ROC AUC:",roc)

fpr,tpr,tresholds = roc_curve(y_test,y_prob)

plt.plot(fpr,tpr,label="ROC Curve")
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

import os

model_path = r"C:\pickle"
scaler_path = r"C:\pickle"

with open(model_path,"wb") as f:
    pickle.dump(classifier,f)

with open(scaler_path,"wb") as f:
    pickle.dump(sc,f)

print("Model saved at:", model_path)
print("Scaler saved at:", scaler_path)

print("File exists:", os.path.exists(model_path))