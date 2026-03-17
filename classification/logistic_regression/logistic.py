import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve

# Load dataset
dataset = pd.read_csv(r"D:\python\machine_learning\2.Classification\Logistic_Regression\logit classification.csv")

# Features and target
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic regression
model = LogisticRegression()

# Hyperparameter tuning
param_grid = {
'C':[0.001,0.01,0.1,1,10,100],
'penalty':['l1','l2'],
'solver':['liblinear'],
'max_iter':[100,200,300,400,500]
}

grid = GridSearchCV(model,param_grid,cv=5,scoring='accuracy')

grid.fit(X_train,y_train)

print("Best Parameters:",grid.best_params_)

classifier = grid.best_estimator_

# Prediction
y_pred = classifier.predict(X_test)

# Metrics
print("Accuracy:",accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)

print("Confusion Matrix:\n",cm)

# ROC
y_prob = classifier.predict_proba(X_test)[:,1]

roc = roc_auc_score(y_test,y_prob)

print("ROC AUC:",roc)

fpr,tpr,tresholds = roc_curve(y_test,y_prob)

plt.plot(fpr,tpr,label="ROC Curve")
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Save model
pickle.dump(classifier,open("model.pkl","wb"))

# Save scaler
pickle.dump(scaler,open("scaler.pkl","wb"))