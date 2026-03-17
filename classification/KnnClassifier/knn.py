# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load dataset
dataset = pd.read_csv(r"C:\Users\Dell\Downloads\logit classification.csv")

# Independent and dependent variables
X = dataset.iloc[:,[2,3]].values   # Age, EstimatedSalary
y = dataset.iloc[:,4].values       # Purchased

# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# KNN Model
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    p=2
)

classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report\n", report)

# ROC Curve
y_prob = classifier.predict_proba(X_test)[:,1]

roc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", roc)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="ROC Curve (AUC="+str(round(roc,2))+")")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# Save model
with open("knn_model.pkl","wb") as f:
    pickle.dump(classifier,f)

with open("scaler.pkl","wb") as f:
    pickle.dump(sc,f)