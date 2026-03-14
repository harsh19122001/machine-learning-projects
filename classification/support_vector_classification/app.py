import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load model
model = pickle.load(open("svm_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Customer Purchase Prediction")

st.write("Upload dataset to predict customer purchase")

# Upload file
file = st.file_uploader("Upload Excel or CSV file",type=["csv","xlsx"])

if file is not None:

    # Read file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("Uploaded Dataset")
    st.write(df)

    # Select features
    X = df.iloc[:,[2,3]].values

    # Scale
    X_scaled = scaler.transform(X)

    # Prediction
    predictions = model.predict(X_scaled)

    df["Prediction"] = predictions

    st.write("Prediction Result")
    st.write(df)

    # ROC curve if target exists
    if "Purchased" in df.columns:

        y_true = df["Purchased"]

        y_prob = model.predict_proba(X_scaled)[:,1]

        fpr,tpr,_ = roc_curve(y_true,y_prob)

        auc = roc_auc_score(y_true,y_prob)

        fig,ax = plt.subplots()

        ax.plot(fpr,tpr,label="AUC = "+str(round(auc,2)))
        ax.plot([0,1],[0,1],'--')

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.legend()

        st.pyplot(fig)