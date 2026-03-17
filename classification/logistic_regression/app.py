import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score

# Load model
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Logistic Regression Prediction App")

st.write("Upload Excel File for Prediction")

file = st.file_uploader("Upload CSV",type=["csv"])

if file:

    data = pd.read_csv(file)

    X = data.iloc[:,[2,3]]

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)

    data['Prediction'] = prediction

    st.write(data)

    st.download_button(
        label="Download Predictions",
        data=data.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

# ROC Curve Section
st.header("ROC Curve")

if st.button("Show ROC Curve"):

    dataset = pd.read_csv(r"D:\python\machine_learning\2.Classification\Logistic_Regression\logit classification.csv")

    X = dataset.iloc[:,[2,3]]
    y = dataset.iloc[:,-1]

    X_scaled = scaler.transform(X)

    y_prob = model.predict_proba(X_scaled)[:,1]

    fpr,tpr,_ = roc_curve(y,y_prob)

    auc = roc_auc_score(y,y_prob)

    fig,ax = plt.subplots()

    ax.plot(fpr,tpr,label=f"AUC = {auc:.2f}")

    ax.plot([0,1],[0,1],'--')

    ax.set_xlabel("False Positive Rate")

    ax.set_ylabel("True Positive Rate")

    ax.legend()

    st.pyplot(fig)