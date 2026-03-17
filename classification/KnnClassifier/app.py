import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load model
model = pickle.load(open("knn_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Customer Purchase Prediction (KNN Model)")

st.write("Upload CSV file containing Age and EstimatedSalary")

# Upload CSV
file = st.file_uploader("Upload CSV",type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Uploaded Data")
    st.write(df)

    # Select features
    X = df[['Age','EstimatedSalary']].values

    # Scaling
    X_scaled = scaler.transform(X)

    # Prediction
    pred = model.predict(X_scaled)

    df["Prediction"] = pred

    st.subheader("Prediction Result")
    st.write(df)

    # ROC Curve
    if 'Purchased' in df.columns:

        y_true = df['Purchased']
        y_prob = model.predict_proba(X_scaled)[:,1]

        fpr,tpr,_ = roc_curve(y_true,y_prob)

        auc = roc_auc_score(y_true,y_prob)

        fig,ax = plt.subplots()

        ax.plot(fpr,tpr,label="AUC="+str(round(auc,2)))
        ax.plot([0,1],[0,1],'--')

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.legend()

        st.pyplot(fig)

    # Download result
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Prediction CSV",
        csv,
        "prediction_result.csv",
        "text/csv"
    )