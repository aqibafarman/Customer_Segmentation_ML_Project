import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('kmeans_saved_model.pkl', 'rb'))

# ---------- Data Preprocessing ----------
def load_and_clean_data(file_path):
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # RFM metrics
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], errors='coerce')
    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = (max_date - retail['InvoiceDate']).dt.days
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    rfm = rfm.select_dtypes(include=['number'])

    # Remove outliers
    Q1 = rfm.quantile(0.05)
    Q3 = rfm.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[
        (rfm.Amount >= Q1.Amount - 1.5 * IQR.Amount) & (rfm.Amount <= Q3.Amount + 1.5 * IQR.Amount)
    ]
    rfm = rfm[
        (rfm.Frequency >= Q1.Frequency - 1.5 * IQR.Frequency) & (rfm.Frequency <= Q3.Frequency + 1.5 * IQR.Frequency)
    ]
    rfm = rfm[
        (rfm.Recency >= Q1.Recency - 1.5 * IQR.Recency) & (rfm.Recency <= Q3.Recency + 1.5 * IQR.Recency)
    ]

    return rfm


def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Amount', 'Frequency', 'Recency'])
    return rfm, rfm_scaled


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("🧩 Customer Segmentation Prediction App")
st.markdown("Upload your retail dataset (CSV) to analyze and visualize customer clusters.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Processing your file...'):
        # Save uploaded file temporarily
        temp_file_path = "uploaded_data.csv"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocess
        rfm, rfm_scaled = preprocess_data(temp_file_path)
        predictions = model.predict(rfm_scaled)
        rfm["Cluster_Id"] = predictions

        st.success("✅ File processed successfully!")
        st.subheader("Clustered Data Sample")
        st.dataframe(rfm.head(10))

        # Visualizations
        st.subheader("📊 Cluster Visualizations")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots()
            sns.stripplot(x='Cluster_Id', y='Amount', data=rfm, hue='Cluster_Id', ax=ax)
            plt.title("Cluster vs Amount")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.stripplot(x='Cluster_Id', y='Frequency', data=rfm, hue='Cluster_Id', ax=ax)
            plt.title("Cluster vs Frequency")
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots()
            sns.stripplot(x='Cluster_Id', y='Recency', data=rfm, hue='Cluster_Id', ax=ax)
            plt.title("Cluster vs Recency")
            st.pyplot(fig)

        # Download option
        csv = rfm.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clustered Data",
            data=csv,
            file_name="clustered_customers.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Please upload a CSV file to begin.")
