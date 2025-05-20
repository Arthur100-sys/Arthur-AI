import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from transformers import pipeline
import pyttsx3
import requests
from bs4 import BeautifulSoup

# -------------------- Page Config --------------------
st.set_page_config(
    layout="wide",
    page_title="Arthur - AI Excel Analyzer",
    page_icon="📊"
)

st.title("🤖 Arthur - Your AI Excel Data Assistant")

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # -------------------- Natural Language Summary --------------------
    def generate_summary(df):
        description = []

        description.append(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(exclude=['number']).columns

        description.append(f"There are {len(num_cols)} numerical columns and {len(cat_cols)} categorical/text columns.")

        if df.isnull().sum().sum() > 0:
            missing_info = df.isnull().sum()
            top_missing = missing_info[missing_info > 0].sort_values(ascending=False).head(3)
            missing_summary = ", ".join([f"{col} ({val} missing)" for col, val in top_missing.items()])
            description.append(f"Top columns with missing data: {missing_summary}")
        else:
            description.append("There are no missing values in the dataset.")

        prompt = " ".join(description) + " Summarize this dataset in one paragraph."
        summarizer = pipeline("text2text-generation", model="google/flan-t5-small", max_length=150)
        summary = summarizer(prompt)[0]['generated_text']

        return summary

    st.subheader("🧠 AI Summary of Your Data")
    summary = generate_summary(df)
    st.write(summary)

    # -------------------- Data Visualization --------------------
    st.subheader("📊 Correlation Heatmap")
    if len(df.select_dtypes(include='number').columns) >= 2:
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # -------------------- Predictive Analytics --------------------
    st.subheader("📈 Predictive Modeling")
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) >= 2:
        target_col = st.selectbox("Select the column to predict", numeric_cols)
        features = df[numeric_cols].drop(columns=[target_col])
        target = df[target_col]

        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)
        target_imputed = SimpleImputer(strategy='mean').fit_transform(target.values.reshape(-1, 1)).ravel()

        X_train, X_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        st.write(f"Model R² score: {r2_score(y_test, predictions):.2f}")

    # -------------------- Text-to-Speech Summary (Optional) --------------------
    if st.checkbox("🔊 Read Summary Aloud"):
        engine = pyttsx3.init()
        engine.say(summary)
        engine.runAndWait()

    # -------------------- Missing Data Autofill (Web Placeholder) --------------------
    st.subheader("🔍 Missing Data Info (Web Assisted Placeholder)")
    if df.isnull().sum().sum() > 0:
        try:
            example_url = "https://www.investopedia.com/"
            response = requests.get(example_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = [h.text for h in soup.find_all('h2')[:3]]
            st.write("Example insights from Investopedia:")
            for h in headlines:
                st.markdown(f"- {h}")
        except:
            st.write("Could not fetch online data. Please check your internet or site availability.")

else:
    st.info("👆 Upload an Excel file to get started.")
