import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytesseract
from PIL import Image
import io
import base64
import openai
import os
import json
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    layout="wide",
    page_title="Arthur - AI Data Analyzer",
    page_icon="📊"
)

# -------------------- HEADER --------------------
st.title("🤖 Arthur - Your AI Data Assistant")

# -------------------- Custom Chat + Upload UI --------------------
st.markdown("---")
col1, col2 = st.columns([10, 1])

with col1:
    user_input = st.text_input("Ready when you are", placeholder="Ask Arthur anything about your data...")

with col2:
    uploaded_file = st.file_uploader("", type=["xlsx", "xls", "csv", "txt", "json", "jpg", "jpeg", "png"], label_visibility="collapsed")

# -------------------- Process Uploaded File --------------------
data_context = ""
dataframe = None
extracted_text = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type in ["xlsx", "xls"]:
            dataframe = pd.read_excel(uploaded_file)
            data_context = dataframe.head(10).to_string()
        elif file_type == "csv":
            dataframe = pd.read_csv(uploaded_file)
            data_context = dataframe.head(10).to_string()
        elif file_type == "txt":
            content = uploaded_file.read().decode("utf-8")
            data_context = content
        elif file_type == "json":
            json_obj = json.load(uploaded_file)
            data_context = json.dumps(json_obj, indent=2)
        elif file_type in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            extracted_text = pytesseract.image_to_string(image)
            data_context = extracted_text
        st.success("File processed successfully!")
    except Exception as e:
        st.error(f"Failed to process file: {e}")

# -------------------- Handle Missing Data --------------------
if dataframe is not None:
    st.markdown("---")
    st.subheader("🧼 Handling Missing Numeric Data")
    try:
        imputer = SimpleImputer(strategy='mean')
        df_imputed = dataframe.copy()
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        df_imputed[numeric_cols] = imputer.fit_transform(dataframe[numeric_cols])
        st.success("Missing numeric values handled successfully!")
        with st.expander("View Cleaned Data"):
            st.dataframe(df_imputed)
        data_context += "\n\nCleaned Data Preview:\n" + df_imputed.head(5).to_string()
    except Exception as e:
        st.error(f"Missing Data Handling Error: {e}")

# -------------------- Arthur Chat (GPT-like) --------------------
openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else os.getenv("OPENAI_API_KEY")

if openai_api_key and user_input:
    st.markdown("---")
    st.subheader("💬 Arthur's Response")
    context_block = data_context if data_context else "No data uploaded."
    try:
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Arthur, a smart AI data assistant. Answer the user's questions based on the context."},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion:\n{user_input}"}
            ]
        )
        reply = response.choices[0].message['content']
        st.markdown(f"**Arthur:** {reply}")
    except Exception as e:
        st.error(f"OpenAI API error: {e}")

elif not openai_api_key:
    st.warning("OpenAI API key not found. Please set it in Streamlit secrets or your environment.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("Made with ❤️ by Arthur AI")
