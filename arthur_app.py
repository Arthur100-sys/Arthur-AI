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
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    layout="wide",
    page_title="Arthur",
    page_icon="📊"
)

# -------------------- HEADER --------------------
st.markdown("""
    <h1 style='text-align: center;'>🤖 Arthur</h1>
""", unsafe_allow_html=True)

# -------------------- Chat Interface + Upload --------------------
openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else os.getenv("OPENAI_API_KEY")

uploaded_file = None
image_file = None

col1, col2 = st.columns([10, 1])
with col1:
    user_prompt = st.text_input("Ready when you are")
with col2:
    uploaded_file = st.file_uploader("", type=["xlsx", "xls", "csv", "json", "txt", "png", "jpg", "jpeg"], label_visibility="collapsed")

context = ""
extracted_text = ""
df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            context = f"Here is a preview of the uploaded Excel file:\n{df.head(10).to_string()}\n"
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            context = f"Here is a preview of the uploaded CSV file:\n{df.head(10).to_string()}\n"
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
            context = f"Here is a preview of the uploaded JSON file:\n{df.head(10).to_string()}\n"
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode("utf-8")
            context = f"Here is the uploaded text:\n{content[:1000]}"
        elif uploaded_file.name.endswith(('png', 'jpg', 'jpeg')):
            image = Image.open(uploaded_file)
            extracted_text = pytesseract.image_to_string(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            context = f"Extracted text from image:\n{extracted_text}"
            image_file = uploaded_file
    except Exception as e:
        st.error(f"File processing error: {e}")

# Show uploaded DataFrame if available
if df is not None:
    with st.expander("View Uploaded Data"):
        st.dataframe(df)

# Handle missing data
if df is not None:
    st.markdown("---")
    st.subheader("🧼 Step 1: Handling Missing Data")
    try:
        imputer = SimpleImputer(strategy='mean')
        df_imputed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        st.success("Missing numeric values handled successfully!")
        with st.expander("View Imputed Data"):
            st.dataframe(df_imputed)
    except Exception as e:
        st.error(f"Missing Data Handling Error: {e}")

# Chat with Arthur (with or without file)
if openai_api_key and user_prompt:
    try:
        openai.api_key = openai_api_key
        base_instruction = "You are Arthur, an AI assistant who specializes in finance, global economics, global market trading, global macroeconomics, and all tradable markets like forex, stocks, futures, and commodities. Respond with clear and accurate insights based only on your niche."
        full_prompt = context + "\nUser question: " + user_prompt if context else user_prompt

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": base_instruction},
                {"role": "user", "content": full_prompt}
            ]
        )
        reply = response.choices[0].message['content']
        st.markdown(f"**Arthur:** {reply}")
    except Exception as e:
        st.error(f"OpenAI API error: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Arthur AI")
