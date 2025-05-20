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
    page_title="Arthur - AI Excel Analyzer",
    page_icon="📊"
)

# -------------------- HEADER --------------------
st.title("🤖 Arthur - Your AI Excel Data Assistant")

# -------------------- Upload Excel File --------------------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File uploaded and read successfully!")
        with st.expander("View Raw Data"):
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None
else:
    df = None

# -------------------- Upload Image for OCR --------------------
st.markdown("---")
st.subheader("📷 Upload Image for Text Extraction (OCR)")
image_file = st.file_uploader("Upload an image (screenshot, photo, etc.)", type=["png", "jpg", "jpeg"])
if image_file:
    try:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        extracted_text = pytesseract.image_to_string(image)
        st.text_area("Extracted Text", extracted_text, height=150)
    except Exception as e:
        st.error(f"Failed to extract text from image: {e}")

# -------------------- Step 1: Handle Missing Data --------------------
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

# -------------------- Chat Interface (GPT-4 style) --------------------
openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else os.getenv("OPENAI_API_KEY")

if openai_api_key:
    st.markdown("---")
    st.subheader("💬 Arthur Chat - Ask Anything About Your Data")

    user_prompt = st.text_input("Ask a question about your data or image...")

    if user_prompt:
        if df is not None:
            df_preview = df.head(10).to_string()
            context = f"Here is a preview of the uploaded Excel file:\n{df_preview}\n"
        elif image_file:
            context = f"Extracted text from uploaded image:\n{extracted_text}\n"
        else:
            context = "No file or image uploaded."

        try:
            openai.api_key = openai_api_key
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Arthur, an AI Excel and OCR assistant."},
                    {"role": "user", "content": context + "\nUser question: " + user_prompt}
                ]
            )
            reply = response.choices[0].message['content']
            st.markdown(f"**Arthur:** {reply}")
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
else:
    st.warning("OpenAI API key not found. Please set it in Streamlit secrets or your environment.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("Made with ❤️ by Arthur AI")
