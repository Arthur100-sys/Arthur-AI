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
import yfinance as yf
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

# -------------------- STATE INIT --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# -------------------- INPUT BAR + FILE UPLOADER --------------------
with st.sidebar:
    if st.button("➕ Upload File"):
        st.session_state.uploaded_file = st.file_uploader("", type=["xlsx", "xls", "csv", "json", "txt", "png", "jpg", "jpeg"])

openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else os.getenv("OPENAI_API_KEY")

context = ""
df = None

uploaded_file = st.session_state.uploaded_file
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
            context = f"Here is a preview of the uploaded Excel file:\n{df.head(10).to_string()}\n"
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            context = f"Here is a preview of the uploaded CSV file:\n{df.head(10).to_string()}\n"
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
            context = f"Here is a preview of the uploaded JSON file:\n{df.head(10).to_string()}\n"
        elif uploaded_file.name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            context = f"Here is the uploaded text:\n{content[:1000]}"
        elif uploaded_file.name.endswith((".png", "jpg", "jpeg")):
            image = Image.open(uploaded_file)
            extracted_text = pytesseract.image_to_string(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            context = f"Extracted text from image:\n{extracted_text}"
    except Exception as e:
        st.error(f"File processing error: {e}")

if df is not None:
    with st.expander("View Uploaded Data"):
        st.dataframe(df)

# -------------------- MISSING DATA --------------------
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

# -------------------- LIVE FINANCE CHECK --------------------
def get_live_finance_answer(prompt):
    try:
        words = prompt.lower().split()
        tickers = [w for w in words if w.isalpha() and len(w) <= 5]
        if not tickers:
            return None

        responses = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    responses.append(f"The latest closing price for **{ticker.upper()}** is **${price:.2f}**.")
            except Exception:
                pass

        return "\n".join(responses) if responses else None
    except Exception as e:
        return f"Live data error: {e}"

# -------------------- CHAT INPUT --------------------
user_prompt = st.chat_input("Ready when you are...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    try:
        openai.api_key = openai_api_key
        base_instruction = "You are Arthur, an AI assistant who specializes in finance, global economics, global market trading, global macroeconomics, and all tradable markets like forex, stocks, futures, and commodities. Respond with clear and accurate insights based only on your niche."
        full_prompt = context + "\nUser question: " + user_prompt if context else user_prompt

        live_answer = get_live_finance_answer(user_prompt)

        messages = [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": full_prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        reply = response.choices[0].message['content']

        final_response = f"{live_answer}\n\n{reply}" if live_answer else reply
        st.session_state.messages.append({"role": "assistant", "content": final_response})

    except Exception as e:
        st.error(f"OpenAI API error: {e}")

# -------------------- CHAT DISPLAY --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Made with ❤️ by Arthur AI")
ccccccccccccccccccccccccccccccccccccccccccccccc