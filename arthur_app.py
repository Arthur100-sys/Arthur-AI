import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textblob import TextBlob
import pyttsx3
import speech_recognition as sr
import openai
import os

# Set page theme
theme = st.sidebar.radio("Select Theme:", ("Light", "Dark"))
if theme == "Dark":
    st.markdown("""<style>body { background-color: #0e1117; color: white; }</style>""", unsafe_allow_html=True)

st.set_page_config(page_title="Arthur - AI Excel Analyzer", layout="wide")
st.title("🤖 Arthur - Your AI Excel Data Assistant")

# Text-to-speech engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech recognition
def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return "Sorry, couldn't understand."

# Load Excel
@st.cache_data
def load_excel(file):
    return pd.read_excel(file, engine='openpyxl')

# Descriptive stats
@st.cache_data
def describe_data(df):
    return df.describe()

# Correlation matrix
@st.cache_data
def compute_correlation(df):
    return df.corr()

# Sentiment analysis
@st.cache_data
def compute_sentiment(column):
    def analyze_sentiment(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0
    return column.dropna().apply(analyze_sentiment)

# --- Chatbot interface appears first ---
st.header("💬 Chat with Arthur")

user_input = st.text_input("Ask Arthur a question about your data:")
if user_input:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful data assistant named Arthur."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response['choices'][0]['message']['content']
        st.markdown(f"**Arthur:** {reply}")
        speak(reply)
    except Exception as e:
        st.error(f"Error: {e}")

# Upload Excel file
st.header("📂 Upload Excel File")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = load_excel(uploaded_file)
    st.success("File uploaded successfully!")
    speak("File uploaded")

    if st.checkbox("Show raw data"):
        st.dataframe(df.head(100))

    # Descriptive
    st.subheader("📊 Descriptive Analysis")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        st.write(describe_data(df[num_cols]))
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(compute_correlation(df[num_cols]), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Sentiment
    text_cols = df.select_dtypes(include='object').columns.tolist()
    if text_cols:
        st.subheader("📝 Sentiment Analysis")
        selected_col = st.selectbox("Select text column", text_cols)
        df['sentiment'] = compute_sentiment(df[selected_col])
        st.write(df[[selected_col, 'sentiment']].head())

        fig2, ax2 = plt.subplots()
        sns.histplot(df['sentiment'], bins=20, ax=ax2)
        st.pyplot(fig2)

    # Suggestions
    st.subheader("💡 Prescriptive Suggestions")
    suggestions = []
    if 'sentiment' in df.columns and df['sentiment'].mean() < 0:
        suggestions.append("Overall sentiment is negative. Investigate customer feedback.")
    if suggestions:
        for s in suggestions:
            st.warning(s)
            speak(s)
    else:
        st.info("No major issues detected.")
        speak("No major issues detected.")
