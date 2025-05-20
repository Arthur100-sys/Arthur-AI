# arthur_app.py

# -------------------- Imports --------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# -------------------- Page Configuration (FIRST Streamlit command) --------------------
st.set_page_config(
    layout="wide",
    page_title="Arthur - AI Excel Analyzer",
    page_icon="📊"
)


# -------------------- Optional: Dark Theme Styling --------------------
theme = "Dark"  # You can change this to "Light" or make it dynamic later
if theme == "Dark":
    st.markdown("""
        <style>
            body {
                background-color: #0e1117;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

# -------------------- Title --------------------
st.title("🤖 Arthur - Your AI Excel Data Assistant")

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx", "xls"])

# -------------------- Main Logic --------------------
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded and read successfully!")

        # Show raw data
        with st.expander("📄 View Raw Data"):
            st.dataframe(df)

        # -------------------- Step 1: Handle Missing Data --------------------
        st.subheader("🧼 Step 1: Handling Missing Data")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if numeric_cols:
            imputer_num = SimpleImputer(strategy='mean')
            df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

        if cat_cols:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

        st.success("✅ Missing values filled.")

        # -------------------- Step 2: Descriptive Statistics --------------------
        st.subheader("📊 Step 2: Descriptive Statistics")
        st.write(df.describe())

        # -------------------- Step 3: Natural Language Summary --------------------
        st.subheader("🧠 Step 3: AI Summary")
        st.write(f"Your dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        st.write(f"Numeric columns: {len(numeric_cols)} | Categorical columns: {len(cat_cols)}")
        st.write("Here's a quick insight: Most columns are now clean and ready for analysis.")

        # -------------------- Step 4: Data Visualization --------------------
        st.subheader("📈 Step 4: Data Visualization")
        col_to_plot = st.selectbox("Choose a numeric column to visualize", numeric_cols)
        if col_to_plot:
            fig, ax = plt.subplots()
            sns.histplot(df[col_to_plot], kde=True, ax=ax)
            st.pyplot(fig)

        # -------------------- Step 5: Predictive Analytics --------------------
        st.subheader("🤖 Step 5: Predictive Analytics")
        target = st.selectbox("Select a numeric target variable to predict", numeric_cols)

        if target:
            df_model = df.copy()
            X = df_model.drop(columns=[target])
            y = df_model[target]

            for col in X.select_dtypes(include='object').columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            st.write(f"Model R² Score: **{r2:.2f}**")
            st.success("✅ Prediction completed using RandomForest.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👈 Please upload an Excel file to get started.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("Made with ❤️ by Arthur AI")
