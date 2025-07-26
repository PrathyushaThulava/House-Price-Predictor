import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt  

# Page Configuration
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Background Gradient Styling (HTML + CSS)
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #fce4ec, #e0f7fa);
        }

        .stApp {
            background: linear-gradient(to bottom right, #ffe0f0, #d0f0ff);
        }

        .stSidebar .sidebar-content {
            background: linear-gradient(to bottom, #f8bbd0, #b2ebf2);
            color: #000000;
        }

        h1 {
            color: #d81b60;  /* Deep pink heading */
        }

        h2, h3, h4 {
            color: #00838f;  /* Soft blue-green heading */
        }

        .stButton>button {
            background-color: #f48fb1; /* Baby pink */
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #81d4fa; /* Light blue */
            transform: scale(1.05);
        }

        .stSlider > div {
            color: #ad1457;
        }

        .css-1offfwp, .css-1v3fvcr {
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)


# Title and Description
st.title("🏠 House Price Predictor")
st.markdown("##### *Using King County Housing Data*")
st.write("📐 Estimate house prices based on square footage, number of bedrooms, and house age.")

# Load dataset
try:
    df = pd.read_csv("train.csv")
    df.columns = df.columns.str.strip().str.lower()
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# Ensure required columns exist
required_cols = ['sqft_living', 'bedrooms', 'yr_built', 'price']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"⚠️ Missing required columns: {missing_cols}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Prepare data
df = df[required_cols].copy()
df['age'] = 2025 - df['yr_built']
X = df[['sqft_living', 'bedrooms', 'age']]
y = df['price']

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Sidebar Inputs
st.sidebar.header("🔧 Enter House Features")
sqft = st.sidebar.number_input("📏 Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
bedrooms = st.sidebar.slider("🛏️ Bedrooms", 1, 10, 3)
age = st.sidebar.slider("🏚️ Age of House (years)", 0, 150, 20)

# Predict Button
if st.sidebar.button("🎯 Predict Price"):
    input_data = np.array([[sqft, bedrooms, age]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ₹{int(predicted_price):,}")

    st.markdown("### 📊 Model Evaluation Metrics")
    st.info(f"**RMSE:** ₹{rmse:,.2f}")
    st.info(f"**MAE:** ₹{mae:,.2f}")

# Footer
st.markdown("---")
st.markdown("🔗 Made with ❤️ by [**Prathyusha**](https://github.com/PrathyushaThulava)")
