import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt  

# Page setup
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor (King County Data)")
st.markdown("Estimate house prices based on sqft, bedrooms, and age.")

# Load data
try:
    df = pd.read_csv("train.csv")
    df.columns = df.columns.str.strip().str.lower()
except Exception as e:
    st.error(f" Failed to load dataset: {e}")
    st.stop()

# Required columns
required_cols = ['sqft_living', 'bedrooms', 'yr_built', 'price']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f" Missing required columns: {missing_cols}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Data preparation
df = df[required_cols].copy()
df['age'] = 2025 - df['yr_built']

X = df[['sqft_living', 'bedrooms', 'age']]
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Sidebar input
st.sidebar.header("🏡 Enter House Features")
sqft = st.sidebar.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
age = st.sidebar.slider("Age of the House (years)", 0, 150, 20)

# Prediction
if st.sidebar.button("Predict Price"):
    input_data = np.array([[sqft, bedrooms, age]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ₹{int(predicted_price):,}")

    # Show evaluation metrics
    st.markdown("### 📊 Model Evaluation")
    st.write(f"- RMSE (Root Mean Squared Error): ₹{rmse:,.2f}")
    st.write(f"- MAE (Mean Absolute Error): ₹{mae:,.2f}")
st.markdown("---")
st.markdown("🔗 Made with ❤️ by [Prathyusha](https://github.com/PrathyushaThulava)")

