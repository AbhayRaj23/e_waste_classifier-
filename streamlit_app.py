import streamlit as st
import joblib
import numpy as np

# Load saved objects
model = joblib.load("e_waste_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

device_encoder = encoders['device']
battery_encoder = encoders['battery']

st.title("🔋 E-Waste Category Classifier")

# Input Form
device_type = st.selectbox("Device Type", device_encoder.classes_)
usage_years = st.slider("Usage Duration (years)", 0, 15, 5)
weight_kg = st.number_input("Weight (kg)", min_value=0.1, max_value=100.0, value=5.0)
battery_type = st.selectbox("Battery Type", battery_encoder.classes_)
metal_ratio = st.slider("Metal Composition Ratio", 0.0, 1.0, 0.5)
repair_count = st.slider("Repair Count", 0, 10, 1)

# Prepare input
if st.button("Predict Category"):
    input_data = np.array([[
        device_encoder.transform([device_type])[0],
        usage_years,
        weight_kg,
        battery_encoder.transform([battery_type])[0],
        metal_ratio,
        repair_count
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    label_map = {
        0: "Large Household Appliances",
        1: "IT and Telecommunication Equipment",
        2: "Consumer Electronics"
    }

    st.success(f"🔍 Predicted E-Waste Category: **{label_map[prediction]}**")
