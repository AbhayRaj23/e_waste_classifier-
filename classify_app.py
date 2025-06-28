import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("e_waste_cnn_model.h5")

# Define image input size and class labels
IMG_SIZE = (150, 150)
class_names = ['Large Appliance', 'Telecom', 'Consumer Electronics']  # <-- Update if needed

# Page title
st.set_page_config(page_title="E-Waste Classifier", layout="centered")
st.title("♻️ E-Waste Image Classifier")
st.markdown("Upload an image of an electronic device to classify it into an E-Waste category.")

# File uploader
uploaded_file = st.file_uploader("Drag & drop or browse an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image = image.resize(IMG_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Show result
        st.success(f"✅ Predicted E-Waste Category: **{predicted_class}**")

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the image: {e}")

else:
    st.info("👈 Please upload a JPG or PNG image to get started.")
