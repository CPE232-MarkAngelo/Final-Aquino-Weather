
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained CNN model
model = load_model('cnn_best_model.h5')

# Class names
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Page config
st.set_page_config(page_title="Weather Classifier", layout="centered")

# Title
st.title("Identifying Weather Image")
st.write("Upload an image to predict whether it is **Cloudy**, **Rain**, **Shine**, or **Sunrise**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    st.subheader("🔍 Prediction:")
    st.success(f"The image is classified as **{predicted_class}**.")
