import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Constants
MODEL_PATH = "cnn_model.h5"
  
# Download the model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']  

# Streamlit UI
st.title("The WeatherLENS 🌤️")
st.write("By Mark Angelo A. Aquino")
st.write("The WeatherLENS is an AI-powered image classification app that identifies weather conditions such as Cloudy, Rain, Shine, or Sunrise from photos. Using deep learning, it provides fast and accurate predictions to help users understand the weather captured in any image.")
st.write("========================================================================================")
st.write("Upload a weather image and let us classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image: This was your weather', use_column_width=True)

    # Preprocess image
    img = img.resize((256, 256))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.write(f"### ✅ The predicted weather was: {predicted_class}")
    st.write(f"### 📊 Accuracy: **{confidence:.2f}%**")

