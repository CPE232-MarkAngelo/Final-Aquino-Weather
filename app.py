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
st.title("The Weather Image Classifier 🌤️")
st.write("By Mark Angelo A. Aquino")
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
    predicted_class = class_labels[np.argmax(prediction)]

    st.write(f"### The Predicted weather was: {predicted_class}")

