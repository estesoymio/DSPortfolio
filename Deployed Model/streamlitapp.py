import streamlit as st
import boto3
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# AWS S3 credentials
AWS_ACCESS_KEY = st.secrets.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = st.secrets.get("AWS_SECRET_KEY")
BUCKET_NAME = st.secrets.get("BUCKET_NAME")
MODEL_PATH = 'Cat_dog_model.h5'
EXPECTED_MODEL_SIZE = 123456789  # Replace with the actual size of your model file in bytes

# Function to verify the integrity of the downloaded file
def verify_file(path, expected_size):
    if os.path.exists(path):
        actual_size = os.path.getsize(path)
        if actual_size == expected_size:
            return True
        else:
            st.error(f"File size mismatch: expected {expected_size}, got {actual_size}")
            return False
    else:
        st.error(f"File not found: {path}")
        return False

# Download model from S3
def download_model():
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        if not os.path.exists(MODEL_PATH) or not verify_file(MODEL_PATH, EXPECTED_MODEL_SIZE):
            st.write("Downloading model from S3...")
            with open(MODEL_PATH, 'wb') as f:
                s3.download_fileobj(BUCKET_NAME, MODEL_PATH, f)
            if verify_file(MODEL_PATH, EXPECTED_MODEL_SIZE):
                st.write("Model downloaded and verified successfully.")
            else:
                st.error("Downloaded model file is corrupted.")
                return None
        else:
            st.write("Model already exists locally and is verified.")
        return MODEL_PATH
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        raise e

# Load the model
try:
    model_path = download_model()
    if model_path and os.path.exists(model_path):
        st.write(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully.")
    else:
        st.error(f"Model file does not exist or is corrupted at {model_path}")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define the prediction function
def predict_image(image):
    try:
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        return prediction[0]
    except Exception as e:
        st.error(f"Error predicting image: {e}")

# Define the Streamlit app
st.title('Image Prediction Model Deployment with Streamlit')

# Collect user input: Image upload
st.header('Upload an Image')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button('Predict'):
        if 'model' in locals() or 'model' in globals():
            result = predict_image(image)
            st.write(f'Prediction: {result}')
        else:
            st.error("Model is not loaded.")
