# -*- coding: utf-8 -*-
"""StreamlitApp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MWsw0VHkPvHHgu9_47JNbIrIEeUWHnz8

# Deploy on Streamlit
"""

import streamlit as st
import boto3
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# AWS S3 credentials
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
MODEL_PATH = 'Cat_dog_model.h5'

# Download model from S3
def download_model():
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'wb') as f:
            s3.download_fileobj(BUCKET_NAME, MODEL_PATH, f)
    return MODEL_PATH

# Load the model
model_path = download_model()
model = tf.keras.models.load_model(model_path)

# Define the prediction function
def predict_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    return prediction[0]

# Define the Streamlit app
st.title('Image Prediction Model Deployment with Streamlit')

# Collect user input: Image upload
st.header('Upload an Image')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        result = predict_image(image)
        st.write(f'Prediction: {result}')