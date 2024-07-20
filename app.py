import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('satellite_image_model.h5')

# Define the class names
class_names = ['cloudy', 'desert', 'green_area', 'water']

# Streamlit app setup
st.title("Satellite Image Classification")
st.write("Upload an image to classify it into one of the categories: Cloudy, Desert, Green Area, or Water.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image_data.resize((128, 72))  # Resize to match model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    prediction_scores = predictions[0]

    # Display results
    st.write(f"**Predicted Category:** {predicted_class}")
    st.write(f"**Prediction Scores:** {prediction_scores}")

    # Display scores for each class
    for i, score in enumerate(prediction_scores):
        st.write(f"{class_names[i]}: {score:.4f}")
