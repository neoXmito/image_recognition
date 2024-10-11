import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your Keras model
model_path = 'Project1/ClassificationProject/models/my_first_model.keras'  # Update with your relative model path if stored locally
model = load_model(model_path)

# Define the class names in the desired order
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Set up Streamlit interface
st.title("Flower Image Recognition Keras Model Prediction App")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image and preprocess it
    img = image.load_img(uploaded_file, target_size=(150, 150, 3))  # Adjust size based on model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image if your model requires it

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get class index

    # Map predicted index to the class name
    predicted_class_name = class_names[predicted_class_index]

    # Display results
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted class: {predicted_class_name}")
