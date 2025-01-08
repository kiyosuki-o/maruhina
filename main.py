import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Paths for model, labels, and home image
MODEL_PATH = "trained_model.h5"
LABELS_PATH = "labels.txt"
HOME_IMG_PATH = "home_img.jpg"

# Check if required files exist
if not tf.io.gfile.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}!")
    st.stop()

if not tf.io.gfile.exists(LABELS_PATH):
    st.error(f"Labels file not found at {LABELS_PATH}!")
    st.stop()

# Load the trained model and class names
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to model input size
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    return input_arr

# Function to predict the class of the image
def predict_image(image):
    input_arr = preprocess_image(image)
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    return class_names[predicted_index]

# Streamlit UI
st.title("Fruits & Vegetables Recognition System")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Home Page
if app_mode == "Home":
    st.header("Welcome to the Fruits & Vegetables Recognition System")
    if tf.io.gfile.exists(HOME_IMG_PATH):
        st.image(HOME_IMG_PATH, use_column_width=True)
    else:
        st.error("Home image not found! Please check the file path.")

# About Page
elif app_mode == "About Project":
    st.header("About the Project")
    st.subheader("Dataset Information")
    st.text("This project recognizes fruits and vegetables using a deep learning model.")
    st.code("Classes: banana, apple, carrot, onion, tomato, etc.")
    st.text("Model trained with a variety of labeled images for classification.")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.info("Predicting...")
            try:
                prediction = predict_image(image)
                st.success(f"Model Prediction: {prediction}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
