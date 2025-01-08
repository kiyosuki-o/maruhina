import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model and labels
MODEL_PATH = "trained_model.h5"
LABELS_PATH = "labels.txt"

if not tf.io.gfile.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

if not tf.io.gfile.exists(LABELS_PATH):
    st.error("Labels file not found!")
    st.stop()

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to model input size
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    return input_arr

# Function to predict the class
def predict_image(image):
    input_arr = preprocess_image(image)
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    return class_names[predicted_index]

# Streamlit UI
st.title("Fruits & Vegetables Recognition System")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Home Page
if app_mode == "Home":
    st.header("Welcome to the Fruits & Vegetables Recognition System")
    st.image("home_img.jpg", use_column_width=True)

# About Page
elif app_mode == "About Project":
    st.header("About the Project")
    st.subheader("Dataset Information")
    st.text("This project recognizes fruits and vegetables using a deep learning model.")
    st.code("Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("Vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soybean, cauliflower, bell pepper, chili pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.info("Predicting...")
            prediction = predict_image(image)
            st.success(f"Model Prediction: {prediction}")
