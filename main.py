import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model using st.cache_resource (since it's a heavy resource)
@st.cache_resource
def load_skin_model():
    print("Loading the model...")
    model = load_model("trained_model.h5")
    print("Model loaded successfully.")
    return model

model = load_skin_model()

# Load class labels using st.cache_data (since it's simple data)
@st.cache_data
def load_labels(filepath="labels.txt"):
    print("Loading class labels...")
    with open(filepath, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
    print(f"Loaded {len(labels)} class labels.")
    return labels

CLASS_LABELS = load_labels()

# Preprocessing function
def preprocess_image(image, target_size=(180, 180)):
    """
    Preprocess an image for the model.
    - Safely resizes large images using thumbnail
    - Resizes to 180x180
    - Normalizes pixel values to [0, 1]
    - Adds batch dimension for prediction
    """
    # Step 1: Ensure the image is safely resized to avoid memory issues
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)  # Downsample large images to 1024x1024
    image = image.resize(target_size, Image.Resampling.LANCZOS)  # Final resize to 180x180

    # Step 2: Normalize the image
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

    # Step 3: Add batch dimension (for batch size = 1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# Streamlit interface
st.title("Skin Cancer Detection Application")
st.write("Upload a skin image, and the model will predict the class.")

uploaded_file = st.file_uploader("Upload your image below", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Step 1: Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Step 2: Preprocess the image
        processed_image = preprocess_image(image, target_size=(180, 180))

        # Step 3: Predict using the model
        prediction = model.predict(processed_image)[0]  # Extract probabilities

        # Map predictions to class labels
        predicted_label = CLASS_LABELS[np.argmax(prediction)]  # Get label of highest probability
        print(f"Predicted label: {predicted_label}")

        st.write(f"The skin condition is: **{predicted_label}**")

        # Display all class probabilities
        st.write("Our prediction probabilities:")
        probability_dict = {CLASS_LABELS[i]: prediction[i] * 100 for i in range(len(CLASS_LABELS))}
        # Display barchart
        df = pd.DataFrame({"Condition": list(probability_dict.keys()), "Probability": list(probability_dict.values())})
        st.bar_chart(df.set_index("Condition"))

    except MemoryError:
        st.error("The uploaded image is too large! Please try a smaller image.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        print(f"Error details: {e}")
