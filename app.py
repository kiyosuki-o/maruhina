from flask import Flask, render_template, request, jsonify
from keras.utils import load_img, img_to_array # type: ignore
from keras.models import load_model # type: ignore
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model and labels
if not os.path.exists("trained_model.h5"):
    raise FileNotFoundError("trained_model.h5 not found!")
if not os.path.exists("labels.txt"):
    raise FileNotFoundError("labels.txt not found!")

model = load_model("trained_model.h5")
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess image for prediction
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(64, 64))
    input_arr = img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    return input_arr

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Process and predict
        input_arr = preprocess_image(file_path)
        predictions = model.predict(input_arr)
        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]

        return jsonify({"prediction": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
