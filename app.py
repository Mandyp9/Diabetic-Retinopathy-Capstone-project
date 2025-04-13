from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# --- Load trained models ---
MODEL_A_PATH = "G:/DR DETECTION APP CAPSTONE/models/more_trained_efficientnet_model.h5"
MODEL_B_PATH = "G:/DR DETECTION APP CAPSTONE/models/trained_efficientnet_model.h5"  # Replace with actual model B path

model_a = tf.keras.models.load_model(MODEL_A_PATH)
model_b = tf.keras.models.load_model(MODEL_B_PATH)

# --- Define labels for severity levels ---
labels = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# --- Severity descriptions ---
severity_descriptions = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild DR: Early-stage signs such as small microaneurysms.",
    2: "Moderate DR: More blood vessel damage, possible hemorrhages.",
    3: "Severe DR: Large hemorrhages, increased risk of vision loss.",
    4: "Proliferative DR: Abnormal blood vessel growth, highest risk of blindness."
}

# --- Image preprocessing function ---
def preprocess_image(image_path):
    """ Load and preprocess the image for model prediction """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected!"
    
    file_path = os.path.join("static", file.filename)
    file.save(file_path)
    
    # Preprocess image
    image = preprocess_image(file_path)

    # --- Model A prediction ---
    predictions_a = model_a.predict(image)[0]
    class_a = np.argmax(predictions_a)
    confidence_a = predictions_a[class_a] * 100

    # --- Model B prediction ---
    predictions_b = model_b.predict(image)[0]
    class_b = np.argmax(predictions_b)
    confidence_b = predictions_b[class_b] * 100

    # --- Prepare result for both models ---
    result_a = {
        'label': labels[class_a],
        'rank': class_a,
        'confidence': f"{confidence_a:.2f}%",
        'explanation': severity_descriptions[class_a]
    }

    result_b = {
        'label': labels[class_b],
        'rank': class_b,
        'confidence': f"{confidence_b:.2f}%",
        'explanation': severity_descriptions[class_b]
    }

    return render_template('result.html',
                           image=file.filename,
                           result_a=result_a,
                           result_b=result_b)

if __name__ == '__main__':
    app.run(debug=True)
