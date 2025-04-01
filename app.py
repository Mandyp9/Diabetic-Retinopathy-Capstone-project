from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "G:/DR DETECTION APP CAPSTONE/models/more_trained_efficientnet_model.h5"  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Define labels for severity levels
labels = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# Severity descriptions
severity_descriptions = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Early-stage signs such as small microaneurysms.",
    2: "More blood vessel damage, possible hemorrhages.",
    3: "Large hemorrhages, increased risk of vision loss.",
    4: "Abnormal blood vessel growth, highest risk of blindness."
}

# Image preprocessing function
def preprocess_image(image_path):
    """ Load and preprocess the image for model prediction """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

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
    
    # Preprocess and predict
    image = preprocess_image(file_path)
    predictions = model.predict(image)[0]  # Get first prediction in batch
    predicted_class = np.argmax(predictions)  # Get class with highest probability
    confidence = predictions[predicted_class] * 100  # Convert to percentage

    severity_label = labels[predicted_class]  # Map class to label
    severity_rank = predicted_class  # Rank (0-4)
    explanation = severity_descriptions[predicted_class]  # Get severity explanation

    return render_template('result.html', 
                           label=severity_label, 
                           rank=severity_rank, 
                           confidence=f"{confidence:.2f}%", 
                           explanation=explanation, 
                           image=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
