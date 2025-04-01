import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Load dataset
CSV_PATH = "data/train.csv"  
IMG_DIR = "data/train_images/"
IMG_SIZE = 224  # Resizing all images to 224x224

df = pd.read_csv(CSV_PATH) 

# Function to preprocess a single image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))  # Resize
    img = img_to_array(img) / 255.0  # Convert to array & normalize
    return img

# Check one sample image
sample_img_path = os.path.join("data/train_images/", df["id_code"].iloc[0] + ".png")  
sample_img = preprocess_image(sample_img_path)

print(f"Sample image shape: {sample_img.shape}")  # Expected: (224, 224, 3)

# Data Augmentation
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

# Apply augmentation to one sample image
augmented_img = data_gen.random_transform(sample_img)

# Show the original and augmented image
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(sample_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(augmented_img)
plt.title("Augmented Image")
plt.axis("off")

plt.show()
