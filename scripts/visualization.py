import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

# Load dataset
train_df = pd.read_csv("data/train_split.csv")

# Set image directory
IMG_DIR = "data/train_images/"

# Define DR severity levels
severity_levels = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

# Select one image per class
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for severity, ax in zip(range(5), axes):
    sample_img_id = train_df[train_df["diagnosis"] == severity]["id_code"].iloc[0]
    img_path = os.path.join(IMG_DIR, f"{sample_img_id}.png")
    
    # Load and display image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(f"Class {severity}: {severity_levels[severity]}")
    ax.axis("off")

plt.show()
