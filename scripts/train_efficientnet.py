import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight

# Enable mixed precision (Uncomment if using NVIDIA GPU)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Paths
TRAIN_CSV = "data/train_split.csv"
VAL_CSV = "data/val_split.csv"
IMG_DIR = "data/train_images/"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50  # Increased for better convergence

# Load dataset
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# Convert labels to string format
train_df["diagnosis"] = train_df["diagnosis"].astype(str)
val_df["diagnosis"] = val_df["diagnosis"].astype(str)

# Append `.png` extension
train_df["id_code"] = train_df["id_code"].astype(str) + ".png"
val_df["id_code"] = val_df["id_code"].astype(str) + ".png"

# 🔹 Data Augmentation for Training Set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  # Added for more diversity
    brightness_range=[0.7, 1.3]
)

# 🔹 Validation Data (No Augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Data Loaders
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMG_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=IMG_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Check data loading
print(f"Total training batches: {len(train_generator)}")
print(f"Total validation batches: {len(val_generator)}")

# 🔹 Load Pre-Trained EfficientNet-B3
base_model = EfficientNetB3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")

# 🔹 Fine-Tuning: Unfreeze last 50 layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

# 🔹 Define the Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Better than Flatten()
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(5, activation="softmax")  # 5 classes
])

# 🔹 Learning Rate Scheduler (Exponential Decay)
def lr_schedule(epoch, lr):
    return lr * 0.95 if epoch > 3 else lr  # Reduce LR by 5% after 3 epochs

# 🔹 Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

# 🔹 Compute Class Weights for Imbalanced Data
class_labels = train_df["diagnosis"].astype(int).values
class_weights = compute_class_weight("balanced", classes=np.unique(class_labels), y=class_labels)
class_weights_dict = {i: class_weights[i] for i in range(5)}
print("Class Weights:", class_weights_dict)

# 🔹 Compile Model
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Fine-tuning LR
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 🔹 Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=max(1, len(train_generator)),
    validation_steps=max(1, len(val_generator)),
    class_weight=class_weights_dict,
    callbacks=[early_stopping, LearningRateScheduler(lr_schedule)]
)

# 🔹 Save Trained Model
model.save("more_trained_efficientnet_model.h5")
print("Model training complete! Saved as 'more_trained_efficientnet_model.h5'")
