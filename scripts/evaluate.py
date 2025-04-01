import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load Test Data
TEST_CSV = "data/test_split.csv"
IMG_DIR = "data/train_images/"  # Change if test images are in a different folder
IMG_SIZE = 224
BATCH_SIZE = 16

test_df = pd.read_csv(TEST_CSV)
test_df["id_code"] = test_df["id_code"].astype(str) + ".png"
test_df["diagnosis"] = test_df["diagnosis"].astype(str)  # Convert labels to string

# Load Trained Model
model = tf.keras.models.load_model("more_trained_efficientnet_model.h5")

# Test Data Generator (No Augmentation, Only Rescaling)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=IMG_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # Important for correct evaluation
)

# Evaluate Model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# Get Predictions
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices, yticklabels=test_generator.class_indices)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
class_report = classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys()))
print(class_report)

# Save Report
with open("classification_report.txt", "w") as f:
    f.write(class_report)
