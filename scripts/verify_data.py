import os

base_path = "data/"
print("Files in dataset directory:", os.listdir(base_path))

# Check number of images
print(f"Total train images: {len(os.listdir(base_path + 'train_images/'))}")
print(f"Total test images: {len(os.listdir(base_path + 'test_images/'))}")
