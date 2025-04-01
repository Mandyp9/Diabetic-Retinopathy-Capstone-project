import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
CSV_PATH = "data/train.csv"  
df = pd.read_csv(CSV_PATH)  # Load the dataset 

# Splitting dataset (80% Train, 10% Validation, 10% Test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["diagnosis"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["diagnosis"])

# Save splits
train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)
test_df.to_csv("data/test_split.csv", index=False)

print(f"Train Set: {len(train_df)} | Validation Set: {len(val_df)} | Test Set: {len(test_df)}")
print("Dataset split successfully!")
