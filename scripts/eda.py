import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = "data/train.csv"
df = pd.read_csv(CSV_PATH)

print(df.head())  # Display first few rows
print("Total images in dataset:", len(df))
print("Unique class labels:", df["diagnosis"].unique())

plt.figure(figsize=(8, 5))
sns.countplot(x=df["diagnosis"], palette="viridis")

# Labels and title
plt.xlabel("DR Severity Level")
plt.ylabel("Number of Images")
plt.title("Distribution of Diabetic Retinopathy Severity Levels")
plt.show()
