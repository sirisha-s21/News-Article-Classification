import pandas as pd

# Load True and Fake CSVs
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add labels
true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

# Combine and shuffle
combined_df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42)

# Keep only text + label columns
combined_df = combined_df[["text","label"]]

# Save merged dataset
combined_df.to_csv("data/news_dataset.csv", index=False)
print("✅ Merged dataset saved as data/news_dataset.csv")
print("Total samples:", len(combined_df))
print(combined_df.head())
