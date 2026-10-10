import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "dataset/processed/master.csv"

df = pd.read_csv(input_file)

print("Total samples:", len(df))

# 80% training, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# Split remaining 20% into 10% validation and 10% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Save files
train_df.to_csv(
    "dataset/processed/train.csv",
    index=False
)

val_df.to_csv(
    "dataset/processed/validation.csv",
    index=False
)

test_df.to_csv(
    "dataset/processed/test.csv",
    index=False
)

print("\nDataset split completed!")

print("\nTraining:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\nTraining distribution:")
print(train_df["label"].value_counts())

print("\nValidation distribution:")
print(val_df["label"].value_counts())

print("\nTesting distribution:")
print(test_df["label"].value_counts())