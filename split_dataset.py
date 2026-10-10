import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "dataset/processed/train.csv"

train_file = "dataset/processed/train.csv"
val_file = "dataset/processed/validation.csv"
test_file = "dataset/processed/test.csv"

df = pd.read_csv(input_file)

print("Total samples:", len(df))

# First: 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# Second: split temporary into 10% validation and 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

print("\nDataset split completed!")

print("\nTraining:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

print("\nValidation class distribution:")
print(val_df["label"].value_counts())

print("\nTesting class distribution:")
print(test_df["label"].value_counts())