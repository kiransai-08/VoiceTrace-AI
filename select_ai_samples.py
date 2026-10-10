import pandas as pd

input_file = "dataset/processed/train.csv"
output_file = "dataset/processed/ai_to_humanize.csv"

df = pd.read_csv(input_file)

# Select AI samples only
ai_df = df[df["label"] == 1].copy()

# Take 10,000 AI samples
ai_df = ai_df.sample(
    n=10000,
    random_state=42
)

ai_df.to_csv(
    output_file,
    index=False
)

print("AI samples selected!")
print("Number of samples:", len(ai_df))
print("Saved to:", output_file)