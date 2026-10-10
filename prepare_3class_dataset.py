import json
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

print("=" * 70)
print("PREPARING 3-CLASS DATASET (HUMAN=0, AI=1, AI-HUMANISED=2)")
print("=" * 70)

output_dir = "dataset/processed_3class"
os.makedirs(output_dir, exist_ok=True)

human_records = []
ai_records = []
reborn_records = []

seen_texts = set()

# Helper to add clean, unique records
def add_record(text, label, source_name, target_list):
    text_clean = text.strip()
    if not text_clean or len(text_clean) < 20:
        return False
    if text_clean in seen_texts:
        return False
    seen_texts.add(text_clean)
    target_list.append({
        "text": text_clean,
        "label": label,
        "source": source_name,
        "word_count": len(text_clean.split()),
        "char_count": len(text_clean)
    })
    return True

# 1. Process HC3 QA files for Human (0) and AI (1)
qa_files = [
    ("dataset/raw/train.jsonl", "hc3_train_qa"),
    ("dataset/raw/val_hc3_QA.jsonl", "hc3_val_qa"),
    ("dataset/raw/test_hc3_QA.jsonl", "hc3_test_qa")
]

for file_path, source in qa_files:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                lbl = data.get("label", 0)
                if lbl == 0:
                    add_record(text, 0, source, human_records)
                else:
                    add_record(text, 1, source, ai_records)
            except Exception:
                continue

# 2. Process HC3 SI files as AI-Humanised (2)
si_files = [
    ("dataset/raw/val_hc3_si.jsonl", "hc3_val_si"),
    ("dataset/raw/test_hc3_si.jsonl", "hc3_test_si")
]

for file_path, source in si_files:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                add_record(text, 2, source, reborn_records)
            except Exception:
                continue

# 3. Add Gemini synthetic humanized samples if available
if os.path.exists("dataset/processed/humanization_test.csv"):
    df_gem = pd.read_csv("dataset/processed/humanization_test.csv")
    if "humanized_text" in df_gem.columns:
        for text in df_gem["humanized_text"].dropna():
            add_record(text, 2, "gemini_synthetic", reborn_records)

print(f"Collected Raw Unique Samples:")
print(f"  Class 0 (Human):        {len(human_records)}")
print(f"  Class 1 (AI):           {len(ai_records)}")
print(f"  Class 2 (AI-Humanised): {len(reborn_records)}")

# Target balanced sampling count
samples_per_class = min(len(human_records), len(ai_records), len(reborn_records))
print(f"\nBalancing dataset to {samples_per_class} samples per class...")

np.random.seed(42)
df_human = pd.DataFrame(human_records).sample(n=samples_per_class, random_state=42)
df_ai = pd.DataFrame(ai_records).sample(n=samples_per_class, random_state=42)
df_reborn = pd.DataFrame(reborn_records).sample(n=samples_per_class, random_state=42)

df_master = pd.concat([df_human, df_ai, df_reborn], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"Master 3-Class Dataset Total: {len(df_master)} rows")
print("Class breakdown:\n", df_master["label"].value_counts())

# Split into Train (80%), Validation (10%), Test (10%)
train_df, temp_df = train_test_split(df_master, test_size=0.2, random_state=42, stratify=df_master["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

print(f"\nSplits Created:")
print(f"  Train:      {len(train_df)} rows")
print(f"  Validation: {len(val_df)} rows")
print(f"  Test:       {len(test_df)} rows")

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print(f"\nSaved all split CSVs to: {output_dir}")
