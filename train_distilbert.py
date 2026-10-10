import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ==========================================
# 1. Load datasets
# ==========================================

train_file = "dataset/processed/train.csv"
val_file = "dataset/processed/validation.csv"

# Load CSV files
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)

# Use smaller datasets for CPU training
train_df = train_df.sample(
    n=20000,
    random_state=42
)

val_df = val_df.sample(
    n=4000,
    random_state=42
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# Keep only the columns needed by DistilBERT
train_df = train_df[["text", "label"]]
val_df = val_df[["text", "label"]]

# Convert pandas DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

# ==========================================
# 2. Load DistilBERT tokenizer
# ==========================================

model_name = "distilbert-base-uncased"

print("\nLoading tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    model_name
)

# ==========================================
# 3. Tokenize dataset
# ==========================================

def tokenize(batch):

    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


print("\nTokenizing training dataset...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

print("\nTokenizing validation dataset...")

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

# Remove original text column
train_dataset = train_dataset.remove_columns(
    ["text"]
)

val_dataset = val_dataset.remove_columns(
    ["text"]
)

# Use PyTorch format
train_dataset.set_format("torch")
val_dataset.set_format("torch")

# ==========================================
# 4. Load DistilBERT model
# ==========================================

print("\nLoading DistilBERT model...")

from transformers import AutoConfig

model_path = "./models/distilbert_baseline"

config = AutoConfig.from_pretrained(model_path)

config.id2label = {
    0: "Human",
    1: "AI"
}

config.label2id = {
    "Human": 0,
    "AI": 1
}

config.save_pretrained(model_path)

print("Labels updated successfully!")
print("0 =", config.id2label[0])
print("1 =", config.id2label[1])

# ==========================================
# 5. Training configuration
# ==========================================

training_args = TrainingArguments(
    output_dir="./models/distilbert_baseline",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,

    report_to="none"
)

# ==========================================
# 6. Create Trainer
# ==========================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset
)

# ==========================================
# 7. Start training
# ==========================================

print("\n")
print("=" * 60)
print("Starting DistilBERT training...")
print("=" * 60)
print("\n")

trainer.train()

# ==========================================
# 8. Save model
# ==========================================

model_path = "./models/distilbert_baseline"

trainer.save_model(model_path)

tokenizer.save_pretrained(model_path)

print("\n")
print("=" * 60)
print("Training completed successfully!")
print("=" * 60)

print("\nModel saved at:")
print(model_path)