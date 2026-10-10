import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# Custom Trainer to apply class weights in loss computation
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    
    # Class-wise metrics
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    metrics = {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "human_f1": f1_cls[0] if len(f1_cls) > 0 else 0,
        "ai_f1": f1_cls[1] if len(f1_cls) > 1 else 0,
        "reborn_f1": f1_cls[2] if len(f1_cls) > 2 else 0,
        "reborn_recall": r_cls[2] if len(r_cls) > 2 else 0
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train 3-Class AI / Human / AI-Humanised Classifier")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Transformer model identifier")
    parser.add_argument("--data_dir", type=str, default="dataset/processed_3class", help="Path to processed 3-class dataset")
    parser.add_argument("--output_dir", type=str, default="models/three_class_detector", help="Output directory for trained model")
    parser.add_argument("--train_size", type=int, default=15000, help="Number of training samples to use")
    parser.add_argument("--val_size", type=int, default=3000, help="Number of validation samples to use")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--use_class_weights", action="store_true", default=True, help="Enable class weighting for loss")
    args = parser.parse_args()

    print("=" * 80)
    print(f"TRAINING 3-CLASS TRANSFORMER MODEL: {args.model_name}")
    print("=" * 80)

    # 1. Load Datasets
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "validation.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Dataset files not found in {args.data_dir}. Run prepare_3class_dataset.py first!")
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if args.train_size and args.train_size < len(train_df):
        train_df = train_df.sample(n=args.train_size, random_state=42).reset_index(drop=True)
    if args.val_size and args.val_size < len(val_df):
        val_df = val_df.sample(n=args.val_size, random_state=42).reset_index(drop=True)

    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")
    print("Training class distribution:\n", train_df['label'].value_counts())

    # 2. Compute Class Weights if enabled
    if args.use_class_weights:
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label'].values
        )
        # Give extra weight boost to AI-Humanised (Class 2) to prevent boundary collapse
        weights[2] *= 1.2
        weights = weights / weights.sum() * 3.0
        print(f"Computed Class Weights [Human, AI, AI-Humanised]: {weights}")
    else:
        weights = None

    # 3. Initialize Tokenizer & Model
    id2label = {0: "Human", 1: "AI", 2: "AI-Humanised"}
    label2id = {"Human": 0, "AI": 1, "AI-Humanised": 2}

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

    # 4. Tokenize Data
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=args.max_length
        )

    ds_train = Dataset.from_pandas(train_df[['text', 'label']])
    ds_val = Dataset.from_pandas(val_df[['text', 'label']])

    ds_train = ds_train.map(tokenize_fn, batched=True, remove_columns=['text'])
    ds_val = ds_val.map(tokenize_fn, batched=True, remove_columns=['text'])

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=False, # Use FP32 for stability
        report_to="none"
    )

    trainer = WeightedLossTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 6. Train Model
    print("\nStarting Training...")
    trainer.train()

    # 7. Save Model & Tokenizer
    print(f"\nSaving best 3-class model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\nTraining Complete! Model saved successfully.")

if __name__ == "__main__":
    main()
