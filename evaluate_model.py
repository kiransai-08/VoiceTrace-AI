import torch
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "./models/distilbert_baseline"
TEST_PATH = "./dataset/processed/test.csv"

MAX_LENGTH = 256
BATCH_SIZE = 16


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.")

print("\nNumber of classes:")
print(model.config.num_labels)

print("\nLabels:")
print(model.config.id2label)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

df = pd.read_csv(TEST_PATH)

print("Test samples:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# DETECT COLUMNS
# ============================================================

if "text" in df.columns:
    text_column = "text"
else:
    raise ValueError("Could not find 'text' column.")

if "label" in df.columns:
    label_column = "label"
else:
    raise ValueError("Could not find 'label' column.")

print("\nText column :", text_column)
print("Label column:", label_column)


# ============================================================
# PREPARE LABELS
# ============================================================

texts = df[text_column].astype(str).tolist()

raw_labels = df[label_column].tolist()

print("\nOriginal labels:")
print(df[label_column].value_counts())


# Convert labels to model IDs
label2id = {}

for i in range(model.config.num_labels):

    label_name = model.config.id2label[i]

    label2id[str(label_name).lower()] = i

print("\nModel label mapping:")
print(label2id)


converted_labels = []

for label in raw_labels:

    # Numeric label
    if isinstance(label, (int, np.integer)):
        converted_labels.append(int(label))

    elif isinstance(label, float) and label.is_integer():
        converted_labels.append(int(label))

    else:

        label_text = str(label).lower().strip()

        if label_text in label2id:
            converted_labels.append(
                label2id[label_text]
            )

        else:
            # Try numeric string
            try:
                converted_labels.append(
                    int(label_text)
                )
            except:
                raise ValueError(
                    f"Unknown label: {label}"
                )


labels = np.array(converted_labels)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("RUNNING TEST PREDICTIONS")
print("=" * 60)

all_predictions = []
all_probabilities = []

total = len(texts)

for start in range(0, total, BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        total
    )

    batch_texts = texts[start:end]

    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1
        )

    all_predictions.extend(
        predictions.cpu().numpy()
    )

    all_probabilities.extend(
        probabilities.cpu().numpy()
    )

    print(
        f"Processed {end}/{total}"
    )


predictions = np.array(all_predictions)

probabilities = np.array(all_probabilities)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = [
    model.config.id2label[i]
    for i in range(model.config.num_labels)
]

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        labels,
        predictions,
        labels=list(range(model.config.num_labels)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    labels,
    predictions,
    labels=list(range(model.config.num_labels))
)

print(cm)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    labels,
    predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    labels,
    predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 60)
print("OVERALL METRICS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")


# ============================================================
# ROC-AUC
# ============================================================

try:

    if model.config.num_labels == 2:

        auc = roc_auc_score(
            labels,
            probabilities[:, 1]
        )

    else:

        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    print(f"ROC-AUC   : {auc:.4f}")

except Exception as e:

    print("ROC-AUC calculation failed:")
    print(e)


# ============================================================
# SAMPLE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

sample_text = (
    "Artificial intelligence can help students "
    "learn difficult concepts by providing "
    "personalized explanations and examples."
)

inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():

    outputs = model(**inputs)

    sample_probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_id = torch.argmax(
        sample_probabilities
    ).item()


print("\nText:")
print(sample_text)

print("\nPrediction:")
print(
    model.config.id2label[predicted_id]
)

print("\nClass probabilities:")

for i, probability in enumerate(
    sample_probabilities
):

    print(
        f"{model.config.id2label[i]}: "
        f"{probability.item() * 100:.2f}%"
    )


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)