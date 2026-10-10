import json
import csv
import os

input_file = "dataset/raw/train.jsonl"
output_file = "dataset/processed/train.csv"

os.makedirs("dataset/processed", exist_ok=True)

rows = []
seen = set()

with open(input_file, "r", encoding="utf-8") as f:

    for line in f:
        data = json.loads(line)

        text = data["text"].strip()
        label = data["label"]

        # Remove empty text
        if not text:
            continue

        # Remove very short text
        if len(text) < 20:
            continue

        # Remove duplicate text
        if text in seen:
            continue

        seen.add(text)

        if label == 0:
            text_type = "human"
        else:
            text_type = "ai"

        rows.append({
            "text": text,
            "label": label,
            "type": text_type,
            "edit_level": 0
        })

with open(output_file, "w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["text", "label", "type", "edit_level"]
    )

    writer.writeheader()
    writer.writerows(rows)

print("Dataset prepared successfully!")
print("Total samples:", len(rows))

human = sum(1 for x in rows if x["label"] == 0)
ai = sum(1 for x in rows if x["label"] == 1)

print("Human:", human)
print("AI:", ai)
print("Saved to:", output_file)