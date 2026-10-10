import json
import csv
import os

input_file = "dataset/raw/train.jsonl"
output_file = "dataset/processed/master.csv"

os.makedirs("dataset/processed", exist_ok=True)

rows = []
seen = set()

with open(input_file, "r", encoding="utf-8") as f:

    for i, line in enumerate(f, start=1):

        data = json.loads(line)

        text = data["text"].strip()
        label = data["label"]

        # Remove empty/short text
        if not text or len(text) < 20:
            continue

        # Remove duplicates
        if text in seen:
            continue

        seen.add(text)

        # Label name
        if label == 0:
            text_type = "human"
        else:
            text_type = "ai"

        rows.append({
            "source_id": f"hc3_train_{i:06d}",
            "text": text,
            "label": label,
            "type": text_type,
            "edit_level": 0
        })

with open(output_file, "w", encoding="utf-8", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source_id",
            "text",
            "label",
            "type",
            "edit_level"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("Master dataset created!")
print("Total samples:", len(rows))
print("Saved to:", output_file)