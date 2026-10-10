import json

file_path = "dataset/raw/train.jsonl"

human = 0
ai = 0
total = 0

with open(file_path, "r", encoding="utf-8") as f:

    for line in f:
        data = json.loads(line)

        label = data["label"]

        if label == 0:
            human += 1
        elif label == 1:
            ai += 1

        total += 1

print("Total:", total)
print("Human:", human)
print("AI:", ai)