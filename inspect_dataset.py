import json

file_path = "dataset/raw/train.jsonl"

with open(file_path, "r", encoding="utf-8") as f:

    for i in range(3):
        line = f.readline()

        if not line:
            break

        data = json.loads(line)

        print("\n" + "=" * 60)
        print("RECORD", i + 1)
        print("=" * 60)
        print(data)