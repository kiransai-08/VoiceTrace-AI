import json

files = [
    "dataset/raw/val_hc3_si.jsonl",
    "dataset/raw/test_hc3_si.jsonl"
]

for file_path in files:

    print("\n")
    print("=" * 70)
    print(file_path)
    print("=" * 70)

    with open(file_path, "r", encoding="utf-8") as f:

        for i in range(3):

            line = f.readline()

            if not line:
                break

            data = json.loads(line)

            print("\nRECORD", i + 1)
            print(data)