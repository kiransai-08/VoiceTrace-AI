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