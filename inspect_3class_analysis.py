import json
import pandas as pd
import numpy as np
import os
import re

print("=" * 80)
print("COMPREHENSIVE DATASET & CLASS CHARACTERISTICS INSPECTION")
print("=" * 80)

# 1. Load sample data from raw and processed files
# Load train.jsonl for AI & Human
human_samples = []
ai_samples = []

with open("dataset/raw/train.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        data = json.loads(line)
        text = data.get("text", "").strip()
        label = data.get("label", 0)
        if label == 0:
            human_samples.append(text)
        else:
            ai_samples.append(text)

# Load SI samples (val_hc3_si.jsonl and test_hc3_si.jsonl)
si_samples = []
for si_path in ["dataset/raw/val_hc3_si.jsonl", "dataset/raw/test_hc3_si.jsonl"]:
    if os.path.exists(si_path):
        with open(si_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                si_samples.append(data.get("text", "").strip())

# Load Gemini humanized samples if present
gemini_samples = []
if os.path.exists("dataset/processed/humanization_test.csv"):
    df_gem = pd.read_csv("dataset/processed/humanization_test.csv")
    if "humanized_text" in df_gem.columns:
        gemini_samples = df_gem["humanized_text"].dropna().tolist()

print(f"Loaded Raw Samples:")
print(f"  Human (train.jsonl sample): {len(human_samples)}")
print(f"  AI (train.jsonl sample):    {len(ai_samples)}")
print(f"  SI / Paraphrased (HC3 SI):  {len(si_samples)}")
print(f"  Gemini Humanized:          {len(gemini_samples)}")

# Helper stats function
def calc_stats(text_list, name):
    char_lens = [len(t) for t in text_list]
    word_lens = [len(t.split()) for t in text_list]
    sent_counts = [len(re.split(r'[.!?]+', t)) for t in text_list]
    avg_word_per_sent = [w / max(1, s) for w, s in zip(word_lens, sent_counts)]
    
    print(f"\n--- Class: {name} (N={len(text_list)}) ---")
    print(f"  Avg Char Length:  {np.mean(char_lens):.2f} +/- {np.std(char_lens):.2f}")
    print(f"  Avg Word Count:   {np.mean(word_lens):.2f} +/- {np.std(word_lens):.2f}")
    print(f"  Avg Sent Count:   {np.mean(sent_counts):.2f} +/- {np.std(sent_counts):.2f}")
    print(f"  Avg Words/Sent:   {np.mean(avg_word_per_sent):.2f}")
    print(f"  Min Word Count:   {min(word_lens) if word_lens else 0}")
    print(f"  Max Word Count:   {max(word_lens) if word_lens else 0}")

calc_stats(human_samples[:1000], "Human")
calc_stats(ai_samples[:1000], "AI")
calc_stats(si_samples[:1000], "SI / Paraphrased (HC3-SI)")
if gemini_samples:
    calc_stats(gemini_samples, "Gemini Synthetic Humanized")

print("\n" + "=" * 80)
print("PRINTING 20 EXAMPLES PER CLASS")
print("=" * 80)

print("\n>>> 20 AI EXAMPLES:")
for idx, text in enumerate(ai_samples[:20], 1):
    preview = text.replace('\n', ' ')
    print(f"[{idx:02d}] ({len(text.split())} words) {preview[:150]}...")

print("\n>>> 20 HUMAN EXAMPLES:")
for idx, text in enumerate(human_samples[:20], 1):
    preview = text.replace('\n', ' ')
    print(f"[{idx:02d}] ({len(text.split())} words) {preview[:150]}...")

print("\n>>> 20 REBORN / SI EXAMPLES:")
for idx, text in enumerate(si_samples[:20], 1):
    preview = text.replace('\n', ' ')
    print(f"[{idx:02d}] ({len(text.split())} words) {preview[:150]}...")

if gemini_samples:
    print("\n>>> GEMINI HUMANIZED EXAMPLES (ALL):")
    for idx, text in enumerate(gemini_samples, 1):
        preview = text.replace('\n', ' ')
        print(f"[{idx:02d}] ({len(text.split())} words) {preview[:150]}...")

