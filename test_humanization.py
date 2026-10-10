import pandas as pd
from google import genai

# Connect to Gemini
client = genai.Client()

# Load AI samples
input_file = "dataset/processed/ai_to_humanize.csv"

df = pd.read_csv(input_file)

# Take only 10 samples for testing
samples = df.head(10)

results = []

for index, row in samples.iterrows():

    original_text = row["text"]
    source_id = row["source_id"]

    print("\n" + "=" * 70)
    print("Processing:", source_id)

    for level in [1, 2, 3]:

        if level == 1:
            instruction = """
Rewrite the text with light human editing.
Keep the original meaning and information.
Make only small changes to wording and sentence structure.
Do not add new information.
"""

        elif level == 2:
            instruction = """
Rewrite the text with moderate human editing.
Keep the original meaning and information.
Change sentence structures and wording naturally.
Make the writing sound less polished and more naturally written.
Do not add new information.
"""

        else:
            instruction = """
Rewrite the text with heavy human editing.
Keep the original meaning and important information.
Substantially change the wording and sentence structure.
Make it sound naturally written by a person, including natural variation in sentence length and phrasing.
Do not add new information.
"""

        prompt = f"""
You are creating a research dataset for AI-text-origin detection.

{instruction}

Return ONLY the rewritten text.
Do not explain what you changed.

Original text:
{original_text}
"""

        try:

            response = client.interactions.create(
                model="gemini-3.6-flash",
                input=prompt
            )

            humanized_text = response.output_text.strip()

            results.append({
                "source_id": source_id,
                "original_text": original_text,
                "humanized_text": humanized_text,
                "label": 1,
                "type": "ai_humanized",
                "edit_level": level
            })

            print(f"Level {level} completed.")

        except Exception as e:

            print(f"Level {level} failed:", e)


# Save results
output_file = "dataset/processed/humanization_test.csv"

result_df = pd.DataFrame(results)

result_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("Humanization test completed!")
print("Samples generated:", len(result_df))
print("Saved to:", output_file)