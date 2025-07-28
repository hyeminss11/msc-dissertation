import json
import os

# Settings
input_path = "experiments/llm_rewriting_trial1/output/av_eval_samples.json"
output_path = "experiments/llm_rewriting_trial1/output/claude_eval_prompts_from_av_eval.txt"

# Load data
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Generate Claude Prompt
prompt_blocks = []
for i, sample in enumerate(data):
    text1 = sample["text1"].strip()
    text2 = sample["text2"].strip()

    prompt = f"""\
Sample {i+1}
--------------------------
You are given two texts. Your task is to determine whether these two texts were likely written by the same author.

Please respond with only one word: "Yes" or "No".

Text 1:
{text1}

Text 2:
{text2}

Were these two texts likely written by the same author?
--------------------------\n"""
    
    prompt_blocks.append(prompt)

# Save
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(prompt_blocks)

print(f"{len(prompt_blocks)} Claude prompts saved to {output_path}")