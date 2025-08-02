from transformers import AutoTokenizer
import json
import warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
with open("msc-dissertation/experiments/data/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lengths = []
for sample in data:
    encoded = tokenizer(sample["text1"], sample["text2"], truncation=False)
    lengths.append(len(encoded["input_ids"]))

print(f"# samples: {len(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Samples > 4096: {sum(l > 4096 for l in lengths)}")