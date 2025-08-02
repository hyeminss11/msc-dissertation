from collections import Counter
import json

with open("msc-dissertation/data/train.json") as f:
    data = json.load(f)
print(Counter([d["label"] for d in data]))