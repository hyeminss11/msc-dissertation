import json
from sklearn.model_selection import train_test_split

# 1. Load existing train.json
with open("experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/article-article/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=7, stratify=[d["label"] for d in data])

# 3. Save
with open("experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/article-article/train_split.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open("experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/article-article/val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples.")