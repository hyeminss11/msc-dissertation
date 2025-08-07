import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
set_seed(7)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 1. Custom dataset
class AVPairDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=4096, sample_size=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        if sample_size is not None:
            self.data = self.data[:sample_size]
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text1"],
            item["text2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": int(item["label"])
        }
    

# 2. metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 3. Load Data
train_dataset = AVPairDataset("msc-dissertation/experiments/data/article-tweet/train.json", max_length = 512, tokenizer = tokenizer, sample_size = None)
val_dataset = AVPairDataset("msc-dissertation/experiments/data/article-tweet/val.json", max_length = 512, tokenizer = tokenizer, sample_size = None)

# 4. Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
experiment_name = "article_tweet_trial3_2"
base_output_dir = f"./models/distilbert/{experiment_name}"
best_model_dir = f"{base_output_dir}/best_model"

# 5. Training setup
training_args = TrainingArguments(
    output_dir=base_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none"
)

# 6. Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=train_dataset.tokenizer,
    compute_metrics=compute_metrics
)

# 7. Start training
trainer.train()

# 8. Save the model and tokeniser
trainer.save_model(best_model_dir)
train_dataset.tokenizer.save_pretrained(best_model_dir)

# 9. Evaluate with eval dataset
results = trainer.evaluate(eval_dataset=val_dataset)
print("\nEvaluation results on validation set:")
for k, v in results.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# 10. Save result
with open(f"{best_model_dir}/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
trainer.save_state()