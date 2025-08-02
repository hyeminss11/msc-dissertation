from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from bigbird_train_trial2 import AVPairDataset

# 1. Define metric function
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

# 2. load model and tokeniser
model_path = "./models/bigbird"
tokeniser = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 3. load dtaset
eval_dataset = AVPairDataset("msc-dissertation/data/val.json", tokenizer_name=model_path)

# 4. eval
trainer = Trainer(
    model=model,
    tokenizer=tokeniser,
    compute_metrics=compute_metrics
)

results = trainer.evaluate(eval_dataset=eval_dataset)
print("Evaluation results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")