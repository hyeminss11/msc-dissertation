import argparse
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from bigbird_train_trial2_1_aa import AVPairDataset

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

# 2. Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test.json")
    args = parser.parse_args()

    # Load model and tokenizer
    model_path = f"./models/bigbird/{args.experiment_name}/best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load test set
    eval_dataset = AVPairDataset(args.test_path, tokenizer_name=model_path)

    # Evaluate
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    results = trainer.evaluate(eval_dataset=eval_dataset)

    # Print and save results
    print("Evaluation results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    with open(f"./models/bigbird/{args.experiment_name}/best_model/eval_results_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()