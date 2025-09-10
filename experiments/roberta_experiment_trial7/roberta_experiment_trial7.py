# save as run_bi_encoder_av.py
import os
import json
import argparse
import numpy as np
import torch
import random
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_recall_fscore_support

# -----------------------------
# Data Loading
# -----------------------------
def load_av_examples(jsonl_path):
    """
    Reads a JSONL file and converts it to a list of sentence_transformers.InputExample.
    The label is converted from an integer (0/1) to a float (0.0/1.0).
    """
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            examples.append(InputExample(texts=[item['text0'], item['text1']], label=float(item['label'])))
    return examples

# -----------------------------
# Metrics
# -----------------------------
def compute_eer_and_tau(labels, probs):
    """Computes the Equal Error Rate (EER) and its corresponding threshold (tau)."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    tau = thresholds[idx]
    return {"eer": float(eer), "tau": float(tau)}

def compute_thresholded_metrics(labels, probs, tau):
    """Applies a fixed threshold (tau) to compute accuracy, precision, recall, and F1 score."""
    preds = (probs >= tau).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"acc_at_tau": float(acc), "precision_at_tau": float(precision), "recall_at_tau": float(recall), "f1_at_tau": float(f1)}

# -----------------------------
# SHAP for Bi-Encoder
# -----------------------------
def run_shap_on_bi_encoder(model_dir, val_jsonl, test_jsonl, shap_k=5):
    """Computes SHAP explanations for a Bi-Encoder model and saves them as an HTML file."""
    try:
        import shap
    except ImportError:
        print("[SHAP] Skipped (import error). Please install SHAP: pip install shap")
        return

    print("[SHAP] Loading model and preparing explainer...")
    model = SentenceTransformer(model_dir)
    tokenizer = model.tokenizer

    def predict_similarity(text_pairs):
        texts1, texts2 = [], []
        for pair in text_pairs:
            parts = pair.split(tokenizer.sep_token, 1)
            texts1.append(parts[0])
            texts2.append(parts[1] if len(parts) > 1 else "")

        embs1 = model.encode(texts1, convert_to_tensor=True, show_progress_bar=False)
        embs2 = model.encode(texts2, convert_to_tensor=True, show_progress_bar=False)

        cosine_scores = util.cos_sim(embs1, embs2)
        sims = np.diag(cosine_scores.cpu().numpy())
        return np.vstack((1 - sims, sims)).T

    explainer = shap.Explainer(predict_similarity, tokenizer)

    def load_texts_for_shap(jsonl_path, k):
        texts = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= k: break
                item = json.loads(line)
                texts.append(item['text0'] + tokenizer.sep_token + item['text1'])
        return texts

    val_texts = load_texts_for_shap(val_jsonl, shap_k)
    test_texts = load_texts_for_shap(test_jsonl, shap_k) if test_jsonl and os.path.exists(test_jsonl) else []

    shap_output_dir = os.path.join(model_dir, "shap_bi_encoder")
    os.makedirs(shap_output_dir, exist_ok=True)

    if val_texts:
        print(f"[SHAP] Running SHAP on {len(val_texts)} validation samples...")
        shap_values = explainer(val_texts)
        shap_html = shap.plots.text(shap_values, display=False)
        with open(os.path.join(shap_output_dir, "val_shap.html"), "w", encoding="utf-8") as f:
            f.write(shap_html)
        print(f"[SHAP] Saved: {os.path.join(shap_output_dir, 'val_shap.html')}")

    if test_texts:
        print(f"[SHAP] Running SHAP on {len(test_texts)} test samples...")
        shap_values = explainer(test_texts)
        shap_html = shap.plots.text(shap_values, display=False)
        with open(os.path.join(shap_output_dir, "test_shap.html"), "w", encoding="utf-8") as f:
            f.write(shap_html)
        print(f"[SHAP] Saved: {os.path.join(shap_output_dir, 'test_shap.html')}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./models/bi-encoder/exp")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--run_shap", action="store_true", default=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    best_model_dir = os.path.join(args.output_dir, "best_model")

    print("Defining Bi-Encoder model...")
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    print("Loading data...")
    train_examples = load_av_examples(args.train_jsonl)
    val_examples = load_av_examples(args.val_jsonl)

    val_sample_size = 1000
    if len(val_examples) > val_sample_size:
        print(f"Validation set is large. Using a random sample of {val_sample_size} for evaluation during training.")
        val_examples_for_eval = random.sample(val_examples, val_sample_size)
    else:
        val_examples_for_eval = val_examples

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size, num_workers=4)
    train_loss = losses.CosineSimilarityLoss(model)

    print("Starting training...")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples_for_eval, name='val')
    warmup_steps = int(len(train_dataloader) * args.num_train_epochs * args.warmup_ratio)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.num_train_epochs,
              warmup_steps=warmup_steps,
              evaluator=evaluator,
              evaluation_steps=args.logging_steps,
              output_path=best_model_dir,
              save_best_model=True,
              optimizer_params={'lr': args.learning_rate})

    print("\n--- Starting Evaluation ---")
    model = SentenceTransformer(best_model_dir)

    print("\n[VAL] Evaluating to find best threshold (tau)...")
    val_data = [json.loads(line) for line in open(args.val_jsonl)]
    val_texts1 = [item['text0'] for item in val_data]
    val_texts2 = [item['text1'] for item in val_data]
    val_labels = np.array([item['label'] for item in val_data])
    
    val_embs1 = model.encode(val_texts1, convert_to_tensor=True, show_progress_bar=True)
    val_embs2 = model.encode(val_texts2, convert_to_tensor=True, show_progress_bar=True)
    val_scores = np.diag(util.cos_sim(val_embs1, val_embs2).cpu().numpy())

    val_eer_tau = compute_eer_and_tau(val_labels, val_scores)
    print(f"[VAL] EER: {val_eer_tau['eer']:.6f}, Best Threshold (tau): {val_eer_tau['tau']:.6f}")

    if args.test_jsonl and os.path.exists(args.test_jsonl):
        print("\n[TEST] Evaluating on test set...")
        test_data = [json.loads(line) for line in open(args.test_jsonl)]
        test_texts1 = [item['text0'] for item in test_data]
        test_texts2 = [item['text1'] for item in test_data]
        test_labels = np.array([item['label'] for item in test_data])

        test_embs1 = model.encode(test_texts1, convert_to_tensor=True, show_progress_bar=True)
        test_embs2 = model.encode(test_texts2, convert_to_tensor=True, show_progress_bar=True)
        test_scores = np.diag(util.cos_sim(test_embs1, test_embs2).cpu().numpy())

        fixed_tau = val_eer_tau["tau"]
        test_metrics = compute_thresholded_metrics(test_labels, test_scores, fixed_tau)
        
        print("[TEST] Metrics (using val tau):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")
            
        test_auroc = roc_auc_score(test_labels, test_scores)
        test_auprc = average_precision_score(test_labels, test_scores)
        print(f"  roc_auc: {test_auroc:.6f}")
        print(f"  auprc: {test_auprc:.6f}")
        
        results = {
            "validation_metrics": val_eer_tau,
            "test_metrics_at_val_tau": test_metrics,
            "test_roc_auc": test_auroc,
            "test_auprc": test_auprc
        }
        with open(os.path.join(best_model_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    if args.run_shap:
        print("\n--- Running SHAP Explanations ---")
        run_shap_on_bi_encoder(
            model_dir=best_model_dir,
            val_jsonl=args.val_jsonl,
            test_jsonl=args.test_jsonl
        )

    print(f"\n[Done] Artifacts saved under: {best_model_dir}")

if __name__ == "__main__":
    main()