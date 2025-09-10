# save as eval_bi_encoder_with_shap.py
import os, json, argparse, numpy as np, torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def read_jsonl(p):
    out=[]
    with open(p,'r',encoding='utf-8') as f:
        for line in f: out.append(json.loads(line))
    return out

def eer_and_tau(y, s):
    fpr, tpr, th = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i]+fnr[i])/2), float(th[i])

def metrics_at_tau(y, s, tau):
    pred = (s >= tau).astype(int)
    p,r,f1,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    acc = accuracy_score(y, pred)
    return dict(acc_at_tau=float(acc), precision_at_tau=float(p),
                recall_at_tau=float(r), f1_at_tau=float(f1))

def encode_pairs(model, a_list, b_list, batch_size=256, show=True):
    emb_a = model.encode(a_list, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=show)
    emb_b = model.encode(b_list, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=show)
    return np.diag(util.cos_sim(emb_a, emb_b).cpu().numpy())

def maybe_run_shap(model, test_items, shap_k, out_dir):
    if shap_k <= 0:
        return
    try:
        import shap
        from shap.maskers import Text as TextMasker

        tok = model.tokenizer
        sep = tok.sep_token or "</s>"

        # --- (1) Select a subset of samples (only top shap_k to keep it lightweight) ---
        items = test_items[:shap_k]
        texts = [f"{d['text0']}{sep}{d['text1']}" for d in items]

        # --- (2) Prediction function: returns scores as (N, 2) class-like probabilities ---
        def predict_similarity(text_pairs):
            left, right = [], []
            for t in text_pairs:
                parts = t.split(sep, 1)
                left.append(parts[0])
                right.append(parts[1] if len(parts) > 1 else "")
            e1 = model.encode(left,  convert_to_tensor=True, show_progress_bar=False)
            e2 = model.encode(right, convert_to_tensor=True, show_progress_bar=False)
            sims = np.diag(util.cos_sim(e1, e2).cpu().numpy())   # [-1, 1]
            sims01 = (sims + 1.0) / 2.0                          # [0, 1]
            # binary “probabilities”: P(diff-author)=1-sims01, P(same-author)=sims01
            return np.vstack((1.0 - sims01, sims01)).T

        # --- (3) Build the SHAP explainer with a text masker ---
        masker = TextMasker(tokenizer=tok)
        explainer = shap.Explainer(predict_similarity, masker)

        # --- (4) Compute SHAP explanations ---
        sv = explainer(texts)  # shap.Explanation object

        # --- (5) Save results: HTML (for visualization) + NPZ (for numeric analysis) ---
        os.makedirs(out_dir, exist_ok=True)

        # Save interactive HTML (text plot returns HTML string, not a Visualizer)
        html_path = os.path.join(out_dir, "shap_text.html")
        html_str = shap.plots.text(sv, display=False)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"[SHAP] html saved -> {html_path}")

        # Save raw SHAP values in NPZ format (easy to reload later)
        npz_path = os.path.join(out_dir, "shap_text.npz")
        np.savez_compressed(
            npz_path,
            values=sv.values,               # SHAP contribution values
            data=np.array(sv.data, dtype=object),  # token strings
            base_values=sv.base_values,     # baseline values
            sep=sep
        )

        # Optionally save original items for reference
        meta_path = os.path.join(out_dir, "shap_items.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        print(f"[SHAP] npz saved -> {npz_path}")
        print(f"[SHAP] meta saved -> {meta_path}")

    except Exception as e:
        print(f"[SHAP] skipped: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to trained SentenceTransformer (best_model)")
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--shap_k", type=int, default=20)
    ap.add_argument("--shap_outdir", type=str, default=None)
    ap.add_argument("--only_shap", action="store_true", help="Skip evaluation and run only SHAP")
    ap.add_argument("--val_jsonl", type=str, default=None, help="(optional) validation jsonl for tau(EER)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--no_shap", action="store_true", help="Skip SHAP generation entirely")


    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    model = SentenceTransformer(args.model_dir, device=device)

# ---- load test ----
    test = read_jsonl(args.test_jsonl)

    run_eval = not args.only_shap  # 평가/CM 수행 여부
    run_shap = (not args.no_shap) and (args.shap_k > 0)  # SHAP 수행 여부

# (A) evaluation
    if run_eval:
        t0 = [d["text0"] for d in test]
        t1 = [d["text1"] for d in test]
        y  = np.array([int(d["label"]) for d in test])

        scores = encode_pairs(model, t0, t1, batch_size=args.batch_size, show=True)
        auroc = roc_auc_score(y, scores)
        auprc = average_precision_score(y, scores)
        print(f"[TEST] AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    else:
        print("[INFO] Skipping evaluation (only_shap).")

    # (B) SHAP
    out_dir = args.shap_outdir or os.path.join(args.model_dir, "shap_bi_encoder")
    if run_shap:
        maybe_run_shap(model, test, args.shap_k, out_dir)
    else:
        print("[INFO] Skipping SHAP.")

    # (C) tau(EER) & Confusion Matrix
    if run_eval:
        if args.val_jsonl and os.path.exists(args.val_jsonl):
            val = read_jsonl(args.val_jsonl)
            vt0 = [d["text0"] for d in val]
            vt1 = [d["text1"] for d in val]
            vy  = np.array([int(d["label"]) for d in val])
            v_scores = encode_pairs(model, vt0, vt1, batch_size=args.batch_size, show=False)
            eer, tau = eer_and_tau(vy, v_scores)
            print(f"[VAL] EER={eer:.4f}, tau={tau:.6f}")
        else:
            eer, tau = eer_and_tau(y, scores)
            print(f"[WARN] No --val_jsonl provided. Using TEST to pick tau (leak). EER={eer:.4f}, tau={tau:.6f}")

        # Confusion Matrix
        pred = (scores >= tau).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])  # [[TN, FP],[FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        m = metrics_at_tau(y, scores, tau)
        print("[CONFUSION MATRIX] labels=[0(diff-author), 1(same-author)]")
        print(f"TN={tn}  FP={fp}\nFN={fn}  TP={tp}")
        print(f"[AT TAU] acc={m['acc_at_tau']:.4f}  precision={m['precision_at_tau']:.4f}  "
            f"recall={m['recall_at_tau']:.4f}  f1={m['f1_at_tau']:.4f}")

        os.makedirs(out_dir, exist_ok=True)
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"])
            plt.title("Confusion Matrix")
            plt.tight_layout()
            cm_path = os.path.join(out_dir, "confusion_matrix.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()
            print(f"[SAVE] Confusion matrix PNG -> {cm_path}")
        except Exception as e:
            print(f"[WARN] Could not save CM PNG: {e}")

if __name__ == "__main__":
    main()