import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ===== Config =====
COLORS = {"DistilBERT": "#eeeeee", "RoBERTa": "#797979", "BigBird": "#000076"}
EDGE_COLOR = "#787878dc"
EDGE_WIDTH = 0.5
BAR_WIDTH  = 0.27

# SHAP .npz paths per model
paths = {
    "DistilBERT": "shap/rq3/shap_text_di_at_imp.npz",
    "RoBERTa":    "shap/rq3/shap_text_ro_at_imp.npz",
    "BigBird":    "shap/rq3/shap_text_bb_at_imp.npz",
}

# Plot hyper-parameters
GLOBAL_TOPK   = 15     # for global plots (tokens)
SAMPLE_TOPK   = 20     # for per-sample plots
model_order   = ["DistilBERT", "RoBERTa", "BigBird"]
sample_idx    = 5      # which sample to inspect for per-sample plots
class_idx     = 1      # 0 = different-author, 1 = same-author

# ===== Token normalization helpers =====
def normalize_roberta_token(tok: str) -> str:
    """RoBERTa: 'Ġ' denotes a preceding space; remove it to get raw form."""
    return tok.replace("Ġ", "")

def normalize_bert_piecewise(tokens):
    """BERT-family (WordPiece): merge '##' continuations into a single word."""
    words = []
    for t in tokens:
        if t.startswith("##") and words:
            words[-1] = words[-1] + t[2:]
        else:
            words.append(t)
    return words

def normalize_tokens(model_name, token_list):
    """
    Normalize tokens to comparable 'word-like' forms across models.
    - RoBERTa: strip 'Ġ'
    - BERT family: merge '##' pieces
    - Lowercase and drop empty/space tokens.
    NOTE: This is a heuristic; for exact word merges see merge_tokens_to_words().
    """
    if "roberta" in model_name.lower():
        toks = [normalize_roberta_token(t) for t in token_list]
    else:
        toks = normalize_bert_piecewise(token_list)
    toks = [t for t in toks if t and not t.isspace()]
    toks = [t.lower() for t in toks]
    return toks

# ===== Utility =====
def minmax(d: dict):
    """Per-model min–max normalize values to [0,1]."""
    if not d:
        return d
    vals = np.array(list(d.values()))
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax == vmin:
        return {k: 0.0 for k in d.keys()}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}

# ===== (A) GLOBAL: token-level common influential tokens across models =====
token_importance = {}  # {model_name: defaultdict(float)}

for model_name, npz_path in paths.items():
    npz = np.load(npz_path, allow_pickle=True)
    values = npz["values"]  # (num_samples, num_tokens, 2)
    tokens = npz["data"]    # (num_samples, num_tokens) object array of strings
    cls_idx = 1             # index for "same-author" class (for importance)

    agg = defaultdict(float)
    for i in range(values.shape[0]):
        tok_list    = list(tokens[i])
        shap_scores = values[i][:, cls_idx]  # (num_tokens,)

        # token-level normalization (approximate word-like forms)
        norm_tokens = normalize_tokens(model_name, tok_list)

        # guard for length mismatch after normalization
        L = min(len(norm_tokens), len(shap_scores))
        for t, s in zip(norm_tokens[:L], shap_scores[:L]):
            agg[t] += abs(float(s))  # accumulate |SHAP|
    token_importance[model_name] = agg

# per-model min–max
norm_importance = {m: minmax(imp) for m, imp in token_importance.items()}

# intersection across models
models = list(norm_importance.keys())
sets = [set(norm_importance[m].keys()) for m in models]
common_tokens = set.intersection(*sets)

# pick top-K by average normalized importance
avg_scores = {t: np.mean([norm_importance[m].get(t, 0.0) for m in models]) for t in common_tokens}
topK = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:GLOBAL_TOPK]
top_tokens = [t for t, _ in topK]

# plot (global, token-level)
x = np.arange(len(top_tokens))
plt.figure(figsize=(12, 5))
for i, m in enumerate(models):
    y = [norm_importance[m].get(t, 0.0) for t in top_tokens]
    plt.bar(x + i*BAR_WIDTH, y, BAR_WIDTH,
            label=m, color=COLORS.get(m, None),
            edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

plt.xticks(x + BAR_WIDTH, top_tokens, rotation=45, ha='right')
plt.ylabel("Normalized importance (|SHAP|)")
plt.title("Common influential tokens across models (global)")
plt.legend()
plt.tight_layout()
plt.savefig("global_common_tokens.png", dpi=300)
plt.close()

# ===== (B) PER-SAMPLE: token-level common influential tokens (one sample) =====
def load_one_sample_token_scores(npz_path, model_name, sample_idx, class_idx):
    """Return dict token->SHAP (signed) for a single sample/class (token-level)."""
    npz = np.load(npz_path, allow_pickle=True)
    vals = npz["values"][sample_idx]          # (num_tokens, 2)
    toks = list(npz["data"][sample_idx])      # (num_tokens,)
    scores = vals[:, class_idx]               # (num_tokens,)
    ntoks = normalize_tokens(model_name, toks)
    L = min(len(ntoks), len(scores))
    ntoks, scores = ntoks[:L], scores[:L]
    d = defaultdict(float)
    for t, s in zip(ntoks, scores):
        d[t] += float(s)  # sum if duplicates after normalization
    return dict(d)

per_model_tok = {m: load_one_sample_token_scores(paths[m], m, sample_idx, class_idx)
                 for m in model_order}
sets_tok = [set(per_model_tok[m].keys()) for m in model_order]
common_tok_sample = set.intersection(*sets_tok)

if common_tok_sample:
    avg_abs = {t: np.mean([abs(per_model_tok[m].get(t, 0.0)) for m in model_order])
               for t in common_tok_sample}
    top_tokens_sample = [t for t,_ in sorted(avg_abs.items(), key=lambda x: x[1], reverse=True)[:SAMPLE_TOPK]]

    x = np.arange(len(top_tokens_sample))
    plt.figure(figsize=(12, 5))
    for i, m in enumerate(model_order):
        y = [per_model_tok[m].get(t, 0.0) for t in top_tokens_sample]
        plt.bar(x + i*BAR_WIDTH, y, BAR_WIDTH,
                label=m, color=COLORS[m],
                edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    plt.xticks(x + BAR_WIDTH, top_tokens_sample, rotation=45, ha='right')
    plt.ylabel(f"SHAP contribution (class {class_idx})")
    plt.title(f"Common tokens (sample={sample_idx}, class={class_idx})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sample{sample_idx}_class{class_idx}_common_tokens.png", dpi=300)
    plt.close()
else:
    print("[WARN] No common tokens across models for the chosen sample at token-level.")

# ===== (C) PER-SAMPLE: word-level (merge subwords) common influential words =====
def merge_tokens_to_words(model_name, tokens, scores):
    """
    Merge subword tokens into whole words and aggregate SHAP scores accordingly.
    Returns:
      words (list[str]), wvals (list[float])
    """
    skip = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "[UNK]"}
    toks, vals = [], []
    for t, s in zip(tokens, scores):
        if t in skip or t.strip() == "":
            continue
        toks.append(t)
        vals.append(float(s))

    words, wvals = [], []
    if "roberta" in model_name.lower():
        # RoBERTa: 'Ġ' indicates a new word boundary
        cur_word, cur_sum = "", 0.0
        for t, v in zip(toks, vals):
            if t.startswith("Ġ"):
                if cur_word:
                    words.append(cur_word.lower()); wvals.append(cur_sum)
                cur_word = t[1:]; cur_sum = v
            else:
                cur_word += t; cur_sum += v
        if cur_word:
            words.append(cur_word.lower()); wvals.append(cur_sum)
    else:
        # BERT-family: '##' indicates continuation
        cur_word, cur_sum = "", 0.0
        for t, v in zip(toks, vals):
            if t.startswith("##"):
                cur_word += t[2:]; cur_sum += v
            else:
                if cur_word:
                    words.append(cur_word.lower()); wvals.append(cur_sum)
                cur_word = t; cur_sum = v
        if cur_word:
            words.append(cur_word.lower()); wvals.append(cur_sum)

    keep = [(w, s) for w, s in zip(words, wvals) if w.strip()]
    if not keep:
        return [], []
    words, wvals = zip(*keep)
    return list(words), list(wvals)

def load_sample_word_scores(npz_path, model_name, sample_idx, class_idx):
    """Return dict word->SHAP (signed) for a single sample/class (word-level merged)."""
    npz = np.load(npz_path, allow_pickle=True)
    vals   = npz["values"][sample_idx]     # (num_tokens, 2)
    toks   = list(npz["data"][sample_idx]) # (num_tokens,)
    scores = vals[:, class_idx]            # (num_tokens,)
    words, wvals = merge_tokens_to_words(model_name, toks, scores)
    d = defaultdict(float)
    for w, s in zip(words, wvals):
        d[w] += s
    return dict(d)

per_model_word = {m: load_sample_word_scores(paths[m], m, sample_idx, class_idx)
                  for m in model_order}
sets_word = [set(per_model_word[m].keys()) for m in model_order]
common_words = set.intersection(*sets_word)

if common_words:
    avg_abs_w = {w: np.mean([abs(per_model_word[m].get(w, 0.0)) for m in model_order])
                 for w in common_words}
    top_words = [w for w,_ in sorted(avg_abs_w.items(), key=lambda x: x[1], reverse=True)[:SAMPLE_TOPK]]

    x = np.arange(len(top_words))
    plt.figure(figsize=(12, 5))
    for i, m in enumerate(model_order):
        y = [per_model_word[m].get(w, 0.0) for w in top_words]
        plt.bar(x + i*BAR_WIDTH, y, BAR_WIDTH,
                label=m, color=COLORS[m],
                edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    plt.xticks(x + BAR_WIDTH, top_words, rotation=45, ha='right')
    plt.ylabel(f"SHAP contribution (class {class_idx})")
    plt.title(f"Common influential words (sample={sample_idx}, class={class_idx})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sample{sample_idx}_class{class_idx}_common_words.png", dpi=300)
    plt.close()
else:
    print("[WARN] No common words across models for the chosen sample at word-level.")