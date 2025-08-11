# quick_preproc_check.py
# Quick sanity checks for your paired jsonl datasets (no training required).

import json, hashlib
import argparse
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def text_hash(s: str):
    # Stable hash of normalized text (collapse whitespace)
    norm = " ".join((s or "").split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()

def basic_stats(df, name):
    print(f"\n[{name}] rows: {len(df)}")
    lbl_counts = Counter(df["label"])
    print(f"[{name}] label counts: {dict(lbl_counts)}  (pos%={lbl_counts.get(1,0)/max(1,len(df)):.3f})")
    # lengths
    lens0 = df["text0"].map(lambda x: len(str(x).split()))
    lens1 = df["text1"].map(lambda x: len(str(x).split()))
    print(f"[{name}] text0 avg words: {lens0.mean():.1f} (min={lens0.min()}, max={lens0.max()})")
    print(f"[{name}] text1 avg words: {lens1.mean():.1f} (min={lens1.min()}, max={lens1.max()})")

def check_self_pair(df, name):
    # id0 != id1 always
    bad = df[df["id0"] == df["id1"]]
    if len(bad):
        print(f"[{name}] ❌ self-pair found (id0==id1): {len(bad)}")
    else:
        print(f"[{name}] ✅ no self-pair (id0!=id1)")

def check_author_label_consistency(df, name):
    # label==1 → same author; label==0 → different authors
    same_author = df["author0"] == df["author1"]
    ok_pos = df[same_author & (df["label"]==1)]
    bad_pos = df[same_author & (df["label"]==0)]
    ok_neg = df[(~same_author) & (df["label"]==0)]
    bad_neg = df[(~same_author) & (df["label"]==1)]
    if len(bad_pos) or len(bad_neg):
        print(f"[{name}] ❌ author/label mismatch: bad_pos={len(bad_pos)} bad_neg={len(bad_neg)}")
    else:
        print(f"[{name}] ✅ author/label consistent")

def check_id_overlap(train, val):
    # No document id should appear in both splits
    ids_train = set(train["id0"]).union(set(train["id1"]))
    ids_val   = set(val["id0"]).union(set(val["id1"]))
    inter = ids_train & ids_val
    print(f"\n[split] shared doc ids between train/val: {len(inter)}")
    if inter:
        print("Examples:", list(sorted(inter))[:10])
        print("❌ Leakage by document ids.")
    else:
        print("✅ No doc-id leakage between train and val.")

def check_text_overlap(train, val):
    # Detect near-exact text duplicates across splits using hashes
    t_hashes_train = set(train["text0"].map(text_hash)) | set(train["text1"].map(text_hash))
    t_hashes_val   = set(val["text0"].map(text_hash))   | set(val["text1"].map(text_hash))
    inter = t_hashes_train & t_hashes_val
    print(f"[split] shared text hashes between train/val: {len(inter)}")
    print("✅ No (near-)exact text duplicates across splits." if not inter else "⚠️ Some texts look identical across splits.")

def check_per_doc_balance(df, name):
    # Roughly verify that each doc participates in both pos/neg similarly
    pos_counts = defaultdict(int)
    neg_counts = defaultdict(int)
    for _, r in df.iterrows():
        a, b, y = r["id0"], r["id1"], int(r["label"])
        if y==1:
            pos_counts[a]+=1; pos_counts[b]+=1
        else:
            neg_counts[a]+=1; neg_counts[b]+=1
    diffs = []
    for k in set(list(pos_counts.keys())+list(neg_counts.keys())):
        diffs.append(pos_counts[k] - neg_counts[k])
    if not diffs:
        print(f"[{name}] (skip) per-doc balance: empty dataset?")
        return
    avg_abs = sum(abs(d) for d in diffs)/len(diffs)
    print(f"[{name}] per-doc pos-neg diff avg(abs): {avg_abs:.3f}  (lower is more balanced)")
    # soft flag
    if avg_abs > 1.5:
        print(f"[{name}] ⚠️ Pairs look imbalanced per doc (not necessarily a bug, but check your generation logic).")
    else:
        print(f"[{name}] ✅ Per-doc pairing looks reasonably balanced.")

def show_examples(df, k=3, name="train"):
    print(f"\n[{name}] sample pairs:")
    for i, (_, r) in enumerate(df.sample(n=min(k, len(df)), random_state=7).iterrows(), start=1):
        print(f"--- example {i} (label={r['label']}, a0={r['author0']}, a1={r['author1']}) ---")
        print("text0:", (r["text0"][:160] + "…") if len(r["text0"])>160 else r["text0"])
        print("text1:", (r["text1"][:160] + "…") if len(r["text1"])>160 else r["text1"])

def main(args):
    train_df = read_jsonl(args.train)
    val_df   = read_jsonl(args.val)

    # Basic stats
    basic_stats(train_df, "train")
    basic_stats(val_df, "val")

    # Intra-split checks
    for df, name in [(train_df, "train"), (val_df, "val")]:
        check_self_pair(df, name)
        check_author_label_consistency(df, name)
        check_per_doc_balance(df, name)

    # Cross-split leakage checks
    check_id_overlap(train_df, val_df)
    check_text_overlap(train_df, val_df)

    # Show some samples to eyeball anonymization/merging quality
    show_examples(train_df, k=3, name="train")
    show_examples(val_df, k=3, name="val")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick preprocessing sanity checks for AV pair datasets.")
    parser.add_argument("--train", required=True, help="Path to train jsonl (paired).")
    parser.add_argument("--val",   required=True, help="Path to val jsonl (paired).")
    args = parser.parse_args()
    main(args)