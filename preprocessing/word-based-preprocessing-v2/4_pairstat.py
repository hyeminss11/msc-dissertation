# pair_stats_table.py
import os, re, glob, pandas as pd

# === directory ===
BASE_DIR = "preprocessing/word-based-preprocessing-v2/pairs_balanced_json"
OUT_TXT  = os.path.join(BASE_DIR, "pair_stats_summary.txt")

# extract split and mode: ...(silver|gold)_(AA|TT|AT)_(train|val|test).jsonl
PATTERN = re.compile(r".*(silver|gold)_(AA|TT|AT)_(train|val|test)\.jsonl$")

def load_df(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)

rows = []
files = sorted(glob.glob(os.path.join(BASE_DIR, "*.jsonl")))
for fp in files:
    m = PATTERN.match(fp)
    if not m:
        continue
    dataset, mode, split = m.groups()
    try:
        df = load_df(fp)
    except ValueError:
        rows.append({
            "dataset": dataset, "mode": mode, "split": split,
            "pairs_total": 0, "pairs_pos": 0, "pairs_neg": 0,
            "unique_docs": 0, "unique_authors": 0
        })
        continue

    # count
    total = len(df)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())

    # unique docs and authors
    uniq_docs = len(set(df["id0"]).union(set(df["id1"])))
    uniq_auth = len(set(df["author0"]).union(set(df["author1"])))

    rows.append({
        "dataset": dataset,     # silver / gold
        "mode": mode,           # AA / TT / AT
        "split": split,         # train / val / test
        "pairs_total": total,
        "pairs_pos": pos,
        "pairs_neg": neg,
        "pos_ratio": round(pos / total, 3) if total else 0.0,
        "unique_docs": uniq_docs,
        "unique_authors": uniq_auth,
    })

# make the table
summary = pd.DataFrame(rows).sort_values(["dataset","mode","split"]).reset_index(drop=True)

# pivot
pivot = summary.pivot_table(
    index=["dataset","mode"],
    columns="split",
    values=["pairs_total","pairs_pos","pairs_neg","unique_docs","unique_authors"],
    aggfunc="first"
)

# save as txt (readable table)
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== Raw summary ===\n")
    f.write(summary.to_string(index=False))
    f.write("\n\n=== Pivot summary (dataset/mode x split) ===\n")
    f.write(pivot.to_string())

print(f"\nSaved TXT -> {OUT_TXT}")