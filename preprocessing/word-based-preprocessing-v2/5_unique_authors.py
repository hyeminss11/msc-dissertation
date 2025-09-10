# save as count_articles_in_pairs.py
# -*- coding: utf-8 -*-
"""
Print per-author ARTICLE counts restricted to docs that appear in given pair files.
"""

import json, argparse
import pandas as pd

def load_json_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def collect_pair_ids(pair_paths) -> set:
    ids = set()
    for p in pair_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                if "id0" in it: ids.add(int(it["id0"]))
                if "id1" in it: ids.add(int(it["id1"]))
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_json", required=True,
                    help="Docs JSON/JSONL with columns: id,author,genre")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="One or more pair JSONL files (with id0, id1)")
    args = ap.parse_args()

    # load docs
    df = load_json_any(args.docs_json)
    for c in ["id","author","genre"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    df["author"] = df["author"].astype(str)
    df["genre"] = df["genre"].astype(str)

    # ids that appear in pairs
    pair_ids = collect_pair_ids(args.pairs)
    if not pair_ids:
        print("[warn] No IDs collected from pairs.")
        return

    df_pair = df[df["id"].isin(pair_ids)].copy()
    df_pair_articles = df_pair[df_pair["genre"].str.lower() == "article"].copy()

    # count per author
    res = (
        df_pair_articles.groupby("author")["id"]
        .nunique()
        .reset_index(name="article_count_in_pairs")
        .sort_values(["article_count_in_pairs","author"], ascending=[False, True])
        .reset_index(drop=True)
    )

    print("author\tarticle_count_in_pairs")
    for _, r in res.iterrows():
        print(f"{r['author']}\t{int(r['article_count_in_pairs'])}")

    print(f"[rows] {len(res)} authors")

if __name__ == "__main__":
    main()