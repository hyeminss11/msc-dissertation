# save as impersonation_script_flan_t5.py
# -*- coding: utf-8 -*-
"""
Impersonation with Flan-T5 (few-shot style imitation), ARTICLES ONLY + diverse support bundles.
- Targets only rows with genre == "article" (even if pairs contain tweet ids)
- Support texts are DISJOINT from target ids (default)
- Per-impersonator multiple support bundles; target id selects bundle deterministically
- Only authors with >= min_articles_impersonator articles can be impersonators
- Balances impersonator usage (least-used author chosen first)
- Safe to resume: skips IDs already written to --out_jsonl
- Optional sharding with --num_shards / --shard_id
- Saves impersonator usage statistics as a .txt file
"""

import os, json, math, argparse, random, hashlib
from typing import List, Set, Dict, Optional
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

RNG_SEED = 7
random.seed(RNG_SEED)

# -------------------------
# Helpers
# -------------------------
def load_json_any(path: str) -> pd.DataFrame:
    """Load array-JSON or JSONL into DataFrame."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def collect_needed_article_ids(pair_paths: List[str], df_docs: pd.DataFrame) -> Set[int]:
    """Collect ONLY article ids from pairs (ignore tweets)."""
    article_ids = set(
        df_docs.loc[df_docs["genre"].astype(str).str.lower() == "article", "id"]
        .astype(int).tolist()
    )
    ids: Set[int] = set()
    for p in pair_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                it = json.loads(line)
                for k in ("id0", "id1"):
                    if k in it:
                        _id = int(it[k])
                        if _id in article_ids:
                            ids.add(_id)
    return ids

def trim_words(text: str, max_words: Optional[int]) -> str:
    """Trim article text by words (if max_words > 0)."""
    if not max_words or max_words <= 0:
        return str(text)
    ws = str(text).split()
    return " ".join(ws[:max_words])

def create_impersonation_prompt(target_text: str, support_texts: List[str], impersonator: str) -> str:
    examples = "\n".join([f"{i+1}) {t}" for i, t in enumerate(support_texts)])
    return (
        f"Here are example writings of Author {impersonator}:\n"
        f"{examples}\n\n"
        f"Rewrite the following text so that it preserves its original meaning and\n"
        f"information, but imitates the phrasing, tone, and stylistic features of\n"
        f"Author {impersonator}:\n"
        f"Do NOT add prefaces or explanations.\n"
        f"Return only the rewritten text."
        f"{target_text}\n\n"
    )

def load_done_ids_from_jsonl(out_jsonl: str) -> Set[int]:
    """Load already processed ids for resume support."""
    done = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                it = json.loads(line)
                if "id" in it:
                    done.add(int(it["id"]))
            except Exception:
                pass
    return done

def choose_impersonator(candidates: List[str], usage_counts: Dict[str, int]) -> Optional[str]:
    """Pick least-used impersonator among candidates."""
    if not candidates:
        return None
    min_count = min(usage_counts.get(a, 0) for a in candidates)
    least_used = [a for a in candidates if usage_counts.get(a, 0) == min_count]
    return random.choice(least_used)

def deterministic_index(key: str, m: int) -> int:
    """Deterministic bundle selection based on SHA1 hash."""
    if m <= 0:
        return 0
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % m

def build_support_bundles(pool_ids: List[int], bundle_size: int, num_bundles: int) -> List[List[int]]:
    """Create multiple bundles of support ids per impersonator."""
    if num_bundles <= 0:
        num_bundles = 1
    pool = list(pool_ids)
    random.shuffle(pool)
    bundles: List[List[int]] = []
    if not pool:
        return [[] for _ in range(num_bundles)]
    idx = 0
    n = len(pool)
    for _ in range(num_bundles):
        bundle = []
        for _ in range(bundle_size):
            bundle.append(pool[idx % n])
            idx += 1
        bundles.append(bundle)
    return bundles

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_json", type=str, required=True)
    parser.add_argument("--pairs", type=str, nargs="+", required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
    parser.add_argument("--max_input_words", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--support_k", type=int, default=10)
    parser.add_argument("--support_bundles", type=int, default=4)
    parser.add_argument("--disjoint_support", action="store_true", default=True)
    parser.add_argument("--min_articles_impersonator", type=int, default=100)
    parser.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", "1")))
    parser.add_argument("--shard_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()

    print("[config]", vars(args), flush=True)

    # 1) Load docs
    df_docs = load_json_any(args.docs_json)
    for c in ["id", "author", "text", "genre"]:
        if c not in df_docs.columns:
            raise ValueError(f"Missing column: {c}")
    df_docs["id"] = pd.to_numeric(df_docs["id"], errors="coerce").astype("Int64")
    df_docs = df_docs.dropna(subset=["id"]).copy()
    df_docs["id"] = df_docs["id"].astype(int)
    df_docs["author"] = df_docs["author"].astype(str)
    df_docs["genre"] = df_docs["genre"].astype(str)
    df_docs["text"] = df_docs["text"].astype(str)

    # 2) Target set
    needed_ids = collect_needed_article_ids(args.pairs, df_docs)
    mask_article = df_docs["genre"].str.lower().eq("article")
    df_targets = df_docs.loc[mask_article & df_docs["id"].isin(needed_ids)].copy()
    df_targets = df_targets.sort_values("id").reset_index(drop=True)
    target_id_set = set(df_targets["id"].tolist())

    # 3) Impersonator pool
    only_articles = df_docs.loc[df_docs["genre"].str.lower()=="article"].copy()
    article_counts = only_articles.groupby("author")["id"].count().to_dict()
    allowed_impersonators = {a for a, cnt in article_counts.items() if cnt >= args.min_articles_impersonator}

    author_to_article_ids = defaultdict(list)
    for _, r in only_articles.iterrows():
        rid = int(r["id"])
        if args.disjoint_support and rid in target_id_set:
            continue
        author_to_article_ids[r["author"]].append(rid)

    # Prebuild support bundles
    impersonator_bundles: Dict[str, List[List[int]]] = {}
    for a in allowed_impersonators:
        pool_ids = author_to_article_ids.get(a, [])
        impersonator_bundles[a] = build_support_bundles(pool_ids, args.support_k, args.support_bundles)

    # 4) Sharding
    n = len(df_targets)
    num_shards = max(1, int(args.num_shards))
    shard_id = min(max(0, int(args.shard_id)), num_shards - 1)
    if num_shards > 1:
        chunk = math.ceil(n / num_shards)
        lo = shard_id * chunk
        hi = min(n, (shard_id + 1) * chunk)
        df_targets = df_targets.iloc[lo:hi].copy()

    if df_targets.empty:
        print("[exit] nothing to process.")
        return

    # 5) Resume
    done_ids = load_done_ids_from_jsonl(args.out_jsonl)

    # 6) Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16
    )

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    processed, skipped, errors = 0, 0, 0
    impersonation_counts: Dict[str, int] = defaultdict(int)

    with open(args.out_jsonl, "a", encoding="utf-8") as fout:
        for _, row in tqdm(df_targets.iterrows(), total=len(df_targets)):
            _id = int(row["id"])
            if _id in done_ids:
                skipped += 1
                continue

            target_text = str(row["text"]).strip()
            if not target_text:
                skipped += 1
                continue

            target_author = row["author"]
            candidates = [a for a in allowed_impersonators if a != target_author and impersonator_bundles.get(a)]
            impersonator = choose_impersonator(candidates, impersonation_counts)
            if not impersonator:
                skipped += 1
                continue

            bundles = impersonator_bundles[impersonator]
            b_idx = deterministic_index(f"{impersonator}|{_id}", len(bundles))
            sup_ids = bundles[b_idx]
            if not sup_ids:
                skipped += 1
                continue

            support_texts = [str(df_docs.loc[df_docs["id"] == sid, "text"].values[0]) for sid in sup_ids]
            text_trimmed = trim_words(target_text, args.max_input_words)
            prompt = create_impersonation_prompt(text_trimmed, support_texts, impersonator)

            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                out_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                # ✅ Only the model output goes into impersonated_text
                rec = {
                    "id": _id,
                    "author": target_author,
                    "impersonator": impersonator,
                    "impersonated_text": out_text,
                    "support_ids": sup_ids,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1
                impersonation_counts[impersonator] += 1

            except Exception as e:
                fout.write(json.dumps({
                    "id": _id,
                    "author": target_author,
                    "impersonator": impersonator,
                    "error": str(e)
                }, ensure_ascii=False) + "\n")
                fout.flush()
                errors += 1

    print(f"[done] processed={processed}, skipped={skipped}, errors={errors}")

    stats_txt = args.out_jsonl.replace(".jsonl", "_impersonator_stats.txt")
    with open(stats_txt, "w", encoding="utf-8") as fstats:
        for auth, cnt in sorted(impersonation_counts.items(), key=lambda x: -x[1]):
            fstats.write(f"{auth}\t{cnt}\n")

if __name__ == "__main__":
    main()