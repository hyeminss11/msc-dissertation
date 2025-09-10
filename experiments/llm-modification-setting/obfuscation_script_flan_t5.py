# save as obfuscation_from_original_for_impersonated_ids.py
# -*- coding: utf-8 -*-
"""
Obfuscation (paraphrasing) for *original articles* whose IDs appeared in an impersonation run.
- Reads IDs from --impersonation_jsonl (skips rows without id)
- Looks up the ORIGINAL article text in --docs_json by id (ignores impersonated_text)
- Paraphrases only those originals
- Safe to resume: skips IDs already present in --out_jsonl
- Optional sharding
"""

import os
import json
import math
import argparse
from typing import List, Set, Dict, Optional
import glob

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Helpers
# -------------------------
def load_json_any(path: str) -> pd.DataFrame:
    """Load array-JSON or JSONL into DataFrame."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def load_impersonated_ids(path: str) -> Set[int]:
    """Collect unique ids from an impersonation JSONL (one record per target id)."""
    ids: Set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                it = json.loads(line)
                if "id" in it:
                    ids.add(int(it["id"]))
            except Exception:
                # ignore broken lines
                pass
    return ids

def load_done_ids_from_jsonl(out_jsonl: str) -> Set[int]:
    """Resume support: read already processed ids from output JSONL."""
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
def load_done_ids_from_many(paths) -> Set[int]:
    out = set()
    if not paths:
        return out
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        if not p:
            continue
        if any(ch in p for ch in ["*", "?", "["]):  # glob pattern
            for f in glob.glob(p):
                out |= load_done_ids_from_jsonl(f)
        elif os.path.exists(p):
            out |= load_done_ids_from_jsonl(p)
    return out

def trim_words(text: str, max_words: Optional[int]) -> str:
    """Cut by words to avoid overly long inputs to LLM (None/<=0 -> no cut)."""
    if not max_words or max_words <= 0:
        return str(text)
    ws = str(text).split()
    return " ".join(ws[:max_words])

def create_paraphrase_prompt(text_to_paraphrase: str) -> str:
    """Prompt for paraphrasing with Flan-T5 style models."""
    return (
        "Paraphrase the following ARTICLE while preserving its meaning and factual content. "
        "Avoid adding new information.\n\n"
        f"{text_to_paraphrase}"
    )

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_json", type=str, required=True,
                        help="Docs JSON/JSONL with columns id, text, genre (original corpus).")
    parser.add_argument("--impersonation_jsonl", type=str, required=True,
                        help="Impersonation output JSONL. IDs in this file determine which originals to obfuscate.")
    parser.add_argument("--out_jsonl", type=str, required=True,
                        help="Output JSONL (append-friendly, resumable).")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large",
                        help="HuggingFace model name (e.g., google/flan-t5-large).")
    parser.add_argument("--max_input_words", type=int, default=800,
                        help="Cut input article by words before sending to LLM (None/<=0 = no cut).")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--progress_every", type=int, default=50)

    parser.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", "1")))
    parser.add_argument("--shard_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    # argparse에 추가
    parser.add_argument(
        "--resume_jsonl",
        type=str,
        default=None,
        help="Optional: global JSONL to use for skipping already-processed IDs (e.g., merged file)."
    )
    parser.add_argument(
        "--resume_glob",
        type=str,
        default=None,
        help="Optional: glob pattern of JSONL files to skip IDs from (e.g., '/path/out_shard_*.jsonl')."
    )
    args = parser.parse_args()

    print("[config]", vars(args), flush=True)

    # 1) Load originals
    print(f"[load] docs: {args.docs_json}")
    df_docs = load_json_any(args.docs_json)
    for c in ["id", "text", "author", "genre"]:
        if c not in df_docs.columns:
            raise ValueError(f"Missing required column in docs: {c}")
    df_docs["id"] = pd.to_numeric(df_docs["id"], errors="coerce").astype("Int64")
    df_docs = df_docs.dropna(subset=["id"]).copy()
    df_docs["id"] = df_docs["id"].astype(int)
    df_docs["author"] = df_docs["author"].astype(str)
    df_docs["text"] = df_docs["text"].astype(str)
    df_docs["genre"] = df_docs["genre"].astype(str)

    # 2) IDs to process come from impersonation JSONL
    target_ids = load_impersonated_ids(args.impersonation_jsonl)
    print(f"[ids] from impersonation file: {len(target_ids)} unique ids")

    # 3) Keep only ORIGINAL *articles* for those ids
    mask_article = df_docs["genre"].str.lower().eq("article")
    df = df_docs.loc[mask_article & df_docs["id"].isin(target_ids)].copy()
    df = df.sort_values("id").reset_index(drop=True)
    print(f"[filter] original articles to obfuscate: {len(df)}")

    # 4) Sharding
    n = len(df)
    num_shards = max(1, int(args.num_shards))
    shard_id = min(max(0, int(args.shard_id)), num_shards - 1)
    if num_shards > 1:
        chunk = math.ceil(n / num_shards)
        lo = shard_id * chunk
        hi = min(n, (shard_id + 1) * chunk)
        df = df.iloc[lo:hi].copy()
        print(f"[shard] {shard_id}/{num_shards} -> rows {lo}..{hi-1} (kept {len(df)})")

    if df.empty:
        print("[exit] nothing to process after filtering/sharding.")
        return

    # 5) Resume support
    done_ids = load_done_ids_from_jsonl(args.out_jsonl)
    # NEW: merge in global/other sources to skip
    done_ids |= load_done_ids_from_many(args.resume_jsonl)
    done_ids |= load_done_ids_from_many(args.resume_glob)
    print(f"[resume] already done (merged): {len(done_ids)}")
    print(f"[resume] already done: {len(done_ids)}")

    # 6) Load model
    print(f"[model] loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 7) Main loop
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    processed, skipped, errors = 0, 0, 0

    with open(args.out_jsonl, "a", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            _id = int(row["id"])
            if _id in done_ids:
                skipped += 1
                continue

            original_text = row["text"].strip()
            if not original_text:
                skipped += 1
                continue

            text_trimmed = trim_words(original_text, args.max_input_words)
            prompt = create_paraphrase_prompt(text_trimmed)

            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                rec = {
                    "id": _id,
                    "author": row["author"],
                    "obfuscated_text": out_text
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1
            except Exception as e:
                rec = {"id": _id, "obfuscated_from": "original_article", "error": str(e)}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                errors += 1

            total_seen = processed + skipped
            if args.progress_every and total_seen % args.progress_every == 0:
                print(f"[progress] processed={processed} skipped={skipped} errors={errors}", flush=True)

    print(f"[done] processed={processed} skipped={skipped} errors={errors}")
    print(f"[out] {args.out_jsonl}")

if __name__ == "__main__":
    main()