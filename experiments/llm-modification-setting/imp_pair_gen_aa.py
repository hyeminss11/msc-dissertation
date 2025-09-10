# save as build_pairs_modified_vs_original_AA.py
# -*- coding: utf-8 -*-
import os, json, argparse, random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

RNG = 7
random.seed(RNG)

def load_json_any(path: str) -> pd.DataFrame:
    """Load either JSON Lines (JSONL) or array-JSON into a DataFrame."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def length_bin(n: int):
    """Coarse length buckets (by character count) for negative sampling control."""
    if n < 200: return "0-199"
    if n < 400: return "200-399"
    if n < 800: return "400-799"
    return "800+"

def load_done_source_ids(out_jsonl: str) -> set[int]:
    """
    Resume helper: scan existing out_jsonl and collect all id0 values (source modified-article ids)
    that already have pairs written (positive/negative). If id0 is present once, we skip generating
    pairs for that source id on resume.
    """
    done: set[int] = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
                if "id0" in it:
                    done.add(int(it["id0"]))
            except Exception:
                pass
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_json", required=True, help="Original corpus with columns: id, author, text, genre.")
    ap.add_argument("--modified_jsonl", required=True, help="LLM-modified results JSONL (impersonation or obfuscation).")
    ap.add_argument("--out_jsonl", required=True, help="Output pairs JSONL.")
    ap.add_argument("--pair_tag", default="AA-IMP", choices=["AA-IMP","AA-OBF"], help="Metadata tag for pair type.")
    ap.add_argument("--neg_ratio", type=int, default=1, help="# negatives per positive.")
    ap.add_argument("--neg_per_author_cap", type=int, default=50, help="Reuse cap for negatives (per candidate row).")
    ap.add_argument("--swap_prob", type=float, default=0.5,
                help="Probability of swapping the left/right model inputs (left_text/right_text).")
    args = ap.parse_args()

    # 1) Load original corpus
    df = load_json_any(args.docs_json)
    for c in ["id","author","text","genre"]:
        if c not in df.columns:
            raise ValueError(f"docs_json missing column: {c}")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    df["author"] = df["author"].astype(str)
    df["genre"] = df["genre"].astype(str)
    df["text"] = df["text"].astype(str)

    # Articles only + precompute length bins
    arts = df[df["genre"].str.lower()=="article"].copy()
    if arts.empty:
        raise ValueError("No articles found in docs_json (genre=='article').")
    arts["len_bin"] = arts["text"].str.len().map(length_bin)

    # Quick lookups
    id2row = arts.set_index("id").to_dict(orient="index")
    # For positives: same-author pools (exclude same id later)
    author_to_idx = defaultdict(list)
    for i, r in arts.iterrows():
        author_to_idx[r["author"]].append(i)
    # For negatives: bin→author→indices
    bin_to_indices_by_author = defaultdict(lambda: defaultdict(list))
    for i, r in arts.iterrows():
        bin_to_indices_by_author[r["len_bin"]][r["author"]].append(i)

    # 2) Load LLM-modified records
    mod_records = []
    with open(args.modified_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                it = json.loads(line)
            except Exception:
                continue
            text_key = None
            for k in ["impersonated_text","obfuscated_text","paraphrased_text"]:
                if k in it:
                    text_key = k
                    break
            if "id" not in it or text_key is None:
                continue
            mod_records.append((int(it["id"]), str(it[text_key]), it))
    total_modified = len(mod_records)
    print(f"[load] modified records: {total_modified}")

    # 3) Resume scan
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    done_ids = load_done_source_ids(args.out_jsonl)
    print(f"[resume] already paired source ids in out_jsonl: {len(done_ids)}")

    # 4) Build pairs (append mode)
    out = open(args.out_jsonl, "a", encoding="utf-8")

    neg_use_count = defaultdict(int)
    pos_cnt = neg_cnt = 0
    skipped_due_to_resume = 0
    missing_in_corpus = 0
    no_pos_alt = 0

    for art_id, mod_text, meta in tqdm(mod_records):
        # Resume: skip if already processed
        if art_id in done_ids:
            skipped_due_to_resume += 1
            continue

        base = id2row.get(art_id)
        if base is None:
            missing_in_corpus += 1
            continue

        author = base["author"]
        bin_mod = length_bin(len(mod_text))

        # ---------- POSITIVE: same author, DIFFERENT article (exclude id1 == id0) ----------
        same_author_idxs = [i for i in author_to_idx[author] if int(arts.loc[i, "id"]) != art_id]
        # Prefer same length bin as modified text
        same_bin_candidates = [i for i in same_author_idxs if arts.loc[i, "len_bin"] == bin_mod]
        pos_pool = same_bin_candidates if same_bin_candidates else same_author_idxs
        if not pos_pool:
            # no alternative doc for this author → skip this source
            no_pos_alt += 1
            continue
        pos_i = random.choice(pos_pool)
        pos_row = arts.loc[pos_i]

        rec_pos = {
            "id0": art_id,
            "id1": int(pos_row["id"]),           # different id
            "text0": mod_text,
            "text1": pos_row["text"],
            "label": 1,
            "author0": author,
            "author1": author,
            "genre0": "modified_article",
            "genre1": "article",
            "pair_type": args.pair_tag
        }

        swap = (random.random() < args.swap_prob)
        rec_pos["swap"] = bool(swap)
        if swap:
            rec_pos["left_text"], rec_pos["right_text"] = rec_pos["text1"], rec_pos["text0"]
        else:
            rec_pos["left_text"], rec_pos["right_text"] = rec_pos["text0"], rec_pos["text1"]
        for k in ["impersonator","support_ids","obfuscated_from"]:
            if k in meta:
                rec_pos[k] = meta[k]
        out.write(json.dumps(rec_pos, ensure_ascii=False) + "\n")
        pos_cnt += 1

        # ---------- NEGATIVES: different author (avoid id1 == id0 just in case) ----------
        need = args.neg_ratio

        # Pool 1: same length bin, different author
        candidates = []
        for au, idxs in bin_to_indices_by_author[bin_mod].items():
            if au == author:
                continue
            for i in idxs:
                if int(arts.loc[i, "id"]) != art_id:
                    candidates.append(i)
        random.shuffle(candidates)

        # Pool 2: any different author
        backup = [i for i in arts.index if arts.loc[i,"author"] != author and int(arts.loc[i,"id"]) != art_id]
        random.shuffle(backup)

        chosen = 0
        for pool in (candidates, backup):
            for i in pool:
                if chosen >= need:
                    break
                if neg_use_count[i] >= args.neg_per_author_cap:
                    continue
                au_n = arts.loc[i,"author"]
                rec_neg = {
                    "id0": art_id,
                    "id1": int(arts.loc[i,"id"]),
                    "text0": mod_text,
                    "text1": arts.loc[i,"text"],
                    "label": 0,
                    "author0": author,
                    "author1": au_n,
                    "genre0": "modified_article",
                    "genre1": "article",
                    "pair_type": args.pair_tag
                }

                swap = (random.random() < args.swap_prob)
                rec_neg["swap"] = bool(swap)
                if swap:
                    rec_neg["left_text"], rec_neg["right_text"] = rec_neg["text1"], rec_neg["text0"]
                else:
                    rec_neg["left_text"], rec_neg["right_text"] = rec_neg["text0"], rec_neg["text1"]

                for k in ["impersonator","support_ids","obfuscated_from"]:
                    if k in meta:
                        rec_neg[k] = meta[k]
                out.write(json.dumps(rec_neg, ensure_ascii=False) + "\n")
                neg_use_count[i] += 1
                chosen += 1
                neg_cnt += 1
            if chosen >= need:
                break

    out.close()
    print(f"[done] wrote -> {args.out_jsonl}")
    print(f"[counts] positives={pos_cnt}, negatives={neg_cnt}, "
          f"skipped_resume={skipped_due_to_resume}, missing_in_corpus={missing_in_corpus}, no_pos_alt={no_pos_alt}")

    # 5) Save stats file
    stats_path = args.out_jsonl.replace(".jsonl", "_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Pair generation statistics (AA, modified vs original — same-author, different-article)\n")
        f.write("=====================================================================================\n")
        f.write(f"docs_json: {args.docs_json}\n")
        f.write(f"modified_jsonl: {args.modified_jsonl}\n")
        f.write(f"out_jsonl: {args.out_jsonl}\n")
        f.write(f"pair_tag: {args.pair_tag}\n")
        f.write(f"neg_ratio: {args.neg_ratio}\n")
        f.write(f"neg_per_author_cap: {args.neg_per_author_cap}\n")
        f.write("\n")
        f.write(f"Total modified records read: {total_modified}\n")
        f.write(f"Already present (skipped by resume): {skipped_due_to_resume}\n")
        f.write(f"Missing in original corpus: {missing_in_corpus}\n")
        f.write(f"No same-author alternative article (skipped): {no_pos_alt}\n")
        f.write(f"Positives written: {pos_cnt}\n")
        f.write(f"Negatives written: {neg_cnt}\n")
        f.write(f"Total pairs written: {pos_cnt + neg_cnt}\n")
    print(f"[stats] saved -> {stats_path}")

if __name__ == "__main__":
    main()