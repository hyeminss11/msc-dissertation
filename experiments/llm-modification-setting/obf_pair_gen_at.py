# save as imp_pair_gen_at.py
# -*- coding: utf-8 -*-
import os, json, argparse, random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

RNG = 7
random.seed(RNG)

def load_json_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def length_bin_tweet(n: int):
    if n < 80: return "0-79"
    if n < 160: return "80-159"
    if n < 320: return "160-319"
    return "320+"

def load_done_source_ids(out_jsonl: str) -> set[int]:
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
    ap.add_argument("--docs_json", required=True,
                    help="Corpus with id, author, text, genre (must contain articles & tweets).")
    ap.add_argument("--modified_jsonl", required=True,
                    help="LLM-modified articles JSONL.")
    ap.add_argument("--out_jsonl", required=True, help="Output pairs JSONL.")
    ap.add_argument("--pair_tag", default="AT-IMP", choices=["AT-IMP","AT-OBF"])
    ap.add_argument("--neg_ratio", type=int, default=1)
    ap.add_argument("--neg_per_author_cap", type=int, default=50)
    ap.add_argument("--swap_prob", type=float, default=0.5,
                    help="Probability of swapping left/right inputs in the pair.")
    args = ap.parse_args()

    # --- Load corpus ---
    df = load_json_any(args.docs_json)
    for c in ["id","author","text","genre"]:
        if c not in df.columns:
            raise ValueError(f"docs_json missing column: {c}")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    df["author"] = df["author"].astype(str)
    df["genre"] = df["genre"].astype(str)
    df["text"]  = df["text"].astype(str)

    arts   = df[df["genre"].str.lower()=="article"].copy()
    tweets = df[df["genre"].str.lower()=="tweet"].copy()
    if arts.empty or tweets.empty:
        raise ValueError("docs_json must contain both article & tweet.")

    id2row_article = arts.set_index("id").to_dict(orient="index")

    tweets["len_bin"] = tweets["text"].str.len().map(length_bin_tweet)
    author_to_tweet_idxs = defaultdict(list)
    for i, r in tweets.iterrows():
        author_to_tweet_idxs[r["author"]].append(i)

    bin_to_idxs_by_author = defaultdict(lambda: defaultdict(list))
    for i, r in tweets.iterrows():
        bin_to_idxs_by_author[r["len_bin"]][r["author"]].append(i)

    # --- Load modified records ---
    mod_records = []
    with open(args.modified_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
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
    print(f"[load] modified article records: {total_modified}")

    # --- Resume set ---
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    done_ids = load_done_source_ids(args.out_jsonl)
    print(f"[resume] already paired source ids in out_jsonl: {len(done_ids)}")

    # --- Build pairs (append) ---
    out = open(args.out_jsonl, "a", encoding="utf-8")
    pos_cnt = neg_cnt = 0
    neg_use_count = defaultdict(int)
    skipped_due_to_resume = 0
    missing_in_corpus = 0
    no_pos_tweet = 0

    for art_id, mod_text, meta in tqdm(mod_records):
        # Resume: skip sources already written
        if art_id in done_ids:
            skipped_due_to_resume += 1
            continue

        base = id2row_article.get(art_id)
        if base is None:
            missing_in_corpus += 1
            continue
        author = base["author"]

        # -------- Positive (same author, article→tweet) --------
        same_author_idxs = [
            i for i in author_to_tweet_idxs.get(author, [])
            if int(tweets.loc[i, "id"]) != art_id
        ]
        if same_author_idxs:
            pos_idx = random.choice(same_author_idxs)
            pos_row = tweets.loc[pos_idx]
            rec_pos = {
                "id0": art_id,
                "id1": int(pos_row["id"]),
                "text0": mod_text,
                "text1": pos_row["text"],
                "label": 1,
                "author0": author,
                "author1": author,
                "genre0": "modified_article",
                "genre1": "tweet",
                "pair_type": args.pair_tag
            }
            # swap handling
            swap = (random.random() < args.swap_prob)
            rec_pos["swap"] = bool(swap)
            if swap:
                rec_pos["left_text"], rec_pos["right_text"] = rec_pos["text1"], rec_pos["text0"]
            else:
                rec_pos["left_text"], rec_pos["right_text"] = rec_pos["text0"], rec_pos["text1"]

            for k in ["impersonator","support_ids","obfuscated_from"]:
                if k in meta: rec_pos[k] = meta[k]
            out.write(json.dumps(rec_pos, ensure_ascii=False) + "\n")
            pos_cnt += 1
            target_bin = pos_row["len_bin"]
        else:
            no_pos_tweet += 1
            continue

        # -------- Negatives (different author tweets) --------
        need = args.neg_ratio

        # Pool 1: same length bin, different author, id != art_id
        candidates = []
        for au, idxs in bin_to_idxs_by_author[target_bin].items():
            if au == author:
                continue
            for i in idxs:
                if int(tweets.loc[i, "id"]) != art_id:
                    candidates.append(i)
        random.shuffle(candidates)

        # Pool 2: any different author, id != art_id
        backup = [i for i in tweets.index
                  if tweets.loc[i, "author"] != author and int(tweets.loc[i, "id"]) != art_id]
        random.shuffle(backup)

        chosen = 0
        for pool in (candidates, backup):
            for i in pool:
                if chosen >= need: break
                if neg_use_count[i] >= args.neg_per_author_cap: continue
                r = tweets.loc[i]
                rec_neg = {
                    "id0": art_id,
                    "id1": int(r["id"]),
                    "text0": mod_text,
                    "text1": r["text"],
                    "label": 0,
                    "author0": author,
                    "author1": r["author"],
                    "genre0": "modified_article",
                    "genre1": "tweet",
                    "pair_type": args.pair_tag
                }
                # swap handling
                swap = (random.random() < args.swap_prob)
                rec_neg["swap"] = bool(swap)
                if swap:
                    rec_neg["left_text"], rec_neg["right_text"] = rec_neg["text1"], rec_neg["text0"]
                else:
                    rec_neg["left_text"], rec_neg["right_text"] = rec_neg["text0"], rec_neg["text1"]

                for k in ["impersonator","support_ids","obfuscated_from"]:
                    if k in meta: rec_neg[k] = meta[k]
                out.write(json.dumps(rec_neg, ensure_ascii=False) + "\n")
                neg_use_count[i] += 1
                chosen += 1
                neg_cnt += 1
            if chosen >= need: break

    out.close()
    print(f"[done] wrote -> {args.out_jsonl}")
    print(f"[counts] positives={pos_cnt}, negatives={neg_cnt}, "
          f"skipped_resume={skipped_due_to_resume}, missing_in_corpus={missing_in_corpus}, "
          f"no_pos_tweet={no_pos_tweet}")

    # --- Save stats file ---
    stats_path = args.out_jsonl.replace(".jsonl", "_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Pair generation statistics (AT, modified article vs original tweet)\n")
        f.write("=================================================================\n")
        f.write(f"docs_json: {args.docs_json}\n")
        f.write(f"modified_jsonl: {args.modified_jsonl}\n")
        f.write(f"out_jsonl: {args.out_jsonl}\n")
        f.write(f"pair_tag: {args.pair_tag}\n")
        f.write(f"neg_ratio: {args.neg_ratio}\n")
        f.write(f"neg_per_author_cap: {args.neg_per_author_cap}\n")
        f.write("\n")
        f.write(f"Total modified records read: {total_modified}\n")
        f.write(f"Already present (skipped by resume): {skipped_due_to_resume}\n")
        f.write(f"Missing in article corpus: {missing_in_corpus}\n")
        f.write(f"No same-author tweet available (skipped): {no_pos_tweet}\n")
        f.write(f"Positives written: {pos_cnt}\n")
        f.write(f"Negatives written: {neg_cnt}\n")
        f.write(f"Total pairs written: {pos_cnt + neg_cnt}\n")
    print(f"[stats] saved -> {stats_path}")

if __name__ == "__main__":
    main()