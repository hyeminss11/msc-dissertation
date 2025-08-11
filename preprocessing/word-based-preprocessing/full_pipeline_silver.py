# build_silver_pairs.py
import json
from pathlib import Path

from text_processing import process_documents_by_word_count
from pair_generation_random import split_by_source_id_and_generate_pairs

# -------- Settings --------
INPUT_JSON = "preprocessing/crossnews_silver.json"  # raw silver set JSON
OUT_DIR = Path("processed_data_new")                    # output directory
MIN_WORDS = 100                                     # min words per doc after merging
MAX_WORDS = 3000                                    # max words before splitting
SEED = 7                                            # random seed
MAX_PER_AUTHOR = 100                                # max docs per author
MIN_WORDS_FOR_PAIR = 20                             # additional short-doc filter

def _wc(s: str) -> int:
    return len((s or "").split())

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 1) Load raw data
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    # 2) Preprocess once for all genres (merge/split + anonymize)
    processed_docs = process_documents_by_word_count(
        raw_docs,
        min_word_threshold=MIN_WORDS,
        max_word_limit=MAX_WORDS,
    )

    # 2.5) Filter out very short docs before pairing
    before_count = len(processed_docs)
    processed_docs = [
        d for d in processed_docs if _wc(d.get("text", "")) >= MIN_WORDS_FOR_PAIR
    ]
    after_count = len(processed_docs)
    print(f"[filter] kept {after_count}/{before_count} docs (min_words={MIN_WORDS_FOR_PAIR})")

    # Save processed docs (after filtering)
    with open(OUT_DIR / "crossnews_silver_processed.json", "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)

    # 3) Train/val split by source_id and generate pairs
    #    Generate AA, TT, AT
    for g1, g2, tag in [
        ("Article", "Article", "Article_Article"),
        ("Tweet", "Tweet", "Tweet_Tweet"),
        ("Article", "Tweet", "Article_Tweet"),
    ]:
        datasets = split_by_source_id_and_generate_pairs(
            processed_docs,
            genre_1=g1,
            genre_2=g2,
            max_pairs_per_author=MAX_PER_AUTHOR,
            seed=SEED,
        )

        # 4) Save train/val splits separately
        for split, df in datasets.items():
            out_path = OUT_DIR / f"silver_{tag}_{split}.jsonl"
            df.to_json(out_path, orient="records", lines=True, force_ascii=False)
            print(f"[saved] {out_path}  ->  {len(df)} rows")

if __name__ == "__main__":
    main()