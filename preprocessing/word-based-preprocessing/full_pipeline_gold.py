from pathlib import Path
import json
import random
from text_processing import process_documents_by_word_count
from pair_generation_random import generate_pairs_like_original  # <- use this

# (Optional) make anonymization + pairing reproducible
random.seed(7)

# Step 1: Load raw data (Gold set)
with open("preprocessing/crossnews_gold.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

# Step 2: Preprocess documents
processed_docs = process_documents_by_word_count(
    raw_docs,
    min_word_threshold=100,
    max_word_limit=3000
)

# ---- Step 2.5: filter very short docs BEFORE pairing ----
def _wc(s): return len((s or "").split())
MIN_WORDS_FOR_PAIR = 20  # or 30, up to you
before = len(processed_docs)
processed_docs = [d for d in processed_docs if _wc(d.get("text","")) >= MIN_WORDS_FOR_PAIR]
print(f"[filter] kept {len(processed_docs)}/{before} docs (min_words={MIN_WORDS_FOR_PAIR})")
# ---------------------------------------------------------

# Optional: Save preprocessed version
Path("processed_data").mkdir(exist_ok=True)
with open("processed_data_new/crossnews_gold_processed.json", "w", encoding="utf-8") as f:
    json.dump(processed_docs, f, ensure_ascii=False, indent=2)

# Step 3: Generate gold AV pairs (no split), original-like balanced logic
df_at = generate_pairs_like_original(processed_docs, genre_1="Article", genre_2="Tweet",
                                     max_docs_per_author=50, seed=7)
df_aa = generate_pairs_like_original(processed_docs, genre_1="Article", genre_2="Article",
                                     max_docs_per_author=50, seed=7)
df_tt = generate_pairs_like_original(processed_docs, genre_1="Tweet", genre_2="Tweet",
                                     max_docs_per_author=50, seed=7)

# Step 4: Save to jsonl
df_at.to_json("processed_data_new/gold_Article_Tweet.jsonl", orient="records", lines=True, force_ascii=False)
df_aa.to_json("processed_data_new/gold_Article_Article.jsonl", orient="records", lines=True, force_ascii=False)
df_tt.to_json("processed_data_new/gold_Tweet_Tweet.jsonl", orient="records", lines=True, force_ascii=False)