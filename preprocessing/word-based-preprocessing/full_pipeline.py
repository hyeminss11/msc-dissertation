import json
import csv
import pandas as pd
from pathlib import Path
from text_processing import process_documents_by_word_count
from pair_generation_random import generate_av_pairs

# -----------------------------
# Step 1: Load raw data
# -----------------------------
with open("preprocessing/crossnews_silver.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

# -----------------------------
# Step 2: Preprocess documents
# -----------------------------
processed_docs = process_documents_by_word_count(
    raw_docs,
    min_word_threshold=100,
    max_word_limit=3000
)

# Optional: Save for inspection
Path("processed_data").mkdir(exist_ok=True)
with open("processed_data/crossnews_silver_processed.json", "w", encoding="utf-8") as f:
    json.dump(processed_docs, f, ensure_ascii=False, indent=2)

# -----------------------------
# Step 3: Generate pairs
# -----------------------------

# Article–Article (use dynamic max per author)
df_aa = generate_av_pairs(processed_docs, genre_1="Article", genre_2="Article", max_pairs_per_author=50)
df_aa.to_json("processed_data/silver_Article_Article.jsonl", orient="records", lines=True, force_ascii=False)

# Tweet–Tweet
df_tt = generate_av_pairs(processed_docs, genre_1="Tweet", genre_2="Tweet", max_pairs_per_author=50)
df_tt.to_json("processed_data/silver_Tweet_Tweet.jsonl", orient="records", lines=True, force_ascii=False)

# Article–Tweet
df_at = generate_av_pairs(processed_docs, genre_1="Article", genre_2="Tweet", max_pairs_per_author=50)
df_at.to_json("processed_data/silver_Article_Tweet.jsonl", orient="records", lines=True, force_ascii=False)