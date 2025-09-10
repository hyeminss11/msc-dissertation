import pandas as pd

# Load dataset
df = pd.read_json("preprocessing/word-based-preprocessing-v2/basic_preprocess/crossnews_gold_clean_no_author_dupes.json")

# Ensure 'text' column is string
df["text"] = df["text"].astype(str)

# Filter: only 'Article' genre and exactly 1 word
single_word_df = df[
    (df["genre"].astype(str).str.lower() == "article") &
    (df["text"].str.split().str.len() == 3)
]

print(f"Found {len(single_word_df)} rows with exactly 1 word in 'Article' genre.")
print(single_word_df.head())