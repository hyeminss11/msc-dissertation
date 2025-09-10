import pandas as pd

# --- Settings ---
INPUT_JSON  = "preprocessing/word-based-preprocessing-v2/crossnews_gold_clean.json"
OUTPUT_JSON = "preprocessing/word-based-preprocessing-v2/crossnews_gold_clean_no_author_dupes.json"

# --- Load JSON (supports JSON Lines and JSON array) ---
def load_json_to_df(path):
    """Load JSON file as DataFrame (supports JSON Lines and JSON Array formats)."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

# Load dataset
df = load_json_to_df(INPUT_JSON)

# --- Ensure required columns ---
for col in ["author", "text"]:
    if col not in df.columns:
        raise ValueError(f"'{col}' column is missing from the dataset.")

# --- Remove duplicates only if both author and text are identical ---
df_no_dupes = df.drop_duplicates(subset=["author", "text"], keep="first").reset_index(drop=True)

# --- Save cleaned dataset ---
df_no_dupes.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=4)

print(f"Original dataset size: {len(df)}")
print(f"Duplicates removed (same author & text): {len(df) - len(df_no_dupes)}")
print(f"Cleaned dataset size: {len(df_no_dupes)}")
print(f"Saved cleaned dataset without same-author duplicates to {OUTPUT_JSON}")