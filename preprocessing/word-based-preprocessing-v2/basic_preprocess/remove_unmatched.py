import pandas as pd

# --- Settings ---
INPUT_JSON  = "preprocessing/word-based-preprocessing-v2/crossnews_gold_clean_v2.json"         # Original file path
OUTPUT_JSON = "preprocessing/word-based-preprocessing-v2/crossnews_gold_clean_v2_1.json" # Output file path after removal

# --- Load JSON (supports both JSON array and JSON Lines) ---
def load_json_to_df(path):
    try:
        df = pd.read_json(path, lines=True)  # Try JSON Lines
    except ValueError:
        df = pd.read_json(path)              # Fallback to JSON array
    return df

df = load_json_to_df(INPUT_JSON)

# --- Remove rows where author is 'lucywardsings' ---
mask_target = df['author'] == 'lucywardsings'
print(f"Before removal: {len(df)} rows")
print(f"Rows to remove: {mask_target.sum()} rows")

df_clean = df[~mask_target].copy()
print(f"After removal: {len(df_clean)} rows")

# --- Save result ---
df_clean.to_json(OUTPUT_JSON, orient='records', force_ascii=False, indent=4)
print(f"Saved cleaned JSON to {OUTPUT_JSON}")