import pandas as pd

# --- Settings ---
INPUT_JSON  = "preprocessing/word-based-preprocessing-v2/basic_preprocess/crossnews_gold_clean_no_author_dupes.json"
OUTPUT_JSON = "preprocessing/word-based-preprocessing-v2/basic_preprocess/crossnews_gold_clean_final.json"

# --- Load JSON (supports both JSON array and JSON Lines) ---
def load_json_to_df(path):
    try:
        # Try reading as JSON Lines
        df = pd.read_json(path, lines=True)
    except ValueError:
        # If failed, read as JSON array
        df = pd.read_json(path)
    return df

df = load_json_to_df(INPUT_JSON)
print(f"Before filtering: {len(df)} rows")

# --- Set conditions for rows TO DELETE ---
# Condition 1: The genre is 'article'
mask_genre_is_article = df['genre'] == 'Article'

# Condition 2: The word count is less than 20
mask_word_count_is_short = df['text'].fillna('').str.split().str.len() < 20

# Create a final mask for rows that meet BOTH deletion conditions
mask_to_delete = mask_genre_is_article & mask_word_count_is_short

# --- Filter by EXCLUDING the rows marked for deletion ---
# The ~ symbol inverts the mask, so we KEEP everything that is NOT marked for deletion.
df_clean = df[~mask_to_delete].copy()

print(f"After filtering: {len(df_clean)} rows")
print(f"Rows removed: {len(df) - len(df_clean)}")


# --- Save cleaned JSON ---
# Using records format to save as a list of objects
df_clean.to_json(OUTPUT_JSON, orient='records', force_ascii=False, indent=4)

print(f"Cleaned JSON file has been saved to {OUTPUT_JSON}")