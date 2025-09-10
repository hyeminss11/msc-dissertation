import pandas as pd

# --- Settings ---
INPUT_JSON = "preprocessing/word-based-preprocessing-v2/crossnews_silver_clean_no_author_dupes.json"

# --- Load JSON (supports JSON Lines and JSON array) ---
def load_json_to_df(path):
    """Load JSON file as DataFrame (supports JSON Lines and JSON Array formats)."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

# Load dataset
df = load_json_to_df(INPUT_JSON)

# --- Ensure 'text' column exists ---
if "text" not in df.columns:
    raise ValueError("'text' column is missing from the dataset.")

# --- Find duplicated texts ---
# keep=False marks all duplicates (not just the later ones)
duplicates = df[df.duplicated(subset=["text"], keep=False)]

# --- Sort for easier viewing ---
duplicates = duplicates.sort_values(by="text").reset_index(drop=True)

# --- Print result ---
if duplicates.empty:
    print("No duplicated texts found.")
else:
    print(f"Found {duplicates['text'].nunique()} unique duplicated texts "
          f"covering {len(duplicates)} total rows.\n")
    # Group duplicates by text for easier inspection
    for text_val, group in duplicates.groupby("text"):
        print("="*80)
        print(f"Text: {text_val[:80]}{'...' if len(text_val) > 80 else ''}")
        print(group[["author", "genre"]])