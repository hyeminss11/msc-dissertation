import re
import random
import unicodedata
import string
from collections import Counter
import pandas as pd

# --- Settings (NO length filtering in this script) ---
INPUT_JSON  = "preprocessing/word-based-preprocessing-v2/basic_preprocess/crossnews_gold_clean_final.json"
OUTPUT_JSON = "preprocessing/word-based-preprocessing-v2/crossnews_gold_clean_anon.json"
RANDOM_SEED = 7  # set to None for non-deterministic randomness

# Toggle whether to also dump n-gram counts for punctuation+emoji (bigger JSON)
DUMP_PE_COUNTS = False
PE_NGRAM_MAX   = 3
PE_VSEP = "|"   # key joiner for n-gram strings

# --- Load JSON ---
def load_json_to_df(path):
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

df = load_json_to_df(INPUT_JSON)

# --- Ensure required columns ---
for c in ["text", "genre"]:
    if c not in df.columns:
        df[c] = None

# ===================== Punctuation & Emoji extraction (from RAW) =====================
# We extract pe features from a raw snapshot, so later text cleaning will not remove this signal.

PUNCT = {
    ".", ",", "!", "?", ":", ";",
    "-", "–", "—", "…",
    "“", "”", "‘", "’",
    "(", ")", "[", "]"
}

def is_emoji(ch: str) -> bool:
    """Lightweight emoji detection via Unicode category (many emojis are 'So')."""
    cat = unicodedata.category(ch)
    return cat.startswith("So")

def punct_emoji_sequence(text: str):
    """Return list of punctuation+emoji in original order."""
    if not isinstance(text, str) or not text:
        return []
    out = []
    for ch in text:
        if ch in PUNCT or is_emoji(ch):
            out.append(ch)
    return out

def ngrams(seq, n):
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i:i+n])

def extract_ngram_counts(text: str, n_max: int = 3) -> dict:
    """Return dict 'tok|tok|...' -> count for punctuation+emoji n-grams (n=1..n_max)."""
    seq = punct_emoji_sequence(text)
    c = Counter()
    for n in range(1, n_max+1):
        c.update(ngrams(seq, n))
    return {PE_VSEP.join(t): int(cnt) for t, cnt in c.items()}

# Keep a raw snapshot ONLY for pe-feature extraction
df["text_raw_for_pe"] = df["text"].astype(str)

# Build pe features from raw
pe_seqs = df["text_raw_for_pe"].apply(punct_emoji_sequence)
df["pe_seq"] = pe_seqs.apply(lambda xs: " ".join(xs))  # space-delimited sequence
df["pe_len"] = pe_seqs.apply(len)
if DUMP_PE_COUNTS:
    df["pe_counts"] = df["text_raw_for_pe"].apply(lambda s: extract_ngram_counts(s, n_max=PE_NGRAM_MAX))

# Drop the helper snapshot to avoid confusion
df = df.drop(columns=["text_raw_for_pe"])

# ===================== Text cleaning / anonymization (NO length filtering here) =====================

# --- Regex patterns ---
EMAIL_RE   = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
MENTION_RE = re.compile(r'@([A-Za-z0-9_]{1,15})')   # Twitter usernames (1–15 chars)
URL_RE     = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)

# --- RNG setup ---
rng = random.Random(RANDOM_SEED) if RANDOM_SEED is not None else random.SystemRandom()

# --- Cleaning helpers ---
def strip_emojis_and_ctrl(text: str) -> str:
    """Remove emoji and control characters based on Unicode category (for the main text)."""
    if not isinstance(text, str):
        return text
    return ''.join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" and not unicodedata.category(ch).startswith("So")
    )

def is_mostly_english(text: str, threshold: float = 0.8) -> bool:
    """Keep only texts with >= threshold proportion of ASCII letters."""
    if not isinstance(text, str) or not text:
        return False
    letters = sum(ch.isalpha() for ch in text)
    ascii_letters = sum(('a' <= ch.lower() <= 'z') for ch in text)
    return (ascii_letters / letters) >= threshold if letters > 0 else False

def anonymize_text_local_random(text: str) -> str:
    """Per-post anonymization with local random USER numbers (unique within the post)."""
    if not isinstance(text, str) or not text:
        return text
    # 1) Replace emails
    text = EMAIL_RE.sub("<EMAIL>", text)
    # 2) Replace URLs
    text = URL_RE.sub("<URL>", text)
    # 3) Replace mentions with randomized local IDs
    local_map = {}
    used_nums = set()
    def assign_number(u_lower: str) -> int:
        if u_lower in local_map:
            return local_map[u_lower]
        while True:
            n = rng.randint(1, 9999)
            if n not in used_nums:
                used_nums.add(n)
                local_map[u_lower] = n
                return n
    def repl(m):
        u = m.group(1).lower()
        return f"USER_{assign_number(u)}"
    return MENTION_RE.sub(repl, text)

# 1) Remove emoji and control characters (NOTE: pe_seq already extracted above)
df["text"] = df["text"].map(strip_emojis_and_ctrl)

# 2) Filter out non-English texts (>= 90% ASCII letters required)
df = df[df["text"].map(is_mostly_english)].reset_index(drop=True)

# 3) Apply anonymization only to Tweets containing @, URL, or Email
is_tweet = df["genre"].astype(str).str.lower().eq("tweet")
tweet_texts = df.loc[is_tweet, "text"].astype(str)
need = tweet_texts.str.contains(
    r'@|https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    regex=True, case=False, na=False
)
if need.any():
    df.loc[need.index[need], "text"] = tweet_texts[need].map(anonymize_text_local_random)

# --- Save (NO length filtering in this script) ---
df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=4)
print(f"Saved JSON with cleaned text and pe features (no length filtering) -> {OUTPUT_JSON}")