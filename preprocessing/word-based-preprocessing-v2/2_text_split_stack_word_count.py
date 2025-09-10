import pandas as pd
from typing import List, Dict, Tuple
import unicodedata
import string

# --- Settings (Silver train/val length normalization) ---
INPUT_JSON  = "preprocessing/word-based-preprocessing-v2/crossnews_silver_clean_anon.json"
OUTPUT_JSON = "preprocessing/word-based-preprocessing-v2/crossnews_silver_wordcount.json"

MIN_WORDS = 300
MAX_WORDS = 420
TWEET_SEP = "\n<SEP>\n"  # NOTE: used only when joining; not counted as words

# --- PE (punctuation+emoji) helpers (recomputed after segmentation) ---
PUNCT = {
    ".", ",", "!", "?", ":", ";",
    "-", "–", "—", "…",
    "“", "”", "‘", "’",
    "(", ")", "[", "]"
}

def is_emoji(ch: str) -> bool:
    """Lightweight emoji detection via Unicode category."""
    return unicodedata.category(ch).startswith("So")

def punct_emoji_sequence(text: str):
    """Extract punctuation+emoji sequence from final (segmented) text."""
    if not isinstance(text, str):
        return []
    return [ch for ch in text if ch in PUNCT or is_emoji(ch)]

# --- IO ---
def load_json_to_df(path: str) -> pd.DataFrame:
    """Load JSON or JSON Lines into a DataFrame."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

# --- Token helpers ---
def words(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return str(text).split()

def join_words(wlist: List[str]) -> str:
    """Join tokens with a single space."""
    return " ".join(wlist)

# --- Build tweet segments (counts exclude the separator) ---
def make_segments_for_tweets(
    rows: List[Tuple[str, str, str]], next_id_start: int, group_author: str
) -> Tuple[List[Dict], int]:
    """
    rows: list of (id, author, text) for a single author/genre group
    We concatenate multiple tweets to reach MIN_WORDS..MAX_WORDS.
    The separator is inserted only when joining texts and is NOT counted as words.
    """
    segments: List[Dict] = []
    # cur_sources holds tuples of (source_id, token_list)
    cur_sources: List[Tuple[str, List[str]]] = []

    def cur_len() -> int:
        return sum(len(toks) for _, toks in cur_sources)

    def flush():
        """Emit a segment if it meets MIN_WORDS; reset buffer."""
        nonlocal next_id_start, cur_sources
        total_len = cur_len()
        if total_len >= MIN_WORDS:
            text = TWEET_SEP.join(" ".join(toks) for _, toks in cur_sources)
            segments.append({
                "id": next_id_start,
                "author": group_author,
                "genre": "Tweet",
                "text": text if total_len <= MAX_WORDS else " ".join(text.split()[:MAX_WORDS]),
                "word_count": min(total_len, MAX_WORDS),
                "source_ids": [sid for sid, _ in cur_sources],
            })
            next_id_start += 1
        cur_sources = []

    for _id, author, text in rows:
        tw = words(text)
        if not tw:
            continue

        # If a single tweet is too long, emit current buffer (if valid) then cut this tweet.
        if len(tw) > MAX_WORDS:
            flush()
            segments.append({
                "id": next_id_start,
                "author": group_author,
                "genre": "Tweet",
                "text": " ".join(tw[:MAX_WORDS]),
                "word_count": MAX_WORDS,
                "source_ids": [_id],
            })
            next_id_start += 1
            continue

        # If it fits in the current bucket, append; else flush and start new bucket.
        if cur_len() + len(tw) <= MAX_WORDS:
            cur_sources.append((_id, tw))
        else:
            flush()
            cur_sources.append((_id, tw))

        # Emit as soon as we reach MIN_WORDS
        if cur_len() >= MIN_WORDS:
            flush()

    # Flush remaining buffer
    flush()
    return segments, next_id_start

# --- Build article segments (safe, no token loss) ---
def make_segments_for_articles(
    rows: List[Tuple[str, str, str]], next_id_start: int, group_author: str
) -> Tuple[List[Dict], int]:
    """
    We accumulate tokens across the author's articles, emitting a segment
    whenever we reach MIN_WORDS (or MAX_WORDS). No 'break' that would drop leftovers.
    """
    segments: List[Dict] = []
    buf: List[str] = []
    src: List[str] = []

    def push():
        """Emit a segment if it meets MIN_WORDS; reset buffer."""
        nonlocal next_id_start, buf, src
        if len(buf) >= MIN_WORDS:
            wc = min(len(buf), MAX_WORDS)
            segments.append({
                "id": next_id_start,
                "author": group_author,
                "genre": "Article",
                "text": join_words(buf[:MAX_WORDS]),
                "word_count": wc,
                "source_ids": src.copy(),
            })
            next_id_start += 1
        buf, src = [], []

    for _id, author, text in rows:
        aw = words(text)
        ptr = 0
        while ptr < len(aw):
            space = MAX_WORDS - len(buf)
            if space == 0:
                push()
                continue
            take = min(space, len(aw) - ptr)
            if take > 0:
                buf.extend(aw[ptr:ptr+take])
                ptr += take
                if _id not in src:
                    src.append(_id)
            # Emit if reached MAX or MIN
            if len(buf) == MAX_WORDS or len(buf) >= MIN_WORDS:
                push()

    # Final leftover
    push()
    return segments, next_id_start

# --- Main processing ---
def build_all_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Create length-normalized segments per (author, genre)."""
    for c in ["id", "author", "genre", "text"]:
        if c not in df.columns:
            df[c] = None

    df["genre"] = df["genre"].astype(str)
    df["author"] = df["author"].astype(str)

    segments_all: List[Dict] = []
    next_id = 0

    for (author, genre), g in df.groupby(["author", "genre"], sort=False):
        rows = list(zip(
            g.get("id", pd.Series(range(len(g)))).astype(str),
            g["author"].astype(str),
            g["text"].astype(str)
        ))
        if genre.lower() == "tweet":
            segs, next_id = make_segments_for_tweets(rows, next_id, author)
        elif genre.lower() == "article":
            segs, next_id = make_segments_for_articles(rows, next_id, author)
        else:
            continue
        segments_all.extend(segs)

    out = pd.DataFrame(segments_all)

    # Recompute punctuation+emoji features on the FINAL (segmented) text
    if not out.empty:
        pe_seq = out["text"].apply(lambda s: " ".join(punct_emoji_sequence(s)))
        out["pe_seq"] = pe_seq
        out["pe_len"] = pe_seq.apply(lambda s: len(s.split()) if isinstance(s, str) else 0)

    return out

# --- Run ---
df = load_json_to_df(INPUT_JSON)

segments_df = build_all_segments(df)
segments_df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=4)

print(f"[segments] total segments: {len(segments_df)} "
      f"(Articles: {sum(segments_df['genre']=='Article')}, "
      f"Tweets: {sum(segments_df['genre']=='Tweet')})")