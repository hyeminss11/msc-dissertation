# text_processing.py
import re
import random
from collections import defaultdict

# ---------- anonymization ----------
def anonymise_text_per_doc(text: str) -> str:
    """
    Anonymise @mentions and URLs with per-doc stable random tags.
    e.g. @jack -> @USER_4821 (same @jack in the same doc -> same number)
         https://... -> <URL_1003> (same URL in the same doc -> same number)
    """
    mention_map = {}
    url_map = {}

    def repl_mention(m):
        mention = m.group(0)
        if mention not in mention_map:
            mention_map[mention] = random.randint(1, 9999)
        return f"@USER_{mention_map[mention]}"

    def repl_url(m):
        url = m.group(0)
        if url not in url_map:
            url_map[url] = random.randint(1, 9999)
        return f"<URL_{url_map[url]}>"

    text = re.sub(r'@\w+', repl_mention, text)
    text = re.sub(r'https?://\S+|www\.\S+', repl_url, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- small text utils ----------
def merge_texts_naturally(texts):
    """Merge texts without visible join tokens (just normalize spaces)."""
    cleaned = []
    for t in texts:
        t = (t or "").strip()
        t = re.sub(r'\s+', ' ', t)
        if t:
            cleaned.append(t)
    return ' '.join(cleaned)

def split_by_words(text, max_words):
    """Split text by word count (no word cutting)."""
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ---------- main pipeline ----------
def process_documents_by_word_count(
    docs,
    min_word_threshold: int = 100,
    max_word_limit: int = 3000,
    slack_ratio: float = 0.20,   # allow up to +10% beyond the min threshold when merging
):
    """
    - Group by (author, genre).
    - For docs with >= min_word_threshold: keep as-is (but split if > max_word_limit).
    - For docs with <  min_word_threshold: greedily merge short docs from the same (author, genre)
      until reaching at least min_word_threshold, then *optionally* keep adding more short docs
      as long as total words <= min_word_threshold * (1 + slack_ratio).
      This avoids wasting tiny leftover snippets and prevents 1-word outputs.
    - Every output gets a new sequential id; original ids are recorded in `source_id`.
    - Mentions/URLs are anonymized per-doc before any length logic.
    """
    grouped = defaultdict(list)
    for d in docs:
        grouped[(d["author"], d["genre"])].append(d)

    processed_docs = []
    next_id = 0

    # helper to emit a merged chunk (split if needed)
    def emit_doc(author, genre, text, src_ids_joined):
        nonlocal next_id, processed_docs
        if not text:
            return
        words = text.split()
        if len(words) > max_word_limit:
            chunks = split_by_words(text, max_word_limit)
        else:
            chunks = [text]
        for ch in chunks:
            processed_docs.append({
                "id": str(next_id),
                "text": ch,
                "author": author,
                "genre": genre,
                "source_id": src_ids_joined if src_ids_joined else str(next_id),
            })
            next_id += 1

    max_merge_limit = int(min_word_threshold * (1.0 + slack_ratio))

    for (author, genre), doc_list in grouped.items():
        # Sort ascending by length so we consume small pieces first
        doc_list = sorted(
            doc_list,
            key=lambda d: len((d.get("text") or "").split())
        )

        i = 0
        buffer_texts = []
        buffer_src_ids = []

        while i < len(doc_list):
            raw = (doc_list[i].get("text") or "").strip()
            if not raw:
                i += 1
                continue

            # anonymize current doc before counting/merging
            cur_text = anonymise_text_per_doc(raw)
            cur_words = cur_text.split()
            cur_len = len(cur_words)

            if cur_len >= min_word_threshold:
                # Large enough by itself: flush any pending buffer first if valid, then emit this one
                if buffer_texts:
                    merged_len = sum(len(t.split()) for t in buffer_texts)
                    if merged_len >= min_word_threshold:
                        merged = merge_texts_naturally(buffer_texts)
                        emit_doc(author, genre, merged, "+".join(buffer_src_ids))
                    # reset buffer regardless (we don't carry undersized buffer forward)
                    buffer_texts, buffer_src_ids = [], []

                # Emit the standalone (split if needed)
                emit_doc(author, genre, cur_text, doc_list[i].get("id", "unknown"))
                i += 1
                continue

            # cur_len < min_word_threshold -> accumulate in buffer
            buffer_texts.append(cur_text)
            buffer_src_ids.append(doc_list[i].get("id", "unknown"))
            merged_len = sum(len(t.split()) for t in buffer_texts)

            if merged_len < min_word_threshold:
                # Not enough yet — move to next small doc
                i += 1
                continue

            # Reached the minimum: try to greedily add more *short* docs up to max_merge_limit
            j = i + 1
            while j < len(doc_list):
                nxt_raw = (doc_list[j].get("text") or "").strip()
                if not nxt_raw:
                    j += 1
                    continue
                nxt_text = anonymise_text_per_doc(nxt_raw)
                nxt_len = len(nxt_text.split())

                # Don't mix in a "long" doc; stop greedily adding here
                if nxt_len >= min_word_threshold:
                    break

                if merged_len + nxt_len <= max_merge_limit:
                    buffer_texts.append(nxt_text)
                    buffer_src_ids.append(doc_list[j].get("id", "unknown"))
                    merged_len += nxt_len
                    j += 1
                else:
                    break

            # Flush the merged short-doc bundle
            merged = merge_texts_naturally(buffer_texts)
            emit_doc(author, genre, merged, "+".join(buffer_src_ids))

            # Reset buffer and continue from j (we already consumed up to j-1)
            buffer_texts, buffer_src_ids = [], []
            i = j

        # End-of-group: if buffer still holds enough, flush; else drop it
        if buffer_texts:
            merged_len = sum(len(t.split()) for t in buffer_texts)
            if merged_len >= min_word_threshold:
                merged = merge_texts_naturally(buffer_texts)
                emit_doc(author, genre, merged, "+".join(buffer_src_ids))
            # else: discard undersized tail

    return processed_docs