# pair_generation_random.py
import random
import re
from collections import defaultdict
import pandas as pd

# -------------------- helpers --------------------
def _normalize_text(t: str) -> str:
    """Light normalization for equality checks (lower + collapse spaces)."""
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _split_even_odd(lst):
    """Return two disjoint views of the list (even/odd indices)."""
    return lst[::2], lst[1::2]

def _texts_equal_norm(id_to_doc, s1, s2) -> bool:
    """Check normalized text equality between two doc IDs."""
    return _normalize_text(id_to_doc[s1]["text"]) == _normalize_text(id_to_doc[s2]["text"])

def build_author_index(docs, genre):
    """
    Build {author: [doc_id, ...]} index for a given genre.
    Deduplicates IDs while preserving order.
    """
    by_author = defaultdict(list)
    for d in docs:
        if d["genre"] == genre:
            by_author[d["author"]].append(str(d["id"]))
    for a in by_author:
        by_author[a] = list(dict.fromkeys(by_author[a]))  # remove dup ids
    return by_author

# -------------------- core pairing --------------------
def generate_pair_ids(
    first_docs,
    second_docs,
    max_docs_per_author=100,
    max_pairs=None,
    rng=None,
    id_to_doc=None,
):
    """
    Create balanced pairs (same-author positives, different-author negatives).

    Improvements over the original:
    - SAME-GENRE (AA or TT): avoid self-pairs by splitting one list into even/odd and cross-matching.
    - Skip pairs whose normalized texts are identical (prevents trivial duplicates).
    """
    rng = rng or random
    pairs = []
    used_first, used_second = [], []
    id_to_auth = {}

    def to_pair_id(s1, s2):
        same = 1 if id_to_auth[s1] == id_to_auth[s2] else 0
        return f"{same}_{s1}_{s2}"

    # Heuristic: if author sets are the same dicts, we consider it "same-genre"
    same_genre = first_docs is second_docs or set(first_docs) == set(second_docs)

    # ---- Positive pairs (same author) ----
    for auth in list(first_docs.keys()):
        f_list = first_docs.get(auth, [])
        s_list = second_docs.get(auth, [])

        if same_genre:
            # One list only; split into two disjoint lists so we never pick the same id on both sides
            L = f_list if f_list else s_list
            L1, L2 = _split_even_odd(L)
            pair_count = min(len(L1), len(L2), max_docs_per_author)
            for i in range(pair_count):
                f_id, s_id = L1[i], L2[i]
                if f_id == s_id:
                    continue
                if id_to_doc is not None and _texts_equal_norm(id_to_doc, f_id, s_id):
                    continue
                used_first.append(f_id)
                used_second.append(s_id)
                id_to_auth[f_id] = auth
                id_to_auth[s_id] = auth
                pairs.append(to_pair_id(f_id, s_id))
                if max_pairs and len(pairs) >= (max_pairs // 2):
                    break
        else:
            # Cross-genre: align by index
            pair_count = min(len(f_list), len(s_list), max_docs_per_author)
            for i in range(pair_count):
                f_id, s_id = f_list[i], s_list[i]
                if f_id == s_id:
                    continue
                if id_to_doc is not None and _texts_equal_norm(id_to_doc, f_id, s_id):
                    continue
                used_first.append(f_id)
                used_second.append(s_id)
                id_to_auth[f_id] = auth
                id_to_auth[s_id] = auth
                pairs.append(to_pair_id(f_id, s_id))
                if max_pairs and len(pairs) >= (max_pairs // 2):
                    break

        if max_pairs and len(pairs) >= (max_pairs // 2):
            break

    # ---- Negative pairs (different author & ideally different text) ----
    for _ in range(len(used_first)):
        if not used_second:
            break
        f_id = used_first.pop()
        tries = 0
        s_id = rng.choice(used_second)
        while (
            id_to_auth.get(s_id) == id_to_auth.get(f_id)
            or (id_to_doc is not None and _texts_equal_norm(id_to_doc, f_id, s_id))
        ):
            tries += 1
            if tries > 30:  # give up if we can't find a decent negative
                s_id = None
                break
            s_id = rng.choice(used_second)
        if s_id is None:
            continue
        used_second.remove(s_id)
        pairs.append(f"0_{f_id}_{s_id}")

    return pairs

def get_pair_entries_df(
    pair_ids,
    docs,
    genre0,
    genre1,
    shuffle=True,
    seed=7,
    drop_identical_text_pairs=True,
    rebalance_after_filter=True,
):
    """
    Materialize pair IDs into a DataFrame with texts/authors/ids/genres.
    - drop_identical_text_pairs: also filters after materialization, as a final guard.
    - rebalance_after_filter: keep 50/50 label distribution after dropping.
    """
    if shuffle:
        r = random.Random(seed)
        r.shuffle(pair_ids)

    id_to_doc = {str(d["id"]): d for d in docs}
    rows = []
    for pid in pair_ids:
        same, first_id, second_id = pid.split("_", 2)
        a = id_to_doc[first_id]
        b = id_to_doc[second_id]

        if drop_identical_text_pairs and _normalize_text(a["text"]) == _normalize_text(b["text"]):
            continue

        rows.append({
            "label": int(same),
            "text0": a["text"], "text1": b["text"],
            "author0": a["author"], "author1": b["author"],
            "id0": first_id, "id1": second_id,
            "genre0": genre0, "genre1": genre1,
        })

    df = pd.DataFrame(rows)

    if rebalance_after_filter and not df.empty:
        c0 = df[df.label == 0]
        c1 = df[df.label == 1]
        n = min(len(c0), len(c1))
        if n > 0:
            df = pd.concat(
                [c0.sample(n, random_state=seed), c1.sample(n, random_state=seed)],
                ignore_index=True
            ).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df

# -------------------- public APIs --------------------
def generate_pairs_like_original(
    processed_docs,
    genre_1="Article",
    genre_2="Tweet",
    max_docs_per_author=100,
    max_pairs=None,
    seed=7,
):
    """
    Generate balanced pairs following the original CrossNews approach,
    while preventing self-pairs and identical-text pairs.
    """
    rng = random.Random(seed)
    first_idx  = build_author_index(processed_docs, genre_1)
    second_idx = build_author_index(processed_docs, genre_2)

    # only authors that have docs in BOTH genres
    common = set(first_idx) & set(second_idx)
    first_idx  = {a: first_idx[a]  for a in common}
    second_idx = {a: second_idx[a] for a in common}

    id_to_doc = {str(d["id"]): d for d in processed_docs}

    pair_ids = generate_pair_ids(
        first_idx, second_idx,
        max_docs_per_author=max_docs_per_author,
        max_pairs=max_pairs,
        rng=rng,
        id_to_doc=id_to_doc,
    )
    return get_pair_entries_df(
        pair_ids, processed_docs, genre_1, genre_2,
        shuffle=True, seed=seed,
        drop_identical_text_pairs=True,
        rebalance_after_filter=True,
    )

def split_by_source_id_and_generate_pairs(
    docs,
    genre_1="Article",
    genre_2="Article",
    max_pairs_per_author=100,
    seed=7,
):
    """
    Split by top-level source_id (so any chunks from the same original piece
    stay together), then generate pairs per split using the same-safe logic.
    """
    # group docs by their top-level source
    source_to_docs = defaultdict(list)
    for d in docs:
        source = d.get("source_id", d["id"])
        source_to_docs[source].append(d)

    sources = list(source_to_docs.keys())
    r = random.Random(seed)
    r.shuffle(sources)

    cut = int(0.8 * len(sources))
    train_sources = set(sources[:cut])
    val_sources   = set(sources[cut:])

    split_docs = {"train": [], "val": []}
    for src, group in source_to_docs.items():
        split = "train" if src in train_sources else "val"
        split_docs[split].extend(group)

    out = {}
    for split in ["train", "val"]:
        out[split] = generate_pairs_like_original(
            split_docs[split],
            genre_1=genre_1,
            genre_2=genre_2,
            max_docs_per_author=max_pairs_per_author,
            max_pairs=None,
            seed=seed,
        )
    return out

# -------------------- quick sanity check (optional) --------------------
def sanity_check(df: pd.DataFrame, label=None):
    """Print quick diagnostics on a generated pair DataFrame."""
    if label:
        print(f"[sanity_check] {label}")
    n_self = (df["id0"] == df["id1"]).sum()
    n_same_text = (
        df["text0"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        ==
        df["text1"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    ).sum()
    print(f"  total: {len(df)}, self-pairs: {n_self}, identical-text pairs: {n_same_text}")