# pair_minimal.py
import os
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# ---------------- Settings ----------------
RANDOM_SEED = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

INPUT_JSON   = "preprocessing/word-based-preprocessing-v2/crossnews_gold_clean_anon.json"
OUTPUT_PATH  = "preprocessing/word-based-preprocessing-v2/pairs_balanced_json/gold_TT.jsonl"

# MODE: choose exactly one of {"TT", "AA", "AT"}
MODE = "TT"

# DATASET_KIND: "silver" -> save train/val ; "gold" -> test only
DATASET_KIND = "gold"

# Optional target pair count (total rows across labels). None = use max feasible.
TARGET_PAIRS: Optional[int] = None  

# Output format
USE_JSONL = True

# ===== Added options =====
CAP_PER_AUTHOR: Optional[int] = 100  # e.g., 200 ; None disables capping

# ---------------- IO + Guards ----------------
def load_df(path: str) -> pd.DataFrame:
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        df = pd.read_json(path)
    req = ["id","author","genre","text"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["author"] = df["author"].astype(str)
    df["genre"] = df["genre"].astype(str)
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["id","text"]).reset_index(drop=True)
    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs detected.")
    df["id"] = df["id"].astype(int)
    return df

# --------- Helpers ---------
def _author_index(df: pd.DataFrame, genre_lower: Optional[str]=None) -> Dict[str, List[int]]:
    if genre_lower:
        mask = df["genre"].str.lower().eq(genre_lower)
        sub = df.loc[mask, ["author","id"]].to_numpy()
    else:
        sub = df[["author","id"]].to_numpy()
    out = {}
    for a, _id in sub:
        out.setdefault(str(a), []).append(int(_id))
    return out

# --------- Positives ---------
def build_pos_AA(df: pd.DataFrame) -> List[Tuple[int,int]]:
    idx = _author_index(df, "article")
    out = []
    for _, ids in idx.items():
        if len(ids) < 2: continue
        random.shuffle(ids)
        if len(ids) & 1: ids = ids[:-1]
        out.extend((ids[i], ids[i+1]) for i in range(0,len(ids),2))
    return out

def build_pos_TT(df: pd.DataFrame) -> List[Tuple[int,int]]:
    idx = _author_index(df, "tweet")
    out = []
    for _, ids in idx.items():
        if len(ids) < 2: continue
        random.shuffle(ids)
        if len(ids) & 1: ids = ids[:-1]
        out.extend((ids[i], ids[i+1]) for i in range(0,len(ids),2))
    return out

def build_pos_AT(df: pd.DataFrame) -> List[Tuple[int,int]]:
    a_idx = _author_index(df, "article")
    t_idx = _author_index(df, "tweet")
    common = a_idx.keys() & t_idx.keys()
    out = []
    for a in common:
        A, T = a_idx[a], t_idx[a]
        if not A or not T: continue
        random.shuffle(A); random.shuffle(T)
        m = min(len(A), len(T))
        out.extend(zip(A[:m], T[:m]))
    return out

# --------- Negatives (new one-use matching) ---------
def make_neg_oneuse(docs: List[int], id2author: Dict[int,str]) -> List[Tuple[int,int]]:
    """One-use matching: each doc appears once across all NEG pairs."""
    docs = docs.copy()
    random.shuffle(docs)
    out = []
    while len(docs) >= 2:
        a = docs.pop()
        for j, b in enumerate(docs):
            if id2author[a] != id2author[b]:
                out.append((a,b))
                docs.pop(j)
                break
    return out

def make_neg_TT(df, used_ids):
    id2author = df.set_index("id")["author"].to_dict()
    docs = [x for x in used_ids if df.loc[df["id"]==x,"genre"].str.lower().eq("tweet").any()]
    return make_neg_oneuse(docs, id2author)

def make_neg_AA(df, used_ids):
    id2author = df.set_index("id")["author"].to_dict()
    docs = [x for x in used_ids if df.loc[df["id"]==x,"genre"].str.lower().eq("article").any()]
    return make_neg_oneuse(docs, id2author)

def make_neg_AT(df, used_A, used_T):
    id2author = df.set_index("id")["author"].to_dict()
    A, T = used_A.copy(), used_T.copy()
    random.shuffle(A); random.shuffle(T)
    out = []
    while A and T:
        a = A.pop()
        for j, t in enumerate(T):
            if id2author[a] != id2author[t]:
                out.append((a,t))
                T.pop(j)
                break
    return out

# --------- Utilities ---------
def _dedup_pairs(pairs: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    seen = set(); out = []
    for a,b in pairs:
        key = (a,b) if a<=b else (b,a)
        if key in seen: continue
        seen.add(key)
        out.append((a,b))
    return out

def _cap_pairs_by_author(pairs, id2author, cap):
    if cap is None: return pairs
    used = defaultdict(int); out = []
    for a,b in pairs:
        if used[id2author[a]]>=cap or used[id2author[b]]>=cap: continue
        out.append((a,b))
        used[id2author[a]]+=1; used[id2author[b]]+=1
    return out

def sanity_checks(df_pairs):
    if df_pairs.empty: return
    cpos = Counter(df_pairs[df_pairs.label==1].id0.tolist() + df_pairs[df_pairs.label==1].id1.tolist())
    cneg = Counter(df_pairs[df_pairs.label==0].id0.tolist() + df_pairs[df_pairs.label==0].id1.tolist())
    assert all(v<=1 for v in cpos.values()), "POS: doc reused"
    assert all(v<=1 for v in cneg.values()), "NEG: doc reused"

# --------- Core sampler ---------
def sample_balanced(df, mode, target_pairs):
    id2author = df.set_index("id")["author"].to_dict()
    id2text   = df.set_index("id")["text"].to_dict()
    if mode=="AA":
        pos = build_pos_AA(df); random.shuffle(pos)
        pos=_dedup_pairs(pos); pos=_cap_pairs_by_author(pos,id2author,CAP_PER_AUTHOR)
        used=[x for a,b in pos for x in (a,b)]
        neg = make_neg_AA(df, used); neg=_dedup_pairs(neg)
    elif mode=="TT":
        pos = build_pos_TT(df); random.shuffle(pos)
        pos=_dedup_pairs(pos); pos=_cap_pairs_by_author(pos,id2author,CAP_PER_AUTHOR)
        used=[x for a,b in pos for x in (a,b)]
        neg = make_neg_TT(df, used); neg=_dedup_pairs(neg)
    elif mode=="AT":
        pos = build_pos_AT(df); random.shuffle(pos)
        pos=_dedup_pairs(pos); pos=_cap_pairs_by_author(pos,id2author,CAP_PER_AUTHOR)
        used_A=[a for a,_ in pos]; used_T=[t for _,t in pos]
        neg = make_neg_AT(df, used_A, used_T); neg=_dedup_pairs(neg)
    else:
        raise ValueError("MODE must be one of {AA,TT,AT}")

    k=min(len(pos), len(neg))
    if target_pairs: k=min(k, target_pairs//2)
    pos,neg=pos[:k],neg[:k]
    rows=[]
    for a,b in pos: rows.append({"label":1,"text0":id2text[a],"text1":id2text[b],"author0":id2author[a],"author1":id2author[b],"id0":a,"id1":b})
    for a,b in neg: rows.append({"label":0,"text0":id2text[a],"text1":id2text[b],"author0":id2author[a],"author1":id2author[b],"id0":a,"id1":b})
    df_pairs=pd.DataFrame(rows).sample(frac=1.0,random_state=RANDOM_SEED).reset_index(drop=True)
    sanity_checks(df_pairs)
    return df_pairs

# --------- Split & Save ---------
def author_disjoint_split(df, frac_val=0.2):
    authors=sorted(df["author"].unique()); rng=np.random.RandomState(RANDOM_SEED); rng.shuffle(authors)
    n_val=int(round(len(authors)*frac_val))
    val=set(authors[:n_val]); train=set(authors[n_val:])
    return df[df.author.isin(train)].copy(), df[df.author.isin(val)].copy()

def save_jsonl(df,path):
    if df.empty: open(path,"w").close(); return
    df.to_json(path,orient="records",force_ascii=False,lines=True)

# ---------------- Main ----------------
if __name__=="__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df=load_df(INPUT_JSON)
    base,_=os.path.splitext(OUTPUT_PATH)
    if DATASET_KIND=="silver":
        tr,val=author_disjoint_split(df,0.2)
        pairs_tr=sample_balanced(tr,MODE,TARGET_PAIRS)
        pairs_val=sample_balanced(val,MODE,TARGET_PAIRS)
        save_jsonl(pairs_tr,f"{base}_train.jsonl")
        save_jsonl(pairs_val,f"{base}_val.jsonl")
        print(f"[Author-disjoint] Saved train={len(pairs_tr)} | val={len(pairs_val)}")
    else:
        pairs=sample_balanced(df,MODE,TARGET_PAIRS)
        save_jsonl(pairs,f"{base}_test.jsonl")
        print(f"Saved test={len(pairs)}")