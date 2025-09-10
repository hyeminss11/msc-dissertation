# save as estimate_at_upper_bounds.py
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter, deque
import random

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def norm_g(g):
    g = str(g or "").lower()
    return "article" if g=="article" else ("tweet" if g=="tweet" else g)

def split_by_authors(docs, ratios=(0.8,0.2), seed=42):
    authors = sorted({d["author"] for d in docs})
    rnd = random.Random(seed)
    rnd.shuffle(authors)
    n = len(authors)
    n_tr = int(n * ratios[0])
    A_tr = set(authors[:n_tr])
    A_vl = set(authors[n_tr:])
    tr = [d for d in docs if d["author"] in A_tr]
    vl = [d for d in docs if d["author"] in A_vl]
    return tr, vl

def pos_upper_bound_AT(docs):
    by_author = defaultdict(lambda: {"article":0, "tweet":0})
    for d in docs:
        g = norm_g(d.get("genre"))
        if g in ("article","tweet"):
            by_author[d["author"]][g] += 1
    return sum(min(v["article"], v["tweet"]) for v in by_author.values())

def neg_upper_bound_AT_max_matching(docs, seed=0):
    """Exact maximum possible A-T negatives with author!=author via Hopcroft–Karp."""
    A = [d for d in docs if norm_g(d.get("genre")) == "article"]
    T = [d for d in docs if norm_g(d.get("genre")) == "tweet"]
    if not A or not T:
        return 0
    # balance lengths (upper bound lives in min(|A|,|T|))
    m = min(len(A), len(T))
    A = A[:m]; T = T[:m]

    rnd = random.Random(seed)
    # index mapping
    a_idx = {i: A[i] for i in range(len(A))}
    t_idx = {j: T[j] for j in range(len(T))}
    # adjacency: different authors only (shuffle neighbors for determinism + variety)
    adj = {i: [] for i in a_idx}
    for i, da in a_idx.items():
        cand = list(range(len(T))); rnd.shuffle(cand)
        for j in cand:
            if da["author"] != t_idx[j]["author"]:
                adj[i].append(j)

    # Hopcroft–Karp
    t_of = [-1]*len(A)  # match on T for each A
    a_of = [-1]*len(T)  # match on A for each T

    def bfs():
        dist = [-1]*len(A)
        q = deque()
        for i in range(len(A)):
            if t_of[i] == -1:
                dist[i] = 0
                q.append(i)
        reached_free = False
        while q:
            i = q.popleft()
            for j in adj[i]:
                ii = a_of[j]
                if ii != -1 and dist[ii] == -1:
                    dist[ii] = dist[i] + 1
                    q.append(ii)
                if a_of[j] == -1:
                    reached_free = True
        return dist, reached_free

    def dfs(i, dist, seen_t):
        for j in adj[i]:
            if seen_t[j]:
                continue
            seen_t[j] = True
            ii = a_of[j]
            if a_of[j] == -1 or (ii != -1 and dist[ii] == dist[i] + 1 and dfs(ii, dist, [False]*len(T))):
                t_of[i] = j
                a_of[j] = i
                return True
        return False

    while True:
        dist, progress = bfs()
        if not progress:
            break
        for i in range(len(A)):
            if t_of[i] == -1:
                dfs(i, dist, [False]*len(T))

    matching = sum(1 for j in t_of if j != -1)
    return matching

def summarize_AT(tag, docs, seed=0):
    # overall counts
    nA = sum(1 for d in docs if norm_g(d.get("genre"))=="article")
    nT = sum(1 for d in docs if norm_g(d.get("genre"))=="tweet")
    both_authors = sum(1 for a,gs in _authors_genres(docs).items() if {"article","tweet"} <= gs)
    pos_up = pos_upper_bound_AT(docs)
    neg_up = neg_upper_bound_AT_max_matching(docs, seed=seed)
    feasible = min(pos_up, neg_up)
    print(f"[{tag}] A={nA} T={nT} authors_with_both={both_authors}  "
          f"pos_upper={pos_up}  neg_upper={neg_up}  feasible_AT_pairs_max={feasible}")
    return feasible, pos_up, neg_up

def _authors_genres(docs):
    m = defaultdict(set)
    for d in docs:
        g = norm_g(d.get("genre"))
        if g in ("article","tweet"):
            m[d["author"]].add(g)
    return m

def main():
    ap = argparse.ArgumentParser(description="Estimate AT pair upper bounds (pos/neg) before pairing.")
    ap.add_argument("--silver", type=str, help="Path to silver JSONL (post-preprocessing).")
    ap.add_argument("--gold",   type=str, help="Path to gold JSONL (post-preprocessing).")
    ap.add_argument("--silver_split", type=str, default="0.8,0.2",
                    help="Author-disjoint split ratio for silver (train,val). Default 0.8,0.2")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--targets", type=str, default="40000,10000,10000",
                    help="Target AT pos-per-split for (train,val,test). Total pairs are 2x this (pos=neg). Default 40000,10000,10000")
    args = ap.parse_args()

    tgt_tr, tgt_vl, tgt_te = [int(x) for x in args.targets.split(",")]

    if args.silver:
        silver = read_jsonl(Path(args.silver))
        r_tr, r_vl = [float(x) for x in args.silver_split.split(",")]
        tr, vl = split_by_authors(silver, ratios=(r_tr, r_vl), seed=args.seed)
        f_tr, p_tr, n_tr = summarize_AT("silver/train", tr, seed=args.seed)
        f_vl, p_vl, n_vl = summarize_AT("silver/val",   vl, seed=args.seed)
        ok_tr = f_tr >= tgt_tr
        ok_vl = f_vl >= tgt_vl
        print(f"[check] train target_pos={tgt_tr} -> {'OK' if ok_tr else 'SHORT by ' + str(tgt_tr - f_tr)}")
        print(f"[check] val   target_pos={tgt_vl} -> {'OK' if ok_vl else 'SHORT by ' + str(tgt_vl - f_vl)}")
    if args.gold:
        gold = read_jsonl(Path(args.gold))
        f_te, p_te, n_te = summarize_AT("gold/test", gold, seed=args.seed)
        ok_te = f_te >= tgt_te
        print(f"[check] test  target_pos={tgt_te} -> {'OK' if ok_te else 'SHORT by ' + str(tgt_te - f_te)}")

if __name__ == "__main__":
    main()
