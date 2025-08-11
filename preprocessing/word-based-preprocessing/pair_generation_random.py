import random
from collections import defaultdict
import pandas as pd

import random
from collections import defaultdict
import pandas as pd

def generate_av_pairs(
    docs,
    genre_1="Article",
    genre_2="Article",
    max_pairs_per_author=50,
    seed=7,
    shuffle=True
):
    random.seed(seed)
    author_to_docs = defaultdict(lambda: {genre_1: [], genre_2: []})
    
    for doc in docs:
        if doc['genre'] in [genre_1, genre_2]:
            author_to_docs[doc['author']][doc['genre']].append(doc)
    
    pos_pairs = []
    neg_pairs = []
    used_doc_ids = set()

    authors = list(author_to_docs.keys())
    
    # Positive pairs (same author)
    for author in authors:
        docs_1 = author_to_docs[author][genre_1]
        docs_2 = author_to_docs[author][genre_2]

        random.shuffle(docs_1)
        random.shuffle(docs_2)

        pair_count = min(len(docs_1), len(docs_2), max_pairs_per_author)
        i = 0
        while len(pos_pairs) < max_pairs_per_author and i < pair_count:
            d1 = docs_1[i]
            d2 = docs_2[i]
            if d1['id'] not in used_doc_ids and d2['id'] not in used_doc_ids:
                pos_pairs.append((
                    1, d1['text'], d2['text'],
                    d1['author'], d2['author'],
                    d1['id'], d2['id'],
                    d1['genre'], d2['genre']
                ))
                used_doc_ids.update([d1['id'], d2['id']])
            i += 1

    # Negative pairs (different authors)
    for author in authors:
        docs_1 = author_to_docs[author][genre_1]
        other_authors = [a for a in authors if a != author and author_to_docs[a][genre_2]]

        for doc1 in docs_1:
            if doc1['id'] in used_doc_ids:
                continue
            if not other_authors:
                continue
            random.shuffle(other_authors)
            for neg_author in other_authors:
                candidates = author_to_docs[neg_author][genre_2]
                random.shuffle(candidates)
                for doc2 in candidates:
                    if doc2['id'] not in used_doc_ids:
                        neg_pairs.append((
                            0, doc1['text'], doc2['text'],
                            doc1['author'], doc2['author'],
                            doc1['id'], doc2['id'],
                            doc1['genre'], doc2['genre']
                        ))
                        used_doc_ids.update([doc1['id'], doc2['id']])
                        break
                else:
                    continue
                break

    all_pairs = pos_pairs + neg_pairs
    if shuffle:
        random.shuffle(all_pairs)

    df = pd.DataFrame(all_pairs, columns=[
        'label', 'text0', 'text1',
        'author0', 'author1',
        'id0', 'id1',
        'genre0', 'genre1'
    ])
    
    return df