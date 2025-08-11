from collections import Counter, defaultdict
import json

# Load the data
with open("preprocessing/crossnews_silver.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

# 1. Total number of documents
print(f"Total documents: {len(raw_docs)}")

# 2. Number of unique authors
authors = [doc["author"] for doc in raw_docs]
print(f"Unique authors: {len(set(authors))}")

# 3. Number of documents per genre
genre_counts = Counter(doc["genre"] for doc in raw_docs)
print("Documents per genre:")
for genre, count in genre_counts.items():
    print(f"  {genre}: {count}")

# 4. Number of documents per author per genre
author_genre_counts = defaultdict(lambda: defaultdict(int))
for doc in raw_docs:
    author = doc["author"]
    genre = doc["genre"]
    author_genre_counts[author][genre] += 1

# Optional: check how many authors have at least 5 docs per genre
min_docs_per_genre = 2
eligible_authors = [
    author for author, genres in author_genre_counts.items()
    if all(genres.get(g, 0) >= min_docs_per_genre for g in ["Article", "Tweet"])
]
print(f"Authors with ≥{min_docs_per_genre} in both Article and Tweet: {len(eligible_authors)}")