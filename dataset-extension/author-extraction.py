import json
from collections import defaultdict, Counter

file_path = "preprocessing/crossnews_gold.json"

# Load data
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# author list
authors = [item["author"] for item in data]

# unique author list
unique_authors = sorted(set(authors))
print(f"Total author: {len(authors)}")
print(f"Unique author: {len(unique_authors)}")

# author sample count
author_counts = Counter(authors)

# Count per author and genre
author_genre_counts = defaultdict(lambda: {"Article": 0, "Tweet": 0})

for item in data:
    author = item["author"]
    genre = item["genre"]  # "Article" or "Tweet"
    author_genre_counts[author][genre] += 1

# Total samples per author (article + tweet)
author_total_counts = {
    author: counts["Article"] + counts["Tweet"]
    for author, counts in author_genre_counts.items()
}

# sort
sorted_authors = sorted(author_total_counts.items(), key=lambda x: x[1], reverse=True)

# print
print("Authors with 850+ samples (split by genre):")
for author, total_count in sorted_authors:
    if total_count >= 850:
        article_count = author_genre_counts[author]["Article"]
        tweet_count = author_genre_counts[author]["Tweet"]
        print(f"{author:50} : total={total_count}, article={article_count}, tweet={tweet_count}")
    else:
        break

# save csv
with open("dataset-extension/top_authors_850plus_split.csv", "w", encoding="utf-8") as f:
    f.write("author,total,article,tweet\n")
    for author, total_count in sorted_authors:
        if total_count >= 850:
            article = author_genre_counts[author]["Article"]
            tweet = author_genre_counts[author]["Tweet"]
            f.write(f"{author},{total_count},{article},{tweet}\n")
        else:
            break

# Save sorted unique authors by total sample count (article + tweet)
with open("dataset-extension/unique_authors_sorted_by_total.txt", "w", encoding="utf-8") as f:
    for author, total in sorted_authors:
        f.write(f"{author}\n")