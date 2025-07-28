import json
from collections import defaultdict

# Input
INPUT_PATH = "preprocessing/crossnews_gold.json"

# Output
ARTICLES_PATH = "experiments/llm_rewriting_trial1/data/gold_articles.json"
TWEETS_PATH = "experiments/llm_rewriting_trial1/data/gold_tweets.json"

# load
with open(INPUT_PATH, "r", encoding = "utf-8") as f:
    data = json.load(f)

# classification
articles = []
tweets = []

for item in data:
    genre = item.get("genre", "").lower()
    if genre == "article":
        articles.append({
            "author": item["author"].strip().lower(),
            "text": item["text"].strip()
        })
    elif genre == "tweet":
        tweets.append({
            "author": item["author"].strip().lower(),
            "text": item["text"].strip()
        })

# save
with open(ARTICLES_PATH, "w", encoding = "utf-8") as f:
    json.dump(articles, f, indent = 2, ensure_ascii = False)

with open(TWEETS_PATH, "w", encoding = "utf-8") as f:
    json.dump(tweets, f, indent = 2, ensure_ascii = False)

print(f"Extracted: {len(articles)} articles, {len(tweets)} tweets")