import json
import os

# path
article_path = "experiments/llm_rewriting_trial1/output/articles_rewritten_manual.json"
tweet_path = "experiments/llm_rewriting_trial1/output/tweets_rewritten_manual.json"
output_path = "experiments/llm_rewriting_trial1/llm_rewriting_trial1.2/output/av_eval_samples_cross_author_type.json"

# load data
with open(article_path, encoding="utf-8") as f:
    articles = json.load(f)

with open(tweet_path, encoding="utf-8") as f:
    tweets = json.load(f)

# organise by author
articles_by_author = {a["author"]: a for a in articles}
tweets_by_author = {t["author"]: t for t in tweets}

# find shared authors
shared_authors = set(articles_by_author.keys()) & set(tweets_by_author.keys())

print(f"Found {len(shared_authors)} shared authors.")

samples = []

# generate combinations
for author in shared_authors:
    article = articles_by_author[author]
    tweet = tweets_by_author[author]

    # 1. article original – tweet original
    samples.append({
        "author": author,
        "text1": article["original"],
        "text2": tweet["original"],
        "same_author": True,
        "type": "article_original-tweet_original"
    })

    # 2. article rewritten – tweet original
    samples.append({
        "author": author,
        "text1": article["rewritten"],
        "text2": tweet["original"],
        "same_author": True,
        "type": "article_rewritten-tweet_original"
    })

    # 3. article original – tweet rewritten
    samples.append({
        "author": author,
        "text1": article["original"],
        "text2": tweet["rewritten"],
        "same_author": True,
        "type": "article_original-tweet_rewritten"
    })

    # 4. article rewritten – tweet rewritten
    samples.append({
        "author": author,
        "text1": article["rewritten"],
        "text2": tweet["rewritten"],
        "same_author": True,
        "type": "article_rewritten-tweet_rewritten"
    })

# save
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"Saved {len(samples)} cross-type AV samples to {output_path}")