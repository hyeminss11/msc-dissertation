import json

with open("experiments/llm_rewriting_trial1/data/sample_article_tweet_pairs.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)

article_samples = []
tweet_samples = []

for item in pairs:
    article_samples.append({
        "author": item["author"],
        "text": item["article"]
    })
    tweet_samples.append({
        "author": item["author"],
        "text": item["tweet"]
    })

with open("experiments/llm_rewriting_trial1/data/article_samples.json", "w", encoding="utf-8") as f:
    json.dump(article_samples, f, indent=2, ensure_ascii=False)

with open("experiments/llm_rewriting_trial1/data/tweet_samples.json", "w", encoding="utf-8") as f:
    json.dump(tweet_samples, f, indent=2, ensure_ascii=False)

print("Article and tweet samples saved separately.")