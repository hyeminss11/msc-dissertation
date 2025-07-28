import json
import random

# Load rewritten article/tweet data
with open("experiments/llm_rewriting_trial1/llm_rewriting_trial1.1/output/articles_rewritten_manual.json", encoding="utf-8") as f:
    articles = json.load(f)

with open("experiments/llm_rewriting_trial1/llm_rewriting_trial1.1/output/tweets_rewritten_manual.json", encoding="utf-8") as f:
    tweets = json.load(f)

samples = []

# Positive samples (same author)
for entry in random.sample(articles, 3):
    samples.append({
        "text1": entry["original"],
        "text2": entry["rewritten"],
        "same_author": True
    })

for entry in random.sample(tweets, 2):
    samples.append({
        "text1": entry["original"],
        "text2": entry["rewritten"],
        "same_author": True
    })

# Negative samples (different author pairs)
articles_pairs = random.sample(articles, 2)
samples.append({
    "text1": articles_pairs[0]["original"],
    "text2": articles_pairs[1]["rewritten"],
    "same_author": False
})

tweets_pairs = random.sample(tweets, 2)
samples.append({
    "text1": tweets_pairs[0]["original"],
    "text2": tweets_pairs[1]["rewritten"],
    "same_author": False
})

# Save
with open("experiments/llm_rewriting_trial1/llm_rewriting_trial1.1/output/av_eval_samples.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print("Saved av_eval_samples.json with", len(samples), "entries.")