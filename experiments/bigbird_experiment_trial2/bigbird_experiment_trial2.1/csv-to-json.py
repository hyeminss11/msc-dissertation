import pandas as pd
import json

def convert_csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path)
    examples = []
    for _, row in df.iterrows():
        examples.append({
            "text1": row["text0"],
            "text2": row["text1"],
            "label": int(row["label"])
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

# Article-Article
convert_csv_to_json("preprocessing/original-reproduction/verification_data/train/CrossNews_Article_Article.csv", "experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/article-article/train.json")
convert_csv_to_json("preprocessing/original-reproduction/verification_data/test/CrossNews_Article_Article.csv", "experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/article-article/test.json")

# Article-Tweet
convert_csv_to_json("preprocessing/original-reproduction/verification_data/train/CrossNews_Article_Tweet.csv", "experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/article-tweet/train.json")
convert_csv_to_json("preprocessing/original-reproduction/verification_data/test/CrossNews_Article_Tweet.csv", "experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/article-tweet/test.json")

# Tweet-Tweet
convert_csv_to_json("preprocessing/original-reproduction/verification_data/train/CrossNews_Tweet_Tweet.csv", "experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/tweet-tweet/train.json")
convert_csv_to_json("preprocessing/original-reproduction/verification_data/test/CrossNews_Tweet_Tweet.csv", "experiments/bigbird_experiment_trial2/bigbird_experiment_trial2.1/data/tweet-tweet/test.json")