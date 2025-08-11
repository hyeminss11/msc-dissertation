import pandas as pd

# load the file
train_df = pd.read_json("processed_data/silver_Tweet_Tweet_train.jsonl", lines=True)
val_df = pd.read_json("processed_data/silver_Tweet_Tweet_val.jsonl", lines=True)

# every id
train_ids = set(train_df['id0']) | set(train_df['id1'])
val_ids = set(val_df['id0']) | set(val_df['id1'])

# check overlap
overlap_ids = train_ids & val_ids

if overlap_ids:
    print(f"겹치는 문서 {len(overlap_ids)}개 발견")
    print(list(overlap_ids)[:20])  # 일부만 출력
else:
    print("Train과 Val에 겹치는 문서 없음")