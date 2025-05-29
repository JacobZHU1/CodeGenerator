import pandas as pd
import json

# 加载 parquet
df = pd.read_parquet("data/train.parquet")

source_col = 'Code'
test_col = 'Unit Test - (Ground Truth)'

with open("data/codetotest_train.jsonl", "w") as f:
    for idx, row in df.iterrows():
        src = row[source_col]
        test = row[test_col]
        json.dump({"source": src, "target": test}, f)
        f.write("\n")

print("✅ 已保存 codetotest_train.jsonl")
