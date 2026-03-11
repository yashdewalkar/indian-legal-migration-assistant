import pandas as pd
import json
import os
import ast
import random

os.makedirs("data/processed", exist_ok=True)

print("Loading raw dataset...")

df = pd.read_csv("data/raw/ipc_bns_raw.csv")

rows = []

for _, row in df.iterrows():
    instruction = str(row["prompts"]).strip()
    response_raw = str(row["response"]).strip()

    try:
        response_dict = ast.literal_eval(response_raw)
        output = json.dumps(response_dict, indent=2)
    except:
        output = response_raw

    rows.append({
        "instruction": instruction,
        "input": "",
        "output": output,
        "source": "ipc_bns"
    })

print("Total rows:", len(rows))

random.shuffle(rows)

train_end = int(len(rows) * 0.8)
val_end = int(len(rows) * 0.9)

train_rows = rows[:train_end]
val_rows = rows[train_end:val_end]
test_rows = rows[val_end:]


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")


save_jsonl(train_rows, "data/processed/train.jsonl")
save_jsonl(val_rows, "data/processed/val.jsonl")
save_jsonl(test_rows, "data/processed/test.jsonl")

print("Dataset prepared successfully")

print("Train:", len(train_rows))
print("Val:", len(val_rows))
print("Test:", len(test_rows))