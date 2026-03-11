import json
import pandas as pd

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


train_data = load_jsonl("data/processed/train.jsonl")

df = pd.DataFrame(train_data)

print("="*60)
print("DATASET QUALITY REPORT")
print("="*60)

print("Total rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nInstruction length statistics:")
print(df["instruction"].astype(str).str.len().describe())

print("\nOutput length statistics:")
print(df["output"].astype(str).str.len().describe())

print("\nSample rows:\n")

for i in range(3):
    print("-"*50)
    print("Instruction:\n", df.iloc[i]["instruction"])
    print("\nOutput:\n", df.iloc[i]["output"][:500])
    print("-"*50)