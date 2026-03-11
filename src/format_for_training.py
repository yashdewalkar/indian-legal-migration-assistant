import json
import os

os.makedirs("data/final_sft", exist_ok=True)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def format_record(record):
    instruction = record["instruction"].strip()
    output = record["output"].strip()

    text = f"""### Instruction:
{instruction}

### Response:
{output}
"""

    return {"text": text}


splits = ["train", "val", "test"]

for split in splits:
    rows = load_jsonl(f"data/processed/{split}.jsonl")

    formatted = [format_record(r) for r in rows]

    save_jsonl(formatted, f"data/final_sft/{split}.jsonl")

    print(f"{split} dataset formatted:", len(formatted))