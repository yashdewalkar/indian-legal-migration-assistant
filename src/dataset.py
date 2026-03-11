from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

print("Downloading dataset...")

dataset = load_dataset("nandhakumarg/IPC_and_BNS_transformation")

df = dataset["train"].to_pandas()

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

df.to_csv("data/raw/ipc_bns_raw.csv", index=False)

print("Saved dataset to data/raw/ipc_bns_raw.csv")