import pandas as pd
import datasets
from datasets import load_dataset
import os
import numpy as np
import requests
import json


np.random.seed(42)

gsm8k = load_dataset("gsm8k", "main")

train_data = []
for item in gsm8k["train"]:
    train_data.append({
        "question": item["question"],
        "answer": item["answer"]
    })

test_data = []
for item in gsm8k["test"]:
    test_data.append({
        "question": item["question"],
        "answer": item["answer"]
    })

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

os.makedirs("data/gsm8k", exist_ok=True)
train_df.to_parquet("data/gsm8k/train.parquet")
test_df.to_parquet("data/gsm8k/test.parquet")


# URL of the dataset
url = "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json"

# Create directory structure
os.makedirs("data/metamathqa", exist_ok=True)

# Download file
print("Downloading MetaMathQA dataset...")
response = requests.get(url)
response.raise_for_status()

# Parse JSON data
print("Processing data...")
data = json.loads(response.content)

# Convert to DataFrame
df = pd.DataFrame(data)

# Randomly select 1000 examples for test set
test_indices = np.random.choice(len(df), size=1000, replace=False)
test_df = df.iloc[test_indices]
train_df = df.drop(test_indices)

# Save as parquet files
print(f"Saving {len(train_df)} examples to train.parquet...")
train_df.to_parquet("data/metamathqa/train.parquet")

print(f"Saving {len(test_df)} examples to test.parquet...")
test_df.to_parquet("data/metamathqa/test.parquet")

print("Complete! Files saved to:")
print("- data/metamathqa/train.parquet")
print("- data/metamathqa/test.parquet")

# SlimPajama-6B
os.makedirs("data/SlimPajama-6B", exist_ok=True)

ds = load_dataset('DKYoon/SlimPajama-6B')
def map_slimpajama(example, idx):
    example = {
        'query': '',
        'response': example['text'],
        'type': 'SlimPajama',
        '__index_level_0__': idx
    }
    return example

ds = ds.map(map_slimpajama, with_indices=True)
train_ds = ds['train']
validation_ds = ds['validation']
test_ds = ds['test']

# Save as parquet files
print(f"Saving {len(train_ds)} examples to train.parquet...")
train_ds.to_parquet("data/SlimPajama-6B/train.parquet")

print(f"Saving {len(validation_ds)} examples to validation.parquet...")
validation_ds.to_parquet("data/SlimPajama-6B/validation.parquet")

print(f"Saving {len(test_ds)} examples to test.parquet...")
test_ds.to_parquet("data/SlimPajama-6B/test.parquet")

# open-r1/OpenR1-Math-220k
os.makedirs("data/OpenR1-Math-220k", exist_ok=True)

ds = load_dataset('open-r1/OpenR1-Math-220k', 'all')['train']

def map_openr1math(example, idx):
    example = {
        'query': example['problem'],
        'response': example['solution'],
        'type': 'openr1-math',
        '__index_level_0__': idx
    }
    return example

ds = ds.map(map_openr1math, with_indices=True)

ds = ds.train_test_split(test_size=0.02, seed=42)
train_ds = ds['train']
test_ds = ds['test']

# Save as parquet files
print(f"Saving {len(train_ds)} examples to train.parquet...")
train_ds.to_parquet("data/OpenR1-Math-220k/train.parquet")

print(f"Saving {len(test_ds)} examples to test.parquet...")
test_ds.to_parquet("data/OpenR1-Math-220k/test.parquet")