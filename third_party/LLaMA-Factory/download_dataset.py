from datasets import load_dataset
from tqdm import tqdm
import json

ds = load_dataset('meta-math/MetaMathQA')['train'] # .select(range(1000))

new_ds = []
for item in tqdm(ds):
    new_ds.append({
        'messages': [{'role': 'user', 'content': item['query']}, {'role': 'assistant', 'content': item['response']}]
    })

with open('data/metamathqa/train.json', 'w', encoding='utf-8') as f:
    json.dump(new_ds, f, indent=4, ensure_ascii=False)