# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer

from datasets import load_dataset, Dataset as HFDataset
from transformers import PreTrainedTokenizer, AutoTokenizer
import os


class BaseDataset(Dataset):
    """
    This is an in-memory BaseDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 text_keys,
                 max_length=1024,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        
        self.text_keys = text_keys if isinstance(text_keys, (tuple, list)) else [text_keys]

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):
        """Load Parquet files using datasets library and process efficiently"""
        print("Loading Parquet files...")
        
        # Load Parquet files with HuggingFace datasets
        datasets_list = []
        for file in self.parquet_files:
            # Check if file exists
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")
                
            # Load Parquet file with datasets library
            try:
                ds = load_dataset('parquet', data_files=file, split='train')
                datasets_list.append(ds)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                raise
        
        # Combine multiple datasets if needed
        if len(datasets_list) > 1:
            dataset = HFDataset.concatenate(datasets_list)
        else:
            dataset = datasets_list[0]
        
        print(f"Loaded {len(dataset)} records")
        
        # Process text fields efficiently using datasets' map function
        def combine_text_fields(example):
            combined_text = ""
            
            # Process text keys
            for key in self.text_keys[0]:
                if key in example:
                    value = example[key]
                    # Handle potential nested structures
                    while isinstance(value, (list, tuple)) and len(value) == 1:
                        value = value[0]
                    
                    # Add to combined text
                    combined_text += f"{key} : {value} "
            
            return {"combined_text": combined_text.strip()}
        
        print("Processing text fields...")
        processed_dataset = dataset.map(
            combine_text_fields,
            num_proc=os.cpu_count(),  # Parallel processing
            desc="Combining text fields"
        )
        
        # Extract processed texts to list
        # print(processed_dataset)
        self.texts = processed_dataset["combined_text"]
        print(f"Processed {len(self.texts)} texts")
        # print(self.texts[0])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        text = self.texts[item]

        ids_output = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        input_ids, attention_mask = ids_output['input_ids'][0], ids_output['attention_mask'][0]

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }

# unittest
if __name__ == '__main__':
    dataset = BaseDataset(
        parquet_files='data/metamathqa/train.parquet',
        tokenizer='gpt2',
        text_keys=[['query','response']],
        max_length=1024,
        truncation='right',
    )
    dataset[0]
        