import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List
from transformers import PreTrainedTokenizer


class InstructionDataset(Dataset):

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_seq_len: int):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} samples from {data_path}.")


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self.data[idx]

        prompt = sample.get("prompt", "")
        response = sample.get("response", "")

        full_text = f"{prompt} {response}"

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        labels = input_ids.clone()

        prompt_end_index = self.tokenizer(prompt, return_tensors="pt", truncation=True)['input_ids'].shape[1]

        labels[:prompt_end_index] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }