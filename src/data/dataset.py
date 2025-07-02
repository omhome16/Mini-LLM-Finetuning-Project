import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List
from transformers import PreTrainedTokenizer


class InstructionDataset(Dataset):
    """
    A PyTorch Dataset for instruction fine-tuning.
    """

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_seq_len: int):
        """
        Initializes the dataset by loading and formatting the data.

        Args:
            data_path (str): Path to the JSONL file containing the data.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            max_seq_len (int): The maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load data from the JSONL file
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} samples from {data_path}.")

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a single sample and returns input tensors.

        Args:
            idx (int): The index of the sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with 'input_ids', 'attention_mask', and 'labels'.
        """
        sample = self.data[idx]

        # Format the instruction data into a single string
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")

        # Instruction format: "[PROMPT] [RESPONSE]"
        full_text = f"{prompt} {response}"

        # Tokenize the full text
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Create labels for supervised fine-tuning.
        # We want the model to predict the response, so we mask the prompt tokens.
        labels = input_ids.clone()

        # Find where the prompt ends and the response begins
        prompt_end_index = self.tokenizer(prompt, return_tensors="pt", truncation=True)['input_ids'].shape[1]

        # Mask the prompt tokens by setting their labels to -100
        # -100 is a special value in PyTorch's CrossEntropyLoss that ignores the token.
        labels[:prompt_end_index] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }