"""
dataset.py

PyTorch Dataset that reads preprocessed .jsonl files.
Preprocessing is done once offline (preprocess.py).
"""

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class HtmlDataset(Dataset):
    def __init__(self, jsonl_path: str, max_len: int = 512):
        self.max_len = max_len
        self.records = [
            json.loads(line)
            for line in Path(jsonl_path).read_text().splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "input_ids":      torch.tensor(r["input_ids"][:self.max_len],      dtype=torch.long),
            "token_labels":   torch.tensor(r["token_labels"][:self.max_len],   dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"][:self.max_len], dtype=torch.long),
            "doc_id":         r["doc_id"],
        }


def collate_fn(batch, pad_token_id: int = 0):
    """Pad variable-length sequences to the longest in the batch."""
    return {
        "input_ids":      pad_sequence([b["input_ids"]      for b in batch], batch_first=True, padding_value=pad_token_id),
        "token_labels":   pad_sequence([b["token_labels"]   for b in batch], batch_first=True, padding_value=-100),
        "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "doc_ids":        [b["doc_id"] for b in batch],
    }
