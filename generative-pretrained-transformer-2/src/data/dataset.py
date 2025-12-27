import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from typing import Dict, List


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
