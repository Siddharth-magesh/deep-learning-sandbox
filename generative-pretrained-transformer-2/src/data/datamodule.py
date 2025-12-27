from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from .dataset import TextDataset
from ..config import DataConfig, TrainingConfig


def load_text_data(data_config: DataConfig):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(
        data_config.dataset_name,
        data_config.dataset_config,
        cache_dir=data_config.cache_dir
    )
    
    train_texts = [item['text'] for item in dataset[data_config.train_split] if item['text'].strip()]
    val_texts = [item['text'] for item in dataset[data_config.validation_split] if item['text'].strip()]
    test_texts = [item['text'] for item in dataset[data_config.test_split] if item['text'].strip()]
    
    train_dataset = TextDataset(train_texts, tokenizer, data_config.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, data_config.max_length)
    test_dataset = TextDataset(test_texts, tokenizer, data_config.max_length)
    
    return train_dataset, val_dataset, test_dataset, tokenizer


def get_dataloaders(train_dataset, val_dataset, test_dataset, training_config: TrainingConfig):
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    return train_loader, val_loader, test_loader
