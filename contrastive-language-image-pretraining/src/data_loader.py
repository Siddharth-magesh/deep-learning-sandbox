
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import kagglehub
from typing import Tuple

def simple_tokenizer(text: str, max_length: int = 77) -> torch.Tensor:
    tokens = text.lower().split()
    idxs = [min(hash(w) % 49408, 49407) for w in tokens][:max_length]
    arr = np.zeros(max_length, dtype=np.int64)
    arr[:len(idxs)] = idxs
    return torch.tensor(arr, dtype=torch.long)

def load_kaggle_flickr30k() ->  Tuple[str, str]:
    path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
    image_dir = os.path.join(path, "flickr30k_images", "flickr30k_images")
    caption_file = os.path.join(path, "flickr30k_images", "results.csv")
    return image_dir, caption_file

class Flickr30kDataset(Dataset):
    def __init__(self,  transform=None, max_length=77, max_samples=None) -> None:
        self.image_dir , caption_file_path = load_kaggle_flickr30k() 
        self.transform = transform
        self.max_length = max_length

        df = pd.read_csv(caption_file_path, delimiter='|', engine='python')
        df.columns = df.columns.str.strip()
        
        self.paired = []
        for _, row in df.iterrows():
            img_name = row['image_name'].strip()
            caption = row['comment'].strip()
            img_path = os.path.join(self.image_dir, img_name)
            
            if os.path.exists(img_path):
                self.paired.append((img_path, caption))
        
        if max_samples is not None and len(self.paired) > max_samples:
            self.paired = self.paired[:max_samples]
        
        print(f"Loaded {len(self.paired)} image-caption pairs from {len(df['image_name'].unique())} unique images")

    def __len__(self) -> int:
        return len(self.paired)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, caption = self.paired[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = simple_tokenizer(caption, self.max_length)
        return image, text

def get_data_loader(batch_size: int = 32, num_workers: int = 2, max_samples: int = None) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = Flickr30kDataset(transform=transform, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader
