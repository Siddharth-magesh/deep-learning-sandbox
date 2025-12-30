from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple

class TinyImageNetWrapper(Dataset):
    def __init__(self, hf_dataset, transform: transforms.Compose = None) -> None:
        super(TinyImageNetWrapper, self).__init__()
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class TinyImageNetDataset:
    def __init__(self, transform: transforms.Compose = None) -> None:
        self.train_dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
        self.test_dataset = load_dataset("zh-plus/tiny-imagenet", split="valid")
        self.transform = transform

    def get_dataset_splits(self) -> Tuple[Dataset, Dataset]:
        train_wrapped = TinyImageNetWrapper(self.train_dataset, self.transform)
        test_wrapped = TinyImageNetWrapper(self.test_dataset, self.transform)
        return train_wrapped, test_wrapped