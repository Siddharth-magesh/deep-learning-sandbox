import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import TinyImageNetDataset
from data.transformation import train_transformation, test_transformation
from data.datamodule import get_dataloader
from typing import Tuple

def get_test_data():
    train_transform = train_transformation([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262], 64)
    test_transform = test_transformation([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262], 64)
    train_dataset, test_dataset = TinyImageNetDataset(transform=train_transform).get_dataset_splits()
    train_loader, test_loader = get_dataloader(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, _ = get_test_data()
    images, labels = next(iter(train_loader))
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")