from torch.utils.data import DataLoader, Dataset
from typing import Tuple

def get_dataloader(
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, test_loader