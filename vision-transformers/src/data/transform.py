from torchvision import transforms
from ..config.data_config import DataConfig

def get_train_transform(image_size: int = 32) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DataConfig.mean,
            std=DataConfig.std
        )
    ])

def get_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DataConfig.mean,
            std=DataConfig.std
        )
    ])