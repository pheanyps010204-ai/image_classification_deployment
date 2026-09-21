from torch.utils.data import DataLoader
from torchvision import datasets

from config.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    RAW_DATA_DIR,
)

from src.transforms import (
    get_train_transforms,
    get_test_transforms,
)

def get_dataloaders():
    train_dataset = datasets.CIFAR10(
        root = RAW_DATA_DIR,
        train = True,
        download = False,
        transform = get_train_transforms(),
    )

    test_dataset = datasets.CIFAR10(
        root = RAW_DATA_DIR,
        train = False,
        download = False,
        transform = get_test_transforms(),

    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = NUM_WORKERS,
        pin_memory = True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS,
        pin_memory = True,

    )
    return train_loader, test_loader