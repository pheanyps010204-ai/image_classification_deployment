from torchvision import datasets
from config.config import RAW_DATA_DIR


def download_cifar10() -> None:

    print("Downloading CIFAR-10 dataset...")

    datasets.CIFAR10(
        root=RAW_DATA_DIR,
        train=True,
        download=True,
    )

    datasets.CIFAR10(
        root=RAW_DATA_DIR,
        train=False,
        download=True,
    )

    print("CIFAR-10 dataset downloaded successfully.")

    
if __name__ == "__main__":
    download_cifar10()