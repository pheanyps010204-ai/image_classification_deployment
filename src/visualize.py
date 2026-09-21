import matplotlib.pyplot as plt
import torch

from src.dataset import get_dataloaders


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def show_batch(num_images: int = 16) -> None:
    """Display a batch of CIFAR-10 images."""

    train_loader, _ = get_dataloaders()

    images, labels = next(iter(train_loader))

    images = images[:num_images]
    labels = labels[:num_images]

    # Undo normalization for visualization
    mean = torch.tensor(
        (0.4914, 0.4822, 0.4465)
    ).view(3, 1, 1)

    std = torch.tensor(
        (0.2470, 0.2435, 0.2616)
    ).view(3, 1, 1)

    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title(CLASS_NAMES[labels[i].item()])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_batch()