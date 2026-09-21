import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path

from src.dataset import get_dataloaders
from src.model import CIFAR10CNN

from config.config import (
    DEVICE,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)


# CIFAR-10 class names
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


def load_model():
    """
    Load the trained CNN from the best checkpoint.
    """

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Create the same CNN architecture used during training
    model = CIFAR10CNN().to(DEVICE)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Evaluation mode
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")

    if "epoch" in checkpoint:
        print(f"Best model epoch: {checkpoint['epoch']}")

    if "test_accuracy" in checkpoint:
        print(
            f"Saved test accuracy: "
            f"{checkpoint['test_accuracy']:.2f}%"
        )

    return model


def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test dataset.
    """

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            running_loss += (
                loss.item() * images.size(0)
            )

            # Get predicted class
            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            # Save predictions and labels
            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    test_loss = running_loss / total
    test_accuracy = 100.0 * correct / total

    return (
        test_loss,
        test_accuracy,
        all_labels,
        all_predictions,
    )


def calculate_per_class_accuracy(
    labels,
    predictions
):
    """
    Calculate accuracy for every CIFAR-10 class.
    """

    class_correct = [0] * len(CLASS_NAMES)
    class_total = [0] * len(CLASS_NAMES)

    for label, prediction in zip(
        labels,
        predictions
    ):

        class_total[label] += 1

        if label == prediction:
            class_correct[label] += 1

    print("\nPer-Class Accuracy")
    print("-" * 45)

    for i, class_name in enumerate(CLASS_NAMES):

        if class_total[i] > 0:
            accuracy = (
                100.0
                * class_correct[i]
                / class_total[i]
            )
        else:
            accuracy = 0.0

        print(
            f"{class_name:<12} : "
            f"{accuracy:6.2f}% "
            f"({class_correct[i]}/{class_total[i]})"
        )


def create_confusion_matrix(
    labels,
    predictions
):
    """
    Create a confusion matrix.
    """

    num_classes = len(CLASS_NAMES)

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64
    )

    for label, prediction in zip(
        labels,
        predictions
    ):

        confusion_matrix[
            label,
            prediction
        ] += 1

    return confusion_matrix.numpy()


def plot_confusion_matrix(
    confusion_matrix,
    save_path
):
    """
    Plot and save the confusion matrix.
    """

    plt.figure(figsize=(10, 8))

    plt.imshow(
        confusion_matrix,
        interpolation="nearest"
    )

    plt.title("CIFAR-10 Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(CLASS_NAMES))

    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        CLASS_NAMES
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Display values inside the matrix
    threshold = confusion_matrix.max() / 2

    for i in range(
        confusion_matrix.shape[0]
    ):

        for j in range(
            confusion_matrix.shape[1]
        ):

            value = confusion_matrix[i, j]

            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                )
            )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to:\n"
        f"{save_path}"
    )


def main():
    """
    Main evaluation pipeline.
    """

    print("=" * 60)
    print("CIFAR-10 MODEL EVALUATION")
    print("=" * 60)

    print(f"\nUsing device: {DEVICE}")

    # Load test DataLoader
    _, test_loader = get_dataloaders()

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # Load trained model
    model = load_model()

    # Evaluate model
    (
        test_loss,
        test_accuracy,
        labels,
        predictions,
    ) = evaluate_model(
        model,
        test_loader
    )

    # Overall results
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"Test Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy:.2f}%"
    )

    # Per-class accuracy
    calculate_per_class_accuracy(
        labels,
        predictions
    )

    # Create confusion matrix
    confusion_matrix = create_confusion_matrix(
        labels,
        predictions
    )

    # Make sure output directory exists
    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    # Save confusion matrix
    confusion_matrix_path = (
        OUTPUT_DIR / "confusion_matrix.png"
    )

    plot_confusion_matrix(
        confusion_matrix,
        confusion_matrix_path
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()