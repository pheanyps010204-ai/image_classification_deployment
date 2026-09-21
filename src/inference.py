import torch
from PIL import Image

from src.model import CIFAR10CNN
from src.transforms import get_test_transforms

from config.config import (
    DEVICE,
    CHECKPOINT_DIR,
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

    # Create the same architecture used during training
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
        print(f"Model trained until epoch: {checkpoint['epoch']}")

    if "test_accuracy" in checkpoint:
        print(
            f"Checkpoint test accuracy: "
            f"{checkpoint['test_accuracy']:.2f}%"
        )

    return model


def predict_image(model, image_path):
    """
    Predict the CIFAR-10 class of a single image.
    """

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Get the same preprocessing used for test images
    transform = get_test_transforms()

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move image to CPU/GPU
    image_tensor = image_tensor.to(DEVICE)

    # Disable gradient calculation
    with torch.no_grad():

        # Model prediction
        outputs = model(image_tensor)

        # Convert logits to probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Get highest probability
        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_class_name = CLASS_NAMES[
        predicted_class.item()
    ]

    confidence_percentage = (
        confidence.item() * 100
    )

    return (
        predicted_class_name,
        confidence_percentage,
        probabilities
    )


def main():

    print("=" * 60)
    print("CIFAR-10 IMAGE INFERENCE")
    print("=" * 60)

    print(f"\nUsing device: {DEVICE}")

    # Load trained model
    model = load_model()

    print("\nModel loaded successfully.")
    print("Inference pipeline is ready.")

    print("\nAvailable classes:")

    for class_name in CLASS_NAMES:
        print(f"  - {class_name}")


if __name__ == "__main__":
    main()