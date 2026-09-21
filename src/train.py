import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import CIFAR10CNN
from config.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_DIR,
)


def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one complete epoch.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Move data to CPU or GPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track loss
        running_loss += loss.item() * images.size(0)

        # Calculate predictions
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


def evaluate(model, test_loader, criterion):
    """
    Evaluate the model on the test dataset.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # No gradients are needed during evaluation
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Calculate predictions
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / total
    test_accuracy = 100.0 * correct / total

    return test_loss, test_accuracy


def main():

    print(f"Using device: {DEVICE}")

    # Load training and test data
    train_loader, test_loader = get_dataloaders()

    # Create CNN model
    model = CIFAR10CNN().to(DEVICE)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Track the best test accuracy
    best_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.2f}% "
            f"| Test Loss: {test_loss:.4f} "
            f"| Test Acc: {test_accuracy:.2f}%"
        )

        # Save the best model
        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_accuracy": test_accuracy,
                },
                checkpoint_path
            )

            print(
                f"  ✓ Best model saved "
                f"(Test Accuracy: {test_accuracy:.2f}%)"
            )

    print("\nTraining completed.")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Best model saved to: {CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()