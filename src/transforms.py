from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = CIFAR10_MEAN,
            std = CIFAR10_STD,
        ),
    ])

def get_test_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = CIFAR10_MEAN,
            std = CIFAR10_STD,
        ),
    ])