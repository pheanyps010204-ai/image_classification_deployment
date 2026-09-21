import torch

from src.model import CIFAR10CNN


def test_model_output_shape():
    model = CIFAR10CNN()

    x = torch.randn(4, 3, 32, 32)

    output = model(x)

    assert output.shape == (4, 10)