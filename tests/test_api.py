from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "CIFAR-10 Image Classification" in response.text


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    import torchvision
    import os

    dataset = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=False,
        download=False,
    )

    image, label = dataset[0]

    image_path = "test_api_image.png"
    image.save(image_path)

    try:
        with open(image_path, "rb") as image_file:
            response = client.post(
                "/predict",
                files={
                    "file": (
                        "test_api_image.png",
                        image_file,
                        "image/png",
                    )
                },
            )

        assert response.status_code == 200

        data = response.json()

        assert data["filename"] == "test_api_image.png"
        assert data["predicted_class"] in dataset.classes
        assert isinstance(data["confidence"], float)

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)