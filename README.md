# CIFAR-10 Image Classification & Deployment

An end-to-end image classification project using **PyTorch** and a Convolutional Neural Network (CNN) to classify images into 10 CIFAR-10 classes.

The project covers the complete machine learning workflow:

**Data → Preprocessing → CNN → Training → Evaluation → Inference → FastAPI → Testing → Deployment**

## Project Overview

This project builds and deploys a CNN-based image classifier trained on the CIFAR-10 dataset.

The model classifies images into:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Model Performance

- **Test Accuracy:** 84.49%
- **Best Epoch:** 19
- **Framework:** PyTorch
- **Dataset:** CIFAR-10
- **Device:** CPU

## Project Structure

```text
image_classification_deployment/
│
├── app/
│   ├── main.py
│   ├── predict.py
│   ├── schemas.py
│   └── __init__.py
│
├── config/
│   └── config.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── checkpoints/
│
├── notebooks/
│   └── experiments.ipynb
│
├── outputs/
│   └── figures/
│
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   ├── transforms.py
│   ├── utils.py
│   └── visualize.py
│
├── tests/
│   ├── test_api.py
│   ├── test_dataset.py
│   └── test_model.py
│
├── .gitignore
├── README.md
└── requirements.txt