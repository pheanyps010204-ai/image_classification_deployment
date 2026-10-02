# CIFAR-10 Image Classification & Deployment

An end-to-end image classification project using **PyTorch**, a Convolutional Neural Network (CNN), and **FastAPI** to classify images into 10 CIFAR-10 classes.

The project covers the complete machine learning workflow:

**Data → Preprocessing → CNN → Training → Evaluation → Inference → FastAPI → Testing → Deployment**

---

## Project Overview

This project builds and deploys a CNN-based image classifier trained on the **CIFAR-10 dataset**.

The model classifies images into 10 classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

---

## Model Performance

| Metric              | Result     |
| ------------------- | ---------- |
| **Test Accuracy**   | **85.05%** |
| **Best Epoch**      | **20**     |
| **Framework**       | PyTorch    |
| **Dataset**         | CIFAR-10   |
| **Training Device** | CPU        |

The model was evaluated on the complete CIFAR-10 test set containing **10,000 images**.

---

## Project Structure

```text
image_classification_deployment/
│
├── app/
│   ├── main.py
│   ├── predict.py
│   └── templates/
│       └── index.html
│
├── config/
│   └── config.py
│
├── data/
│   └── raw/
│
├── models/
│   └── checkpoints/
│       └── best_model.pth
│
├── notebooks/
│   └── colab_gpu_training.ipynb
│
├── sample_images/
│   ├── airplane_01.png
│   ├── airplane_02.png
│   ├── automobile_01.png
│   ├── automobile_02.png
│   └── ...
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
├── Dockerfile
├── README.md
├── requirements.txt
└── run.py
```

---

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd image_classification_deployment
```

Create and activate a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## Run the Web Application

Start the FastAPI application:

```powershell
uvicorn app.main:app --reload
```

Then open the following address in your browser:

```text
http://127.0.0.1:8000
```

The application provides a simple web interface where users can upload an image and receive a prediction from the trained CIFAR-10 model.

---

## Test with Sample Images

The repository includes sample CIFAR-10 images in:

```text
sampl
```
## 🚀 Live Demo

Try the deployed application:

https://image-classification-deployment.onrender.com/

