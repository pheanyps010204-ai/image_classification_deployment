from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.predict import predict_uploaded_image
from app.schemas import PredictionResponse


app = FastAPI(
    title="CIFAR-10 Image Classification API",
    description="""
## 🧠 CIFAR-10 Image Classification API

A deep learning API that classifies images into one of the **10 CIFAR-10 classes**
using a trained **PyTorch Convolutional Neural Network (CNN)**.

### Supported Classes

- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

### Model Performance

- **Test Accuracy:** 84.49%
- **Best Epoch:** 19
- **Framework:** PyTorch
- **Dataset:** CIFAR-10

### How to use

1. Open the **POST /predict** endpoint.
2. Click **Try it out**.
3. Upload a JPEG or PNG image.
4. Click **Execute**.
5. The API returns the predicted class and confidence score.
""",
    version="1.0.0",
    contact={
        "name": "Pheany Phal",
    },
    license_info={
        "name": "MIT License",
    },
)


# Frontend HTML templates
templates = Jinja2Templates(directory="app/templates")


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["System"],
    summary="AI Vision Interface",
)
def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="Check the health status of the API.",
)
def health():
    return {"status": "healthy"}


# --------------------------------------------------
# Image Prediction
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify an Image",
    description="""
Upload a **JPEG or PNG image** and classify it using the trained CIFAR-10 CNN.

The API returns:

- **filename** — name of the uploaded image
- **predicted_class** — predicted CIFAR-10 category
- **confidence** — model confidence percentage
""",
)
async def predict(
    file: UploadFile = File(
        ...,
        description="Upload a JPEG or PNG image for classification.",
    ),
):
    return await predict_uploaded_image(file)