from fastapi import FastAPI, File, UploadFile

from app.predict import predict_uploaded_image
from app.schemas import PredictionResponse


app = FastAPI(
    title="CIFAR-10 Image Classification API",
    description="API for classifying CIFAR-10 images using a trained CNN.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "CIFAR-10 Image Classification API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    return await predict_uploaded_image(file)