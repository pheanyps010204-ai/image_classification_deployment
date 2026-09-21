from pydantic import BaseModel


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float