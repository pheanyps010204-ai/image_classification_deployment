from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image

from src.inference import load_model, predict_image


# Load the trained model once when the API starts.
model = load_model()


async def predict_uploaded_image(file: UploadFile):
    """
    Process an uploaded image and run model inference.
    """

    # Check that a file was uploaded
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded."
        )

    # Check file type
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG or PNG image."
        )

    try:
        # Read uploaded image
        image_data = await file.read()

        # Check that the file is not empty
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Verify that the uploaded file is a valid image
        image = Image.open(BytesIO(image_data))
        image.verify()

        # Re-open the image because verify() closes the image
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Save a temporary image
        temp_path = "temp_image.png"
        image.save(temp_path)

        # Run prediction
        predicted_class, confidence, probabilities = predict_image(
            model,
            temp_path
        )

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"Prediction error: {e}")

        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded image: {str(e)}"
        )

    finally:
        # Remove temporary image
        import os

        if os.path.exists("temp_image.png"):
            os.remove("temp_image.png")