from pathlib import Path

from fastapi import HTTPException, UploadFile

from src.inference import load_model, predict_image


# Load the trained model once when the API starts.
model = load_model()


async def predict_uploaded_image(file: UploadFile):
    """
    Save an uploaded image temporarily and run model inference.
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

    temp_path = Path("temp_image.png")

    try:
        # Read uploaded image
        image_data = await file.read()

        # Check that the file is not empty
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Save temporarily
        temp_path.write_bytes(image_data)

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

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not process the uploaded image."
        )

    finally:
        # Remove temporary image
        if temp_path.exists():
            temp_path.unlink()