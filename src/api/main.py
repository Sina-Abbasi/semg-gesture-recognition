import os 
import joblib 
import numpy as np 
from fastapi import FastAPI , HTTPException 
from pydantic import BaseModel 



app = FastAPI(
    title="sEMG Gesture Recognition API",
    description="API for predicting hand gestures from sEMG signal features or raw signals.",
    version="1.1.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(
    os.path.join(BASE_DIR, "..", "models", "gesture_rf_model.pkl")
)

model = joblib.load(model_path)

# Gesturing Code to Name Mapping
GESTURE_MAP = {
    1: "Rest",
    2: "Fist",
    3: "Flexion",
    4: "Extension",
    5: "Radial Deviation",
    6: "Ulnar Deviation",
}


# Defining Request Data Schema
class FeatureRequest(BaseModel):
    features: list[float]

class RawSignalRequest(BaseModel):
    raw_signals: list[list[float]]


# Endpoints
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "sEMG Gesture Recognition API is running!",
    }

# Endpoint 1 (predict from features)

@app.post("/predict")
def predict_features(request: FeatureRequest):
    # Validating feature vector length (16 features expected)
    if len(request.features) != 16:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 16 features (8 RMS + 8 MAV), but got {len(request.features)}",
        )

    # Converting list to 2D numpy array for model input
    input_data = np.array(request.features).reshape(1, -1)

    # Predict class and probability
    prediction = int(model.predict(input_data)[0])
    gesture_name = GESTURE_MAP.get(prediction, "Unknown")

    return {
        "gesture_code": prediction,
        "gesture_name": gesture_name,
        "status": "success",
    }

# Endpoint 2 (predict from raw signal)

@app.post("/predict-raw")
def predict_from_raw_signal(request: RawSignalRequest):
    signals = np.array(request.raw_signals)

    # validation raw signal length and dim (8 colomns and 2 dim expected)
    if signals.ndim  != 2 or signals.shape[1] != 8:
        raise HTTPException(
            status_code=400,
            detail=f"Expected a 2D matrix with 8 channels (columns), got shape: {signals.shape}")

    rms_vals = np.sqrt(np.mean(signals**2, axis=0))
    mav_vals = np.mean(np.abs(signals), axis =0)

    # merging rms and mav together
    extracted_features = np.hstack([rms_vals, mav_vals]).reshape(1, -1)

    prediction = int(model.predict(extracted_features)[0])
    gesture_name= GESTURE_MAP.get(prediction, "Unknown")

    return { 
        "gesture_code": prediction,
        "gesture_name": gesture_name,
        "status": "success",
        "extracted_features_count": extracted_features.shape[1],

    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)