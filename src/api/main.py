import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from rag.rag_engine import ProjectRAGEngine

# ----------------------------------------------------
# 1. FastAPI Customization & Styling
# ----------------------------------------------------
app = FastAPI(
    title="🤖 sEMG Gesture Recognition & RAG AI Engine",
    description="""
    ### ⚡ Real-Time Muscle Signal Classifier & Intelligent Assistant
    
    Welcome to the **sEMG AI Engine** interface! This API bridges biosignal processing with modern LLMs.
    
    * **🔮 Gesture Classification**: Predict hand gestures from pre-extracted features or raw 8-channel sEMG signals.
    * **📚 RAG Assistant**: Ask any technical question about signal processing, feature calculation, or architecture.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ----------------------------------------------------
# 2. Loading ML Model
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(
    os.path.join(BASE_DIR, "..", "models", "gesture_rf_model.pkl")
)

model = joblib.load(model_path)

GESTURE_MAP = {
    1: "Rest ✋",
    2: "Fist ✊",
    3: "Flexion 🤛",
    4: "Extension 🤜",
    5: "Radial Deviation 👈",
    6: "Ulnar Deviation 👉",
}

# ----------------------------------------------------
# 3. Initializing RAG Engine
# ----------------------------------------------------
rag_engine = ProjectRAGEngine(docs_path=".")
try:
    rag_engine.build_index()
except Exception as e:
    print(f"Warning: RAG index build failed: {e}")

# ----------------------------------------------------
# 4. Schemas with Examples
# ----------------------------------------------------
class FeatureRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="16 Time-Domain Features (8 RMS + 8 MAV)",
        json_schema_extra={
            "example": [
                0.24, 0.18, 0.35, 0.42, 0.12, 0.29, 0.31, 0.20, # RMS (8 channels)
                0.15, 0.11, 0.22, 0.28, 0.08, 0.19, 0.21, 0.14  # MAV (8 channels)
            ]
        }
    )

class RawSignalRequest(BaseModel):
    raw_signals: list[list[float]] = Field(
        ...,
        description="Matrix of raw sEMG values (N samples x 8 channels)",
        json_schema_extra={
            "example": [
                [0.12, -0.05, 0.31, 0.44, -0.11, 0.21, 0.05, -0.02],
                [0.18, -0.02, 0.29, 0.51, -0.08, 0.25, 0.08, -0.01],
                [0.22,  0.01, 0.34, 0.48, -0.05, 0.28, 0.10,  0.02]
            ]
        }
    )

class RAGQueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Ask anything about the sEMG project documentation",
        json_schema_extra={
            "example": "How is RMS calculated and what features are extracted?"
        }
    )

# ----------------------------------------------------
# 5. Interactive UI Dashboard at Root `/`
# ----------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>sEMG AI Control Center</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
        <style>
            body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; }
            main { max-width: 900px; margin: 40px auto; padding: 20px; }
            .card { background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }
            h1 { color: #38bdf8; font-weight: 800; }
            h3 { color: #94a3b8; }
            .badge { background: #0284c7; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; }
            a.button { display: inline-block; background: #0284c7; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: bold; }
            a.button:hover { background: #0369a1; }
        </style>
    </head>
    <body>
        <main>
            <div class="card" style="text-align: center;">
                <h1>🦾 sEMG Gesture AI & RAG Control Center</h1>
                <p>Real-Time Biosignal Processing & Intelligent Documentation Retrieval Engine</p>
                <p><span class="badge">API v2.0.0</span> &nbsp; <span class="badge" style="background:#16a34a">Status: Online</span></p>
                <div style="margin-top: 20px;">
                    <a href="/docs" class="button">🚀 Open Swagger Interactive Docs</a>
                    <a href="/redoc" class="button" style="background:#475569; margin-left:10px;">📄 View ReDoc</a>
                </div>
            </div>

            <div class="card">
                <h3>⚡ Active Endpoints Summary</h3>
                <ul>
                    <li><b>POST <code>/predict</code></b>: Direct gesture classification via 16 time-domain features.</li>
                    <li><b>POST <code>/predict-raw</code></b>: Automatic feature extraction (RMS + MAV) & prediction from raw signals.</li>
                    <li><b>POST <code>/rag/ask</code></b>: Natural language QA over project reports & README (Powered by Llama 3).</li>
                </ul>
            </div>
        </main>
    </body>
    </html>
    """

# ----------------------------------------------------
# 6. Endpoints
# ----------------------------------------------------
@app.post("/predict", tags=["Gesture Classification"])
def predict_features(request: FeatureRequest):
    if len(request.features) != 16:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 16 features (8 RMS + 8 MAV), but got {len(request.features)}",
        )

    input_data = np.array(request.features).reshape(1, -1)
    prediction = int(model.predict(input_data)[0])
    gesture_name = GESTURE_MAP.get(prediction, "Unknown")

    return {
        "status": "success",
        "gesture_code": prediction,
        "gesture_name": gesture_name,
    }

@app.post("/predict-raw", tags=["Gesture Classification"])
def predict_from_raw_signal(request: RawSignalRequest):
    signals = np.array(request.raw_signals)

    if signals.ndim != 2 or signals.shape[1] != 8:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 2D matrix with 8 channels, got shape: {signals.shape}",
        )

    rms_vals = np.sqrt(np.mean(signals**2, axis=0))
    mav_vals = np.mean(np.abs(signals), axis=0)

    extracted_features = np.hstack([rms_vals, mav_vals]).reshape(1, -1)
    prediction = int(model.predict(extracted_features)[0])
    gesture_name = GESTURE_MAP.get(prediction, "Unknown")

    return {
        "status": "success",
        "gesture_code": prediction,
        "gesture_name": gesture_name,
        "extracted_features_count": extracted_features.shape[1],
    }

@app.post("/rag/ask", tags=["RAG Intelligent Assistant"])
def ask_rag_documentation(request: RAGQueryRequest):
    """
    Query the project documentation using RAG + Llama 3.
    """
    try:
        answer = rag_engine.ask(request.question)
        return {
            "status": "success",
            "question": request.question,
            "answer": answer,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)