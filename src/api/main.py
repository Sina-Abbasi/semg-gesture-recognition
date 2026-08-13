import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
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
    docs_url=None,
    redoc_url=None
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
# 4. Schemas with Examples (Paired RMS & MAV)
# ----------------------------------------------------
class FeatureRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=16,
        max_length=16,
        description="16 Time-Domain Features (Paired Interleaved: RMS_ch1, MAV_ch1, ...). Values MUST be within realistic biological bounds (0.0 to 0.01).",
        json_schema_extra={
            "example": [
                0.00045, 0.00035, 0.00021, 0.00018, 0.00030, 0.00025, 0.00015, 0.00012, 
                0.00018, 0.00015, 0.00022, 0.00018, 0.00055, 0.00045, 0.00028, 0.00022
            ]
        }
    )

class RawSignalRequest(BaseModel):
    raw_signals: list[list[float]] = Field(
        ...,
        description="Matrix of raw sEMG values (N samples x 8 channels). Values MUST be in range -0.01 to 0.01.",
        json_schema_extra={
            "example": [
                [0.00065, 0.00028, 0.00029, 0.00007, 0.00009, -0.00005, -0.00042, 0.00014],
                [0.00014, -0.00014, -0.00043, -0.00014, 0.00014, 0.00002, -0.00021, 0.00013],
                [-0.00061, -0.00022, -0.00019, -0.00016, -0.00016, -0.00035, -0.00109, -0.00068]
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
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>sEMG AI Control Center</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
            .btn-accent { background: linear-gradient(135deg, #0284c7, #0369a1); color: white; font-weight: 600; border: none; }
            .btn-accent:hover { background: linear-gradient(135deg, #0369a1, #075985); color: white; }
            .btn-success-gradient { background: linear-gradient(135deg, #16a34a, #15803d); color: white; font-weight: 600; border: none; }
            .badge-custom { background-color: #334155; color: #38bdf8; font-weight: 500; font-size: 0.85rem; }
            .chat-box { height: 320px; overflow-y: auto; background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 15px; }
            .msg-user { background-color: #0284c7; color: white; border-radius: 12px 12px 0 12px; padding: 8px 14px; margin-bottom: 10px; width: fit-content; max-width: 80%; float: right; clear: both; }
            .msg-bot { background-color: #334155; color: #f8fafc; border-radius: 12px 12px 12px 0; padding: 8px 14px; margin-bottom: 10px; width: fit-content; max-width: 80%; float: left; clear: both; }
            .result-badge { font-size: 1.3rem; font-weight: bold; color: #38bdf8; }
            .nav-tabs .nav-link { color: #94a3b8; border: none; font-weight: 600; }
            .nav-tabs .nav-link.active { background-color: #334155; color: #38bdf8; border-radius: 8px 8px 0 0; }
        </style>
    </head>
    <body>
        <div class="container my-5" style="max-width: 1000px;">
            <!-- Header Banner -->
            <div class="card p-4 mb-4 text-center">
                <h1 class="fw-bold text-info mb-2">🦾 sEMG Gesture AI & RAG Control Center</h1>
                <p class="text-secondary mb-3">Real-Time Biosignal Processing & Intelligent Documentation Retrieval Engine</p>
                <div>
                    <span class="badge badge-custom px-3 py-2 me-2"><i class="fa-solid fa-code me-1"></i> API v2.0.0</span>
                    <span class="badge badge-custom px-3 py-2 me-3" style="color: #4ade80;"><i class="fa-solid fa-circle-check text-success me-1"></i> Status: Online</span>
                    <a href="/docs" class="btn btn-sm btn-outline-info me-2"><i class="fa-solid fa-rocket me-1"></i> Swagger Docs</a>
                    <a href="/redoc" class="btn btn-sm btn-outline-secondary"><i class="fa-solid fa-file-lines me-1"></i> ReDoc</a>
                </div>
            </div>

            <div class="row g-4">
                <!-- Section 1: Dual-Mode Gesture Predictor -->
                <div class="col-md-6">
                    <div class="card h-100 p-4">
                        <h4 class="fw-bold text-info mb-3"><i class="fa-solid fa-bolt me-2"></i>Live Gesture Predictor</h4>
                        
                        <!-- Nav Tabs -->
                        <ul class="nav nav-tabs mb-3" id="predictorTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="features-tab" data-bs-toggle="tab" data-bs-target="#features-panel" type="button" role="tab">16 Features</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="raw-tab" data-bs-toggle="tab" data-bs-target="#raw-panel" type="button" role="tab">Raw Signal (8 Ch)</button>
                            </li>
                        </ul>

                        <div class="tab-content" id="predictorTabContent">
                            <!-- Tab 1: 16 Features Input -->
                            <div class="tab-pane fade show active" id="features-panel" role="tabpanel">
                                <p class="text-secondary small">Send pre-extracted features (RMS1, MAV1... in range 0.0 to 0.01):</p>
                                <div class="mb-3">
                                    <textarea id="featureInput" class="form-control bg-dark text-info border-secondary font-monospace" rows="4">[0.00045, 0.00035, 0.00021, 0.00018, 0.00030, 0.00025, 0.00015, 0.00012, 0.00018, 0.00015, 0.00022, 0.00018, 0.00055, 0.00045, 0.00028, 0.00022]</textarea>
                                </div>
                                <button onclick="predictFeatures()" class="btn btn-accent w-100 mb-3"><i class="fa-solid fa-hand me-2"></i> Predict From Features</button>
                            </div>

                            <!-- Tab 2: Raw Signals Input -->
                            <div class="tab-pane fade" id="raw-panel" role="tabpanel">
                                <p class="text-secondary small">Send raw multichannel matrix (N Samples x 8 Channels in range -0.01 to +0.01):</p>
                                <div class="mb-3">
                                    <textarea id="rawInput" class="form-control bg-dark text-info border-secondary font-monospace" rows="4">[[0.00065, 0.00028, 0.00029, 0.00007, 0.00009, -0.00005, -0.00042, 0.00014], [0.00014, -0.00014, -0.00043, -0.00014, 0.00014, 0.00002, -0.00021, 0.00013], [-0.00061, -0.00022, -0.00019, -0.00016, -0.00016, -0.00035, -0.00109, -0.00068]]</textarea>
                                </div>
                                <button onclick="predictRaw()" class="btn btn-accent w-100 mb-3"><i class="fa-solid fa-wave-square me-2"></i> Predict From Raw Signal</button>
                            </div>
                        </div>
                        
                        <!-- Prediction Output Box -->
                        <div id="predictResultBox" class="p-3 border border-secondary rounded bg-dark d-none">
                            <div class="text-secondary small">Prediction Result:</div>
                            <div id="gestureOutput" class="result-badge my-1">--</div>
                            <div class="text-secondary small">Gesture Code: <span id="classIdOutput" class="text-light">--</span></div>
                        </div>
                    </div>
                </div>

                <!-- Section 2: Interactive RAG AI Chatbot -->
                <div class="col-md-6">
                    <div class="card h-100 p-4">
                        <h4 class="fw-bold text-warning mb-3"><i class="fa-solid fa-robot me-2"></i>RAG Knowledge Assistant</h4>
                        <p class="text-secondary small">Ask any question about the project documentation (BM25 + Llama 3):</p>
                        
                        <div id="chatBox" class="chat-box mb-3">
                            <div class="msg-bot">Welcome! Ask me anything about sEMG signal processing, RMS/MAV feature extraction, or system architecture.</div>
                        </div>

                        <div class="input-group">
                            <input type="text" id="ragQuestion" class="form-control bg-dark text-light border-secondary" placeholder="e.g. How is RMS calculated?" onkeydown="if(event.key==='Enter') askRAG()">
                            <button onclick="askRAG()" class="btn btn-success-gradient"><i class="fa-solid fa-paper-plane"></i></button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            async function predictFeatures() {
                const rawVal = document.getElementById('featureInput').value;
                const resultBox = document.getElementById('predictResultBox');
                try {
                    const features = JSON.parse(rawVal);
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features: features })
                    });
                    const data = await response.json();
                    
                    if (response.ok && data.status === "success") {
                        document.getElementById('gestureOutput').innerText = data.gesture_name;
                        document.getElementById('classIdOutput').innerText = data.gesture_code;
                        resultBox.classList.remove('d-none');
                    } else {
                        alert("API Error: " + (data.detail || "Invalid Payload"));
                    }
                } catch(e) {
                    alert("Please provide a valid 16-element JSON array!");
                }
            }

            async function predictRaw() {
                const rawVal = document.getElementById('rawInput').value;
                const resultBox = document.getElementById('predictResultBox');
                try {
                    const rawSignals = JSON.parse(rawVal);
                    const response = await fetch('/predict-raw', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ raw_signals: rawSignals })
                    });
                    const data = await response.json();
                    
                    if (response.ok && data.status === "success") {
                        document.getElementById('gestureOutput').innerText = data.gesture_name;
                        document.getElementById('classIdOutput').innerText = data.gesture_code;
                        resultBox.classList.remove('d-none');
                    } else {
                        alert("API Error: " + (data.detail || "Invalid Payload"));
                    }
                } catch(e) {
                    alert("Please provide a valid 2D array matrix (N x 8 channels)!");
                }
            }

            async function askRAG() {
                const qInput = document.getElementById('ragQuestion');
                const chatBox = document.getElementById('chatBox');
                const question = qInput.value.trim();
                if(!question) return;

                chatBox.innerHTML += `<div class="msg-user">${question}</div>`;
                qInput.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                const loadingId = 'load-' + Date.now();
                chatBox.innerHTML += `<div id="${loadingId}" class="msg-bot">Thinking... 🧠</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch('/rag/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });
                    const data = await response.json();
                    document.getElementById(loadingId).innerText = data.answer;
                } catch(e) {
                    document.getElementById(loadingId).innerText = "Error connecting to RAG assistant!";
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

# ----------------------------------------------------
# Custom Dark Theme for Swagger UI & ReDoc
# ----------------------------------------------------
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={"deepLinking": True, "syntaxHighlight.theme": "monokai"}
    )

@app.get("/redoc", include_in_schema=False)
def custom_redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc"
    )

# ----------------------------------------------------
# 6. Endpoints
# ----------------------------------------------------
@app.post("/predict", tags=["Gesture Classification"])
def predict_features(request: FeatureRequest):
    if len(request.features) != 16:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly 16 features, but got {len(request.features)}",
        )

    features_arr = np.array(request.features)
    
    # Validation: strict bound checking based on actual dataset scale
    if np.any(features_arr < 0.0) or np.any(features_arr > 0.01):
        raise HTTPException(
            status_code=400,
            detail="Out of bounds! Feature values must be strictly between 0.0 and 0.01 based on trained sEMG scale."
        )

    if hasattr(model, "feature_names_in_"):
        feature_df = pd.DataFrame([request.features], columns=model.feature_names_in_)
        prediction = int(model.predict(feature_df)[0])
    else:
        input_data = features_arr.reshape(1, -1)
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

    # Validation: strict bound checking based on actual dataset scale
    if np.any(signals < -0.01) or np.any(signals > 0.01):
        raise HTTPException(
            status_code=400,
            detail="Out of bounds! Raw signal values must be strictly between -0.01 and +0.01 V/mV based on trained sEMG scale."
        )

    # Calculate RMS & MAV per channel
    rms_vals = np.sqrt(np.mean(signals**2, axis=0))
    mav_vals = np.mean(np.abs(signals), axis=0)

    # Interleave RMS & MAV per channel (RMS_ch1, MAV_ch1, RMS_ch2, MAV_ch2, ...)
    extracted_features = []
    for r, m in zip(rms_vals, mav_vals):
        extracted_features.extend([r, m])

    if hasattr(model, "feature_names_in_"):
        feature_df = pd.DataFrame([extracted_features], columns=model.feature_names_in_)
        prediction = int(model.predict(feature_df)[0])
    else:
        input_data = np.array(extracted_features).reshape(1, -1)
        prediction = int(model.predict(input_data)[0])

    gesture_name = GESTURE_MAP.get(prediction, "Unknown")

    return {
        "status": "success",
        "gesture_code": prediction,
        "gesture_name": gesture_name,
        "extracted_features_count": 16,
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