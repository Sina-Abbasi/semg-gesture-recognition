# Real-Time sEMG Gesture Recognition for Robotic Arm Control

This repository presents an end-to-end Machine Learning and Signal Processing pipeline designed to classify human hand gestures using surface Electromyography (sEMG) signals. The primary objective is to translate neuromuscular intentions into discrete control commands for **prosthetic hands and robotic arms**.

---

## 🎯 Project Goal & System Overview
- **Signal Input:** 8-channel sEMG signals collected during active hand gestures.
- **Processing Pipeline:** Real-time digital filtering, feature extraction (RMS, MAV,  across 8 channels = 16 features).
- **Classification Engine:** Random Forest model predicting target gestures with high confidence and minimal latency.
- **Deployment:** Containerized REST API (FastAPI + Docker) engineered to stream predictions directly to an embedded micro-controller / robotic arm controller.

## 📊 Feature Extraction Methodology

To capture the amplitude and contraction intensity of the sEMG signals across the 8 channels, we extract key time-domain features:

### 1. Root Mean Square (RMS)
Represents the mean power and physical intensity of the muscle signal. Higher RMS values correlate directly with stronger muscle contractions.
$$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$

### 2. Mean Absolute Value (MAV)
Measures the average magnitude of the sEMG signal over a time window, serving as a reliable indicator of muscle activity level with lower computational complexity.
$$MAV = \frac{1}{N} \sum_{i=1}^{N} |x_i|$$

## Quick Start with Docker

You do not need to install Python or any dependencies manually. Simply pull and run the pre-built Docker image from Docker Hub:

docker run -d -p 8000:8000 --name semg-app sina22sas/semg-api:v1

After running the container, open your browser and go to:
http://localhost:8000/docs to test the API via Swagger UI.

## API Endpoints

- GET / : Health check endpoint
- POST /predict : Classify gesture from 16 pre-extracted features (RMS and MAV)
- POST /predict-raw : Process raw sEMG signals (8 channels), extract features, and predict gesture

## Supported Gestures

1. Rest
2. Fist
3. Flexion
4. Extension
5. Radial Deviation
6. Ulnar Deviation

## 📊 Model Evaluation & Reports

The model evaluation metrics and visualizations are automatically saved in the `reports/` directory:

- **`reports/confusion_matrix.png`**: Visual representation of model predictions across all 6 gesture classes.
- **`reports/classification_report.txt`**: Detailed metrics including Precision, Recall, and F1-Score.

### Confusion Matrix
![Confusion Matrix](reports/confusion_matrix.png)