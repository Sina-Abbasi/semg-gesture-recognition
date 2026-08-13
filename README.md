# sEMG Gesture Recognition System

This project is a Machine Learning pipeline and REST API built with FastAPI to classify hand gestures using surface Electromyography (sEMG) signal features or raw signals.

The whole application is fully containerized using Docker for seamless execution and deployment.

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