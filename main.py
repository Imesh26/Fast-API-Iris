from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
import json


app = FastAPI(
    title="ML Model API",
    description="API for ML model inference (Iris classification)"
)


# Load model & metadata at startup
try:
    model = joblib.load("model.pkl")
except Exception as e:
    model = None
    print(f"WARNING: Could not load model.pkl at import time: {e}")

try:
    with open("model_info.json", "r") as f:
        MODEL_INFO = json.load(f)
except Exception:
    MODEL_INFO = {
        "model_type": "unknown",
        "problem_type": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target_names": ["setosa", "versicolor", "virginica"],
    }

TARGET_NAMES = MODEL_INFO.get("target_names", ["setosa", "versicolor", "virginica"])
FEATURES = MODEL_INFO.get("features", ["sepal_length", "sepal_width", "petal_length", "petal_width"])


# Pydantic schemas (input validation)
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float] = None


class BatchPredictionInput(BaseModel):
    items: List[PredictionInput]


# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}


# Model metadata endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_type": MODEL_INFO.get("model_type", "unknown"),
        "problem_type": MODEL_INFO.get("problem_type", "classification"),
        "features": FEATURES,
        "test_accuracy": MODEL_INFO.get("test_accuracy", None),
        "target_names": TARGET_NAMES,
    }


# Single prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        if model is None:
            raise RuntimeError("Model not loaded. Train the model and ensure model.pkl is present.")

        # Convert input to correct shape
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])

        pred_idx = int(model.predict(features)[0])

        # Confidence score from predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))
        else:
            confidence = None

        pred_name = TARGET_NAMES[pred_idx] if 0 <= pred_idx < len(TARGET_NAMES) else str(pred_idx)

        return PredictionOutput(prediction=pred_name, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Batch prediction endpoint
@app.post("/predict-batch", response_model=List[PredictionOutput])
def predict_batch(batch: BatchPredictionInput):
    try:
        if model is None:
            raise RuntimeError("Model not loaded. Train the model and ensure model.pkl is present.")

        rows = [
            [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]
            for item in batch.items
        ]

        X = np.array(rows)
        preds = model.predict(X)

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
            confs = np.max(probas, axis=1).tolist()
        else:
            confs = [None] * len(preds)

        outputs = []
        for idx, conf in zip(preds, confs):
            idx = int(idx)
            name = TARGET_NAMES[idx] if 0 <= idx < len(TARGET_NAMES) else str(idx)
            outputs.append(
                PredictionOutput(
                    prediction=name,
                    confidence=(float(conf) if conf is not None else None)
                )
            )

        return outputs

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
