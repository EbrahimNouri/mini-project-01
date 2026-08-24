import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from train_mlp import FraudMLP, predict_proba as mlp_predict_proba

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_THRESHOLD = 0.5
MLP_MODEL_PATH = MODEL_DIR / "mlp_model.pt"


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionRequest(BaseModel):
    transaction: Transaction
    threshold: Optional[float] = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    model: Optional[str] = Field(default="logistic_regression")


class BatchPredictionRequest(BaseModel):
    transactions: List[Transaction]
    threshold: Optional[float] = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    model: Optional[str] = Field(default="logistic_regression")


class PredictionResponse(BaseModel):
    prediction: str
    class_id: int
    probability: float
    threshold: float
    model_used: str
    status: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    artifacts_available: bool
    version: str


class ModelLoader:
    _instance = None
    _models = {}
    _scaler = None
    _encoder = None
    _mlp_checkpoint = None
    _is_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_models(cls):
        if cls._is_loaded:
            return

        print("\n" + "=" * 60)
        print("LOADING MODELS FOR FASTAPI SERVICE")
        print("=" * 60)

        model_files = {
            "logistic_regression": "model.pkl",
            "knn": "knn_model.pkl",
            "decision_tree": "dt_model.pkl",
        }

        for name, filename in model_files.items():
            path = MODEL_DIR / filename
            if path.exists():
                cls._models[name] = joblib.load(path)
                print(f"  [OK] {name} loaded")
            else:
                cls._models[name] = None
                print(f"  [WARN] {name} not found")

        if MLP_MODEL_PATH.exists():
            cls._mlp_checkpoint = torch.load(MLP_MODEL_PATH, map_location='cpu')
            mlp = FraudMLP(cls._mlp_checkpoint["input_dim"])
            mlp.load_state_dict(cls._mlp_checkpoint["state_dict"])
            mlp.eval()
            cls._models["mlp"] = mlp
            print(f"  [OK] mlp loaded")
        else:
            cls._models["mlp"] = None
            print(f"  [WARN] mlp not found")

        try:
            cls._scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            cls._encoder = joblib.load(MODEL_DIR / "encoder.pkl")
            cls._artifacts_available = True
            print(f"  [OK] scaler and encoder loaded")
        except Exception as e:
            cls._artifacts_available = False
            print(f"  [ERROR] Failed to load artifacts: {e}")

        cls._is_loaded = True
        print("=" * 60 + "\n")

    @classmethod
    def get_model(cls, model_name: str):
        if not cls._is_loaded:
            cls.load_models()
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' not found")
        if cls._models[model_name] is None:
            raise ValueError(f"Model '{model_name}' is not available")
        return cls._models[model_name]

    @classmethod
    def get_scaler(cls):
        if not cls._is_loaded:
            cls.load_models()
        return cls._scaler

    @classmethod
    def get_encoder(cls):
        if not cls._is_loaded:
            cls.load_models()
        return cls._encoder

    @classmethod
    def get_available_models(cls):
        if not cls._is_loaded:
            cls.load_models()
        return {name: model is not None for name, model in cls._models.items()}

    @classmethod
    def is_ready(cls):
        if not cls._is_loaded:
            cls.load_models()
        return cls._artifacts_available and any(v is not None for v in cls._models.values())


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Starting FastAPI Fraud Detection Service...")
    ModelLoader.load_models()
    yield
    print("\n👋 Shutting down FastAPI service...")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Detect fraudulent credit card transactions",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_transaction(transaction: Transaction) -> np.ndarray:
    df = pd.DataFrame([transaction.dict()])
    encoder = ModelLoader.get_encoder()
    encoded = encoder.transform(df)
    scaler = ModelLoader.get_scaler()
    scaled = scaler.transform(encoded)
    return scaled


def predict_with_model(features: np.ndarray, model_name: str, threshold: float) -> Dict:
    model = ModelLoader.get_model(model_name)

    if model_name == "mlp":
        X_t = torch.as_tensor(features, dtype=torch.float32)
        proba = mlp_predict_proba(model, X_t)[0]
    else:
        proba = model.predict_proba(features)[0][1]

    class_id = 1 if proba >= threshold else 0
    prediction = "Fraud" if class_id == 1 else "Legitimate"

    return {
        "prediction": prediction,
        "class_id": class_id,
        "probability": float(proba),
        "threshold": threshold,
        "model_used": model_name,
    }


@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "models": "/models"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    available = ModelLoader.get_available_models()
    models_loaded = [name for name, loaded in available.items() if loaded]
    return HealthResponse(
        status="healthy" if models_loaded and ModelLoader._artifacts_available else "degraded",
        models_loaded=models_loaded,
        artifacts_available=ModelLoader._artifacts_available,
        version="1.0.0"
    )


@app.get("/models")
async def list_models():
    available = ModelLoader.get_available_models()
    return {
        "models": {
            name: {
                "available": loaded,
                "type": "sklearn" if name != "mlp" else "pytorch"
            }
            for name, loaded in available.items()
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    try:
        if not ModelLoader.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded. Run training first."
            )

        features = prepare_transaction(request.transaction)
        result = predict_with_model(features, request.model, request.threshold)

        return PredictionResponse(
            prediction=result["prediction"],
            class_id=result["class_id"],
            probability=round(result["probability"], 6),
            threshold=result["threshold"],
            model_used=result["model_used"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    try:
        if not ModelLoader.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded."
            )

        predictions = []
        fraud_count = 0

        for transaction in request.transactions:
            features = prepare_transaction(transaction)
            result = predict_with_model(features, request.model, request.threshold)

            if result["class_id"] == 1:
                fraud_count += 1

            predictions.append({
                "prediction": result["prediction"],
                "class_id": result["class_id"],
                "probability": round(result["probability"], 6),
                "threshold": result["threshold"],
                "model_used": result["model_used"],
                "status": "success"
            })

        return {
            "predictions": predictions,
            "total": len(predictions),
            "fraud_count": fraud_count,
            "legitimate_count": len(predictions) - fraud_count,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("FRAUD DETECTION API")
    print("=" * 60)
    print("Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    print("=" * 60)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)