import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.pipeline import load_pipeline, preprocess_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="ML API for predicting customer churn using Random Forest",
    version="1.0.0"
)

# Load the trained model
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

# Load the preprocessing pipeline
PIPELINE_PATH = Path(__file__).parent.parent.parent / "models" / "preprocessing_pipeline.pkl"
try:
    preprocessor = load_pipeline(str(PIPELINE_PATH))
except FileNotFoundError:
    raise Exception(f"Preprocessing pipeline not found at {PIPELINE_PATH}. Please train the model first.")


# Pydantic models for request/response validation
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str

class PredictionResponse(BaseModel):
    """Prediction response"""
    churn_prediction: int
    churn_probability: float
    no_churn_probability: float
    optimized_threshold: float
    status: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str


class InfoResponse(BaseModel):
    """API info response"""
    service: str
    version: str
    endpoints: dict
    model_info: dict


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "churn-prediction-api"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([customer.model_dump()])

        # Apply preprocessing pipeline (handles encoding + scaling)
        df_transformed = preprocess_data(df, preprocessor)

        # Predict
        probability = model.predict_proba(df_transformed)[0]

        # Threshold
        threshold = 0.20
        prediction = 1 if probability[1] >= threshold else 0

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability[1]),
            "no_churn_probability": float(probability[0]),
            "optimized_threshold": threshold,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    


@app.get("/info", response_model=InfoResponse)
def info():
    """Get API information"""
    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/info": "GET - API information",
            "/predict": "POST - Predict churn probability",
            "/docs": "GET - Interactive API documentation (Swagger UI)",
            "/redoc": "GET - Alternative API documentation (ReDoc)"
        },
        "model_info": {
            "type": "Random Forest Classifier",
            "model_path": str(MODEL_PATH)
        }
    }


if __name__ == "__main__":
    import uvicorn
    # Run FastAPI development server
    uvicorn.run(app, host="0.0.0.0", port=5000)
