from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent  # racine projet
MODEL_PATH = APP_DIR / "model" / "model.pkl"

app = FastAPI(title="Bank Churn Prediction API", version="1.0")

class ChurnFeatures(BaseModel):
    age: float
    credit_score: float
    balance: float
    tenure: float
    products: float
    is_active: float

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python train.py"
        )
    return joblib.load(MODEL_PATH)

model = load_model()
FEATURES_ORDER = ["age", "credit_score", "balance", "tenure", "products", "is_active"]

@app.get("/")
def root():
    return {"status": "ok", "message": "Churn API is running"}

@app.post("/predict")
def predict(features: ChurnFeatures):
    x = [[
        features.age,
        features.credit_score,
        features.balance,
        features.tenure,
        features.products,
        features.is_active
    ]]
    pred = model.predict(x)[0]
    proba = float(model.predict_proba(x)[0][1])
    return {"churn": int(pred), "churn_probability": proba, "features_order": FEATURES_ORDER}
