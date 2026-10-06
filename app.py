import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
candidate_paths = []

if os.getenv("MODEL_PATH"):
    candidate_paths.append(Path(os.getenv("MODEL_PATH")))

candidate_paths.extend(
    [
        BASE_DIR / "artifacts" / "models" / "best_model.pkl",
        BASE_DIR.parent / "artifacts" / "models" / "best_model.pkl",
        BASE_DIR.parent.parent / "artifacts" / "models" / "best_model.pkl",
    ]
)

MODEL_PATH = next((path for path in candidate_paths if path.exists()), candidate_paths[0])

FEATURE_COLUMNS = [
    "brand",
    "fuel_type",
    "transmission",
    "clean_title",
    "hp",
    "engine displacement",
    "is_v_engine",
    "Accident_Impact",
    "Vehicle_Age",
    "Mileage_per_Year",
    "Age_Mid",
    "Age_Old",
    "Age_Very Old",
    "Milage_Medium",
    "Milage_High",
    "Milage_Very High",
]

FUEL_TYPE_MAP = {
    "DIESEL": 0,
    "E85 FLEX FUEL": 1,
    "GASOLINE": 2,
    "HYBRID": 3,
    "OTHER": 4,
}

TRANSMISSION_MAP = {
    "A/T": 0,
    "CVT": 1,
    "M/T": 2,
    "OTHER": 3,
}

app = FastAPI(title="Used Car Price Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_error = None

try:
    loaded_object = joblib.load(MODEL_PATH)

    if hasattr(loaded_object, "predict"):
        model = loaded_object
        print(f"✅ Model loaded successfully: {MODEL_PATH}")
    else:
        model = None
        model_error = (
            f"Invalid model artifact at {MODEL_PATH}: object type is "
            f"{type(loaded_object).__name__}. Expected a trained model object with a predict() method."
        )
        print(f"⚠️ {model_error}")
except Exception as exc:
    model_error = str(exc)
    print(f"⚠️ Model could not be loaded: {exc}")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mode": "demo" if model is None else "live",
        "model_path": str(MODEL_PATH),
        "model_error": model_error,
    }


class PredictionInput(BaseModel):
    brand: str
    fuel_type: str
    transmission: str
    clean_title: bool = True
    model_year: int
    hp: float
    engine_displacement: float
    is_v_engine: bool = False
    accident_reported: bool = False
    milage: float


def map_fuel_type(value: str) -> int:
    normalized = str(value).strip().upper()
    return FUEL_TYPE_MAP.get(normalized, FUEL_TYPE_MAP["OTHER"])


def map_transmission(value: str) -> int:
    normalized = str(value).strip().upper()
    return TRANSMISSION_MAP.get(normalized, TRANSMISSION_MAP["OTHER"])


def to_binary(value):
    return 1 if bool(value) else 0


def age_bin_flags(vehicle_age: float):
    if vehicle_age <= 5:
        return {"Age_Mid": 0, "Age_Old": 0, "Age_Very Old": 0}
    if vehicle_age <= 8:
        return {"Age_Mid": 1, "Age_Old": 0, "Age_Very Old": 0}
    if vehicle_age <= 13:
        return {"Age_Mid": 0, "Age_Old": 1, "Age_Very Old": 0}
    return {"Age_Mid": 0, "Age_Old": 0, "Age_Very Old": 1}


def mileage_bin_flags(mileage: float):
    if mileage <= 27343:
        return {"Milage_Medium": 0, "Milage_High": 0, "Milage_Very High": 0}
    if mileage <= 57737.5:
        return {"Milage_Medium": 1, "Milage_High": 0, "Milage_Very High": 0}
    if mileage <= 95811.5:
        return {"Milage_Medium": 0, "Milage_High": 1, "Milage_Very High": 0}
    return {"Milage_Medium": 0, "Milage_High": 0, "Milage_Very High": 1}


def build_feature_frame(data: PredictionInput) -> pd.DataFrame:
    vehicle_age = max(1, 2025 - int(data.model_year))
    mileage_per_year = float(data.milage) / vehicle_age if vehicle_age > 0 else float(data.milage)

    row = {
        "brand": data.brand.strip(),
        "fuel_type": map_fuel_type(data.fuel_type),
        "transmission": map_transmission(data.transmission),
        "clean_title": to_binary(data.clean_title),
        "hp": float(data.hp),
        "engine displacement": float(data.engine_displacement),
        "is_v_engine": to_binary(data.is_v_engine),
        "Accident_Impact": to_binary(data.accident_reported),
        "Vehicle_Age": vehicle_age,
        "Mileage_per_Year": mileage_per_year,
    }
    row.update(age_bin_flags(vehicle_age))
    row.update(mileage_bin_flags(float(data.milage)))
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        demo_prediction = (data.hp * 0.8 + data.engine_displacement * 1200 + data.milage * 0.002) / 2
        return {
            "prediction": round(float(demo_prediction), 2),
            "status": "demo",
            "message": (
                "Demo mode is active because no valid trained model file was found. "
                "The saved notebook model is in C:/Users/student/Downloads/mlops/artifacts/models/best_model.pkl."
            ),
        }

    try:
        input_frame = build_feature_frame(data)
        prediction = model.predict(input_frame)
        value = prediction[0]
        try:
            value = value.item()
        except AttributeError:
            pass
        return {"prediction": float(value), "status": "live"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
