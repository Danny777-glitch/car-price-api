import pytest
from app import app

VALID_CAR = {
 "brand": "BMW", "model_year": 2018, "mileage": 45000,
 "fuel_type": "Gasoline", "transmission": "A/T",
 "hp": 300, "engine_displacement": 3.0,
 "is_v_engine": True, "accident": False, "clean_title": True,
}

@pytest.fixture
def client():
 return app.test_client() 

def test_health_ok(client):
 r = client.get("/health")
 assert r.status_code == 200
 assert r.get_json()["model_loaded"] is True

def test_predict_returns_positive_price(client):
 r = client.post("/predict", json=VALID_CAR)
 assert r.status_code == 200
 assert r.get_json()["predicted_price"] > 0
