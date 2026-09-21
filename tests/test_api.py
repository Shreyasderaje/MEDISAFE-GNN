"""End-to-end tests of the Flask API using the test client."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medisafe import app as flask_app
from medisafe.predict import MedisafeEngine


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("api_models")
    eng = MedisafeEngine(checkpoint_path=str(model_dir / "api_model.pt"))
    eng.train_on_sample_data()

    flask_app.app.config["TESTING"] = True
    # inject trained engine so the API does not retrain
    import medisafe.app as appmod
    appmod._engine = eng
    with flask_app.app.test_client() as c:
        yield c


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"MEDISAFE-GNN" in res.data
    assert b"should not replace" in res.data


def test_drug_search(client):
    res = client.get("/api/drugs?q=war")
    assert res.status_code == 200
    assert {"name": "warfarin"} in res.get_json()["drugs"]


def test_predict_endpoint(client):
    res = client.post("/api/predict", json={"drugs": ["warfarin", "aspirin"]})
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["pairs"]) == 1
    assert 0 <= data["pairs"][0]["interaction_probability"] <= 1
    assert "disclaimer" in data


def test_predict_requires_two(client):
    res = client.post("/api/predict", json={"drugs": ["aspirin"]})
    assert res.status_code == 400


def test_predict_unknown_drug(client):
    res = client.post("/api/predict", json={"drugs": ["aspirin", "xyznotadrug"]})
    assert res.status_code == 400


def test_model_endpoint(client):
    res = client.get("/api/model")
    assert res.status_code == 200
    data = res.get_json()
    assert data["n_drugs"] > 50
    assert "metrics" in data
