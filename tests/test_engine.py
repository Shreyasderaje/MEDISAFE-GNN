"""Integration tests: prediction engine + Flask API."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medisafe.predict import MedisafeEngine


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Train a quick model on the demo dataset and cache it."""
    config_dir = tmp_path_factory.mktemp("models")
    eng = MedisafeEngine(checkpoint_path=str(config_dir / "test_model.pt"))
    eng.train_on_sample_data()
    return eng


class TestEngine:
    def test_metrics_computed(self, engine):
        m = engine.metrics
        assert 0.0 <= m["test_auc"] <= 1.0
        assert m["n_drugs"] > 50
        assert m["n_interactions"] > 100

    def test_known_interaction_detected(self, engine):
        out = engine.predict_pair("warfarin", "fluconazole")
        assert out["known_in_knowledge_base"]
        assert out["known_record"]["interaction_type"] == "major"
        assert out["interaction_probability"] > 0.5

    def test_unknown_drug_raises(self, engine):
        with pytest.raises(ValueError):
            engine.predict_pair("warfarin", "unobtainium")

    def test_adverse_effects_present(self, engine):
        out = engine.predict_pair("warfarin", "aspirin")
        assert isinstance(out["adverse_effects"], list)
        assert "explanation" in out
        assert "disclaimer" in out

    def test_responsible_messaging(self, engine):
        out = engine.predict_pair("sildenafil", "nitroglycerin")
        assert "should not replace" in out["disclaimer"]

    def test_multi_drug(self, engine):
        res = engine.predict_multi(["warfarin", "aspirin", "fluoxetine"])
        assert len(res["pairs"]) == 3
        assert 0.0 <= res["overall_risk"] <= 1.0

    def test_search(self, engine):
        hits = engine.search_drugs("war")
        assert {"name": "warfarin"} in hits
