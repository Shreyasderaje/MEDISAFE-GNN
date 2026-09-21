"""Flask web dashboard for MEDISAFE-GNN.

Endpoints:
  GET  /                 dashboard page
  GET  /api/drugs        searchable drug list
  POST /api/predict     analyse a drug combination (1-4 drugs)
  GET  /api/model        model metadata + test metrics
"""
from __future__ import annotations

import threading

from flask import Flask, jsonify, render_template, request

from . import __version__
from .predict import DISCLAIMER, MedisafeEngine

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

_engine: MedisafeEngine | None = None
_lock = threading.Lock()


def get_engine() -> MedisafeEngine:
    """Lazily create the engine, training the demo model if no checkpoint."""
    global _engine
    with _lock:
        if _engine is None or _engine.model is None:
            eng = MedisafeEngine.from_checkpoint()
            if eng.model is None:
                app.logger.info("No checkpoint found - training demo model...")
                eng.train_on_sample_data()
                app.logger.info("Training complete: %s", eng.metrics)
            _engine = eng
        return _engine


# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html", version=__version__,
                           disclaimer=DISCLAIMER)


@app.route("/api/drugs")
def api_drugs():
    q = request.args.get("q", "").strip().lower()
    engine = get_engine()
    matches = engine.search_drugs(q, limit=15)
    return jsonify({"drugs": matches})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    drugs = [str(d).strip() for d in data.get("drugs", []) if str(d).strip()]
    drugs = list(dict.fromkeys(drugs))[:4]  # unique, max 4

    if len(drugs) < 2:
        return jsonify({"error": "Please select at least two drugs."}), 400

    engine = get_engine()
    valid = set(engine.graph.drug_names)
    unknown = [d for d in drugs if d not in valid]
    if unknown:
        return jsonify({"error": f"Unknown drugs: {', '.join(unknown)}"}), 400

    result = engine.predict_multi(drugs)
    result["model_metrics"] = engine.metrics
    return jsonify(result)


@app.route("/api/model")
def api_model():
    engine = get_engine()
    g = engine.graph
    return jsonify({
        "version": __version__,
        "metrics": engine.metrics,
        "n_drugs": g.n_drugs if g else 0,
        "n_known_interactions": int(len(g.edge_index)) if g else 0,
        "adverse_effect_vocabulary": g.adverse_vocab if g else [],
        "disclaimer": DISCLAIMER,
    })


def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run(debug=True)
