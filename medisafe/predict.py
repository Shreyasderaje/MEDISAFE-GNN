"""High-level prediction engine used by the web dashboard and CLI."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from . import config
from .data import INTERACTION_TYPES, load_dataset
from .explain import explain_pair
from .graph import DrugGraph
from .model import MedisafeGNN
from .train import normalise_adjacency, save_checkpoint

DISCLAIMER = (
    "Potential interaction detected - this is a research/decision-support "
    "prediction and should not replace pharmacist/physician review."
)


class MedisafeEngine:
    """Loads (or trains) a model and answers drug-pair queries."""

    def __init__(self, checkpoint_path: str = None, graph: Optional[DrugGraph] = None,
                 model: Optional[MedisafeGNN] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph = graph
        self.model = model
        self.checkpoint_path = checkpoint_path or config.CHECKPOINT_PATH
        self.metrics: Dict = {}
        self._X = None
        self._A = None
        if model is not None and graph is not None:
            self._prepare()

    @classmethod
    def from_checkpoint(cls, path: str = None) -> "MedisafeEngine":
        path = path or config.CHECKPOINT_PATH
        if not os.path.exists(path):
            return cls()  # untrained engine; caller will train on first use
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload.get("config", config.CONFIG)

        drugs = pd.DataFrame(
            payload.get("drug_records", [{"name": n} for n in payload["drug_names"]])
        )
        features = np.asarray(payload["features"], dtype=np.float32)
        ddi = pd.DataFrame(payload.get("edges", np.zeros((0, 2))),
                           columns=["_i", "_j"])
        graph = cls._rebuild_graph(drugs, ddi, features, payload)

        model = MedisafeGNN(
            feature_dim=features.shape[1], hidden_dim=cfg["hidden_dim"],
            embedding_dim=cfg["embedding_dim"], num_gnn_layers=cfg["num_gnn_layers"],
            heads=cfg["num_attention_heads"], gnn_type=cfg["gnn_type"],
            dropout=cfg["dropout"], decoder_hidden=cfg["decoder_hidden"],
            n_interaction_types=len(payload["interaction_types"]),
            n_adverse_effects=len(payload["adverse_vocab"]),
        )
        model.load_state_dict(payload["state_dict"])
        engine = cls(checkpoint_path=path, graph=graph, model=model)
        engine.metrics = payload.get("metrics", {})
        return engine

    @staticmethod
    def _rebuild_graph(drugs, ddi, features, payload) -> DrugGraph:
        drug_names = list(drugs["name"])
        edges = np.asarray(payload.get("edges", np.zeros((0, 2))), dtype=np.int64)
        records = payload.get("ddi_records")
        if records:
            ddi = pd.DataFrame(records)
        else:
            ddi = pd.DataFrame({
                "drug_1": [drug_names[int(i)] for i, _ in edges],
                "drug_2": [drug_names[int(j)] for _, j in edges],
                "interaction_type": ["" for _ in edges],
                "adverse_effects": ["" for _ in edges],
                "mechanism": ["documented knowledge base edge" for _ in edges],
            })
        g = DrugGraph.__new__(DrugGraph)
        g.drug_names = drug_names
        g.name_to_idx = {n: i for i, n in enumerate(drug_names)}
        g.features = features
        g.drugs = drugs.reset_index(drop=True)
        g.adverse_vocab = list(payload["adverse_vocab"])
        g.edge_index = edges
        g.edge_type = np.asarray(
            payload.get("edge_type", np.zeros(len(edges))), dtype=np.int64
        )
        g.edge_adverse = np.asarray(
            payload.get(
                "edge_adverse",
                np.zeros((len(edges), len(g.adverse_vocab)), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        g.n_drugs = len(drug_names)
        g.adverse_counts = {}
        g.ddi = ddi
        return g

    # ------------------------------------------------------------------ #
    def _prepare(self) -> None:
        self.model.eval()
        self._X = torch.tensor(self.graph.features, dtype=torch.float32,
                               device=self.device)
        self._A = torch.tensor(
            normalise_adjacency(self.graph.adjacency()), dtype=torch.float32,
            device=self.device
        )
        with torch.no_grad():
            self._emb = self.model.encoder(self._X, self._A)

    def train_on_sample_data(self) -> Dict:
        from .train import train as train_model
        drugs, ddi, feats = load_dataset()
        self.graph = DrugGraph(drugs, ddi, feats)
        result = train_model(self.graph)
        self.model = result["model"]
        self.metrics = result["metrics"]
        save_checkpoint(result, self.graph, path=self.checkpoint_path)
        self._prepare()
        return self.metrics

    def is_known_interaction(self, drug_a: str, drug_b: str):
        """Return the known DDI record if it exists in the knowledge base."""
        for _, row in self.graph.ddi.iterrows():
            if {row["drug_1"], row["drug_2"]} == {drug_a, drug_b}:
                return row
        return None

    def predict_pair(self, drug_a: str, drug_b: str) -> Dict:
        """Predict interaction probability, type and adverse effects."""
        if self.model is None or self.graph is None:
            raise RuntimeError("Model not initialised; train or load a checkpoint first.")
        for d in (drug_a, drug_b):
            if d not in self.graph.name_to_idx:
                raise ValueError(f"Unknown drug name: {d}")

        i, j = self.graph.name_to_idx[drug_a], self.graph.name_to_idx[drug_b]
        edge = torch.tensor([[i, j]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            p_inter, type_logits = self.model.decoder(self._emb, edge)
            a_probs = torch.sigmoid(self.model.adverse(self._emb, edge))

        p = float(p_inter.item())
        type_idx = int(type_logits.argmax(dim=1).item())
        interaction_type = INTERACTION_TYPES[type_idx] if p >= 0.5 else "none detected"

        adverse: List[Dict] = []
        probs = a_probs[0].cpu().numpy()
        # adaptive threshold: keep top effects at >= 35% of the max score
        # (absolute probabilities run low on small multi-label data)
        thr = max(0.08, 0.35 * float(probs.max()))
        shown = 0
        for k in np.argsort(-probs):
            if probs[k] < thr or shown >= 4:
                break
            adverse.append({
                "effect": self.graph.adverse_vocab[int(k)],
                "probability": round(float(probs[k]), 4),
                "source": "predicted",
            })
            shown += 1

        known = self.is_known_interaction(drug_a, drug_b)

        # documented effects from the knowledge base always lead the list
        if known is not None and str(known["adverse_effects"]).strip():
            documented = [
                s.strip().lower() for s in str(known["adverse_effects"]).split(";")
                if s.strip()
            ]
            doc_set = set(documented)
            adverse = (
                [{"effect": eff, "probability": None, "source": "documented"}
                 for eff in documented]
                + [a for a in adverse if a["effect"] not in doc_set]
            )
        explanation = explain_pair(
            self.graph, self.model, self._X, self._A, drug_a, drug_b
        )
        confidence = float(min(abs(p - 0.5) * 2, 1.0))

        return {
            "drug_a": drug_a, "drug_b": drug_b,
            "interaction_probability": round(p, 4),
            "interaction_detected": bool(p >= config.CONFIG["interaction_threshold"]),
            "interaction_type": interaction_type,
            "adverse_effects": adverse,
            "confidence": round(confidence, 4),
            "known_in_knowledge_base": bool(known is not None),
            "known_record": {
                "interaction_type": str(known["interaction_type"]),
                "adverse_effects": str(known["adverse_effects"]),
                "mechanism": str(known["mechanism"]),
            } if known is not None else None,
            "explanation": explanation,
            "disclaimer": DISCLAIMER,
        }

    def predict_multi(self, drugs: List[str]) -> Dict:
        """Analyse all pairs in a multi-drug combination."""
        results = [
            self.predict_pair(drugs[x], drugs[y])
            for x in range(len(drugs)) for y in range(x + 1, len(drugs))
        ]
        results.sort(key=lambda r: -r["interaction_probability"])
        return {
            "drugs": drugs, "pairs": results,
            "overall_risk": max(
                (r["interaction_probability"] for r in results), default=0.0
            ),
            "disclaimer": DISCLAIMER,
        }

    def search_drugs(self, query: str, limit: int = 10) -> List[Dict]:
        q = query.strip().lower()
        return [n for n in self.graph.drug_names if q in n.lower()][:limit]

        return g
