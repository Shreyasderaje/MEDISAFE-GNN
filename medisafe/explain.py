"""Explainability utilities for MEDISAFE-GNN.

Predictions are explained by combining three evidence sources:
  1. GAT attention mass flowing along the shortest interaction paths
     between the two drugs (model-internal evidence).
  2. Shared pharmacological annotations (ATC classes, CYP substrates /
     inhibitors) from the curated drug metadata.
  3. The learned adverse-effect probabilities themselves.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .graph import DrugGraph
from .model import MedisafeGNN


def _attention_paths(graph: DrugGraph, model: MedisafeGNN, X: torch.Tensor,
                     A_norm: torch.Tensor, src: int, dst: int,
                     max_hops: int = 2) -> List[Dict]:
    """Collect high-attention message paths between two drugs."""
    attentions = []
    hooks = []

    def make_hook(store):
        def hook(module, inp, out):
            store.append(module.last_attention)
        return hook

    for layer in model.encoder.layers:
        hooks.append(layer.register_forward_hook(make_hook(attentions)))

    with torch.no_grad():
        model.encoder(X, A_norm)
    for h in hooks:
        h.remove()

    paths: List[Dict] = []
    if not attentions:
        return paths

    # layer-1 attention (N, N, heads): find strongest 1-hop and 2-hop routes
    attn = attentions[0].mean(dim=-1).cpu().numpy()  # (N, N)
    n = graph.n_drugs
    direct = float(attn[src, dst]) if 0 <= src < n and 0 <= dst < n else 0.0

    # strongest 2-hop path via an intermediary
    best_mid, best_score = None, 0.0
    row = attn[src] * attn[:, dst]
    for m in range(n):
        if m in (src, dst):
            continue
        score = float(row[m])
        if score > best_score:
            best_score, best_mid = score, m

    if direct > 0.01:
        paths.append({
            "route": [graph.drug_names[src], graph.drug_names[dst]],
            "hops": 1,
            "attention_mass": round(direct, 4),
            "interpretation": "Direct message exchange between the two drugs.",
        })
    if best_mid is not None and best_score > 0.01:
        paths.append({
            "route": [
                graph.drug_names[src],
                graph.drug_names[best_mid],
                graph.drug_names[dst],
            ],
            "hops": 2,
            "attention_mass": round(best_score, 4),
            "interpretation": (
                f"Indirect coupling mediated by {graph.drug_names[best_mid]} "
                "(drugs that share neighbours in the interaction graph tend to "
                "share interaction risks)."
            ),
        })
    return paths[:2]


def _shared_annotations(graph: DrugGraph, drug_a: str, drug_b: str) -> List[str]:
    """Human-readable pharmacological overlaps of the two drugs."""
    meta = graph.drugs.set_index("name")
    if drug_a not in meta.index or drug_b not in meta.index:
        return []
    ra, rb = meta.loc[drug_a], meta.loc[drug_b]
    notes: List[str] = []

    atc_a = {a.strip() for a in str(ra.get("atc_class", "")).split(";") if a.strip()}
    atc_b = {b.strip() for b in str(rb.get("atc_class", "")).split(";") if b.strip()}
    shared_atc = atc_a & atc_b
    if shared_atc:
        notes.append(
            "Both drugs share ATC class(es): " + ", ".join(sorted(shared_atc)) +
            " - additive pharmacodynamic effects are plausible."
        )

    def _cyp_set(value) -> set:
        if value is None or (isinstance(value, float) and value != value):
            return set()  # NaN
        s = str(value).strip().lower()
        if not s or s in ("nan", "none", "null", "nan;"):
            return set()
        return {x.strip().lower() for x in s.split(";") if x.strip()
                and x.strip().lower() not in ("nan", "none", "null")}

    for col_a, col_b, label in (
        ("cyp_substrate", "cyp_inhibitor", "inhibits the metabolism of"),
        ("cyp_inhibitor", "cyp_substrate", "levels are raised by"),
        ("cyp_substrate", "cyp_substrate", "competes with"),
    ):
        a_set = _cyp_set(ra.get(col_a))
        b_set = _cyp_set(rb.get(col_b))
        for enzyme in sorted(a_set & b_set):
            if label == "inhibits the metabolism of":
                notes.append(
                    f"{drug_b} inhibits {enzyme}, which metabolises {drug_a} - "
                    f"raised {drug_a} exposure (pharmacokinetic mechanism)."
                )
            elif label == "levels are raised by":
                notes.append(
                    f"{drug_a} inhibits {enzyme}, which metabolises {drug_b} - "
                    f"raised {drug_b} exposure (pharmacokinetic mechanism)."
                )
            else:
                notes.append(
                    f"Both drugs are {enzyme} substrates - metabolic "
                    "competition is plausible."
                )
    return notes


def explain_pair(graph: DrugGraph, model: MedisafeGNN, X: torch.Tensor,
                 A_norm: torch.Tensor, drug_a: str, drug_b: str) -> Dict:
    """Assemble a full explanation record for one drug pair."""
    i, j = graph.name_to_idx[drug_a], graph.name_to_idx[drug_b]
    return {
        "attention_paths": _attention_paths(graph, model, X, A_norm, i, j),
        "pharmacology": _shared_annotations(graph, drug_a, drug_b),
    }
