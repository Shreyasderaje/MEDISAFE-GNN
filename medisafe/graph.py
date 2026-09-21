"""Drug interaction graph construction and edge splitting.

The drug graph is a homogeneous graph whose nodes are drugs (with molecular
features as node attributes) and whose edges are known / potential drug-drug
interactions.  Negative edges (non-interacting pairs) are sampled to balance
the training objective, following standard link-prediction practice.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .data import INTERACTION_TYPES, extract_adverse_effect_vocabulary


class DrugGraph:
    """Container for the drug interaction graph."""

    def __init__(self, drugs: pd.DataFrame, ddi: pd.DataFrame, features: np.ndarray):
        self.drug_names: List[str] = drugs["name"].tolist()
        self.name_to_idx: Dict[str, int] = {
            n: i for i, n in enumerate(self.drug_names)
        }
        self.features = np.asarray(features, dtype=np.float32)
        self.drugs = drugs.reset_index(drop=True)
        self.ddi = ddi.reset_index(drop=True)
        self.n_drugs = len(self.drug_names)
        self.adverse_vocab, self.adverse_counts = extract_adverse_effect_vocabulary(
            ddi, top_k=config.CONFIG["max_adverse_effects"]
        )
        self._build_edges()

    # ------------------------------------------------------------------ #
    def _build_edges(self) -> None:
        src, dst, types, effects = [], [], [], []
        for _, row in self.ddi.iterrows():
            i = self.name_to_idx[row["drug_1"]]
            j = self.name_to_idx[row["drug_2"]]
            src.append(i)
            dst.append(j)
            types.append(
                INTERACTION_TYPES.index(row["interaction_type"])
                if row["interaction_type"] in INTERACTION_TYPES else 1
            )
            effects.append(str(row["adverse_effects"]))
        self.edge_index = np.vstack([src, dst]).T          # (E, 2)
        self.edge_type = np.asarray(types, dtype=np.int64)  # (E,)

        # multi-label matrix of adverse effects per edge
        lab = np.zeros((len(src), len(self.adverse_vocab)), dtype=np.float32)
        vocab_idx = {name: k for k, name in enumerate(self.adverse_vocab)}
        for e, eff_str in enumerate(effects):
            for eff in str(eff_str).split(";"):
                k = vocab_idx.get(eff.strip().lower())
                if k is not None:
                    lab[e, k] = 1.0
        self.edge_adverse = lab

    # ------------------------------------------------------------------ #
    def adverse_labels_for(self, drug_a: str, drug_b: str) -> np.ndarray:
        """Known adverse-effect labels for a specific pair (or zeros)."""
        for e in range(len(self.edge_index)):
            i, j = self.edge_index[e]
            if {self.drug_names[i], self.drug_names[j]} == {drug_a, drug_b}:
                return self.edge_adverse[e]
        return np.zeros(len(self.adverse_vocab), dtype=np.float32)

    def adjacency(self, weighted: bool = False) -> np.ndarray:
        """Dense adjacency matrix with self-loops (symmetrised DDI graph)."""
        A = np.zeros((self.n_drugs, self.n_drugs), dtype=np.float32)
        for i, j in self.edge_index:
            A[i, j] = 1.0
            A[j, i] = 1.0
        A += np.eye(self.n_drugs, dtype=np.float32)
        return A

    def sample_negative_edges(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample random non-interacting drug pairs (negative edges)."""
        positives = {
            frozenset((i, j)) for i, j in self.edge_index
        }
        negs = []
        while len(negs) < n:
            i = int(rng.integers(0, self.n_drugs))
            j = int(rng.integers(0, self.n_drugs))
            if i == j or frozenset((i, j)) in positives:
                continue
            negs.append((i, j))
            positives.add(frozenset((i, j)))
        return np.asarray(negs, dtype=np.int64)

    def split_edges(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random train / val / test split of positive edges."""
        seed = seed if seed is not None else config.CONFIG["random_seed"]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.edge_index))
        n_val = int(len(perm) * config.CONFIG["val_fraction"])
        n_test = int(len(perm) * config.CONFIG["test_fraction"])
        test_idx = perm[:n_test]
        val_idx = perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]
        return (
            self.edge_index[train_idx],
            self.edge_index[val_idx],
            self.edge_index[test_idx],
        )
