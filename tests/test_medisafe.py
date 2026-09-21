"""Unit tests for MEDISAFE-GNN core components."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medisafe import config
from medisafe.data import extract_adverse_effect_vocabulary, load_dataset
from medisafe.features import molecular_weight, smiles_features, tokenize_smiles
from medisafe.graph import DrugGraph
from medisafe.layers import GATLayer, GCNLayer
from medisafe.model import MedisafeGNN
from medisafe.train import average_precision, normalise_adjacency, roc_auc


class TestFeatures:
    def test_tokenizer_elements(self):
        toks = tokenize_smiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert "C" in toks and "O" in toks and "c" in toks
        assert "(" in toks and "=" in toks

    def test_molecular_weight_ordering(self):
        assert molecular_weight("CC(=O)OC1=CC=CC=C1C(=O)O") < \
            molecular_weight("CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O")

    def test_feature_vector_shape_and_norm(self):
        v = smiles_features("CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3C1")
        assert v.shape == (config.CONFIG["feature_dim"],)
        assert np.isfinite(v).all()


class TestLayers:
    def test_gcn_shape(self):
        layer = GCNLayer(16, 8)
        X = torch.randn(5, 16)
        A = torch.eye(5) + torch.ones(5, 5)
        assert layer(X, A).shape == (5, 8)

    def test_gat_shape_and_attention(self):
        # layer outputs heads * out_dim per node (multi-head concat)
        layer = GATLayer(16, 4, heads=3)
        X = torch.randn(6, 16)
        A = torch.eye(6)
        A[0, 1] = A[1, 0] = 1.0
        y = layer(X, A)
        assert y.shape == (6, 12)

    def test_gat_masks_non_neighbours(self):
        layer = GATLayer(8, 8, heads=2)
        X = torch.randn(4, 8)
        A = torch.eye(4)  # only self-loops
        layer(X, A)
        attn = layer.last_attention
        assert torch.all(attn[0, 1:] < 1e-6)


@pytest.fixture(scope="module")
def dataset():
    drugs, ddi, feats = load_dataset()
    return drugs, ddi, feats


class TestData:
    def test_dataset_loads(self, dataset):
        drugs, ddi, feats = dataset
        assert len(drugs) > 50
        assert len(ddi) > 100
        assert feats.shape == (len(drugs), config.CONFIG["feature_dim"])

    def test_all_ddi_drugs_known(self, dataset):
        drugs, ddi, _ = dataset
        unknown = (set(ddi["drug_1"]) | set(ddi["drug_2"])) - set(drugs["name"])
        assert not unknown, unknown

    def test_interaction_types_valid(self, dataset):
        _, ddi, _ = dataset
        assert set(ddi["interaction_type"]).issubset(
            {"major", "moderate", "minor", "contraindicated"})

    def test_adverse_vocab(self, dataset):
        _, ddi, _ = dataset
        vocab, counts = extract_adverse_effect_vocabulary(ddi, top_k=24)
        assert 0 < len(vocab) <= 24
        assert counts["bleeding"] >= 1


class TestGraph:
    def test_build(self, dataset):
        drugs, ddi, feats = dataset
        g = DrugGraph(drugs, ddi, feats)
        assert g.n_drugs == len(drugs)
        assert len(g.edge_index) == len(ddi)
        assert g.edge_adverse.shape[1] == len(g.adverse_vocab)
        assert (g.edge_index < g.n_drugs).all()
        A = g.adjacency()
        assert (A == A.T).all()
        assert (np.diag(A) == 1).all()

    def test_negative_sampling(self, dataset):
        drugs, ddi, feats = dataset
        g = DrugGraph(drugs, ddi, feats)
        rng = np.random.default_rng(0)
        negs = g.sample_negative_edges(50, rng)
        assert negs.shape == (50, 2)
        pos = {frozenset((int(i), int(j))) for i, j in g.edge_index}
        for i, j in negs:
            assert frozenset((int(i), int(j))) not in pos

    def test_split(self, dataset):
        drugs, ddi, feats = dataset
        g = DrugGraph(drugs, ddi, feats)
        tr, va, te = g.split_edges()
        assert len(tr) + len(va) + len(te) == len(g.edge_index)


class TestMetrics:
    def test_roc_auc_perfect(self):
        assert roc_auc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.0

    def test_roc_auc_inverted(self):
        assert roc_auc(np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1])) == 0.0

    def test_average_precision_perfect(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        assert abs(average_precision(y, s) - 1.0) < 1e-9

    def test_normalised_adjacency(self):
        A = np.array([[0, 1], [1, 0]], dtype=np.float32)
        assert np.allclose(normalise_adjacency(A).sum(axis=1), 1.0, atol=1e-5)


class TestModel:
    def test_forward(self, dataset):
        drugs, ddi, feats = dataset
        g = DrugGraph(drugs, ddi, feats)
        model = MedisafeGNN(
            feature_dim=feats.shape[1], hidden_dim=16, embedding_dim=16,
            num_gnn_layers=2, heads=2, decoder_hidden=16,
            n_adverse_effects=len(g.adverse_vocab),
        )
        X = torch.tensor(feats, dtype=torch.float32)
        A = torch.tensor(normalise_adjacency(g.adjacency()), dtype=torch.float32)
        edges = torch.tensor(g.edge_index[:8], dtype=torch.long)
        emb, p, t_logits, a_logits = model(X, A, edges)
        assert emb.shape == (g.n_drugs, 16)
        assert p.shape == (8,)
        assert ((0 <= p) & (p <= 1)).all()
        assert t_logits.shape == (8, 4)
        assert a_logits.shape == (8, len(g.adverse_vocab))
