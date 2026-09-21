"""Training pipeline for MEDISAFE-GNN (link prediction + multi-task heads)."""
from __future__ import annotations

import json
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .data import INTERACTION_TYPES
from .graph import DrugGraph
from .model import MedisafeGNN


def normalise_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric normalisation D^-1/2 (A + I) D^-1/2 for dense adjacency."""
    A = A + np.eye(A.shape[0], dtype=np.float32)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    Dm = np.diag(d_inv_sqrt)
    return (Dm @ A @ Dm).astype(np.float32)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pure-numpy ROC AUC (rank based, tie-averaged)."""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        mask = s == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    pos, neg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pure-numpy average precision."""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y)) + 1)
    return float(np.sum(precision * y_sorted) / max(y.sum(), 1))


def train(graph: DrugGraph, epochs: int = None, verbose: bool = True) -> Dict:
    """Train the full MEDISAFE-GNN model on the given DrugGraph."""
    cfg = config.CONFIG
    epochs = epochs or cfg["epochs"]
    torch.manual_seed(cfg["random_seed"])
    rng = np.random.default_rng(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(graph.features, dtype=torch.float32, device=device)
    A = torch.tensor(
        normalise_adjacency(graph.adjacency()), dtype=torch.float32, device=device
    )

    train_e, val_e, test_e = graph.split_edges()
    tr_idx = _match_indices(graph, train_e)
    va_idx = _match_indices(graph, val_e)
    te_idx = _match_indices(graph, test_e)

    y_type_tr = torch.tensor(graph.edge_type[tr_idx], dtype=torch.long, device=device)
    y_adv_tr = torch.tensor(graph.edge_adverse[tr_idx], dtype=torch.float32, device=device)

    model = MedisafeGNN(
        feature_dim=X.size(1),
        hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"],
        num_gnn_layers=cfg["num_gnn_layers"],
        heads=cfg["num_attention_heads"],
        gnn_type=cfg["gnn_type"],
        dropout=cfg["dropout"],
        decoder_hidden=cfg["decoder_hidden"],
        n_interaction_types=len(INTERACTION_TYPES),
        n_adverse_effects=len(graph.adverse_vocab),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    history = {"loss": [], "val_auc": [], "val_ap": []}
    best_val, best_state, patience, bad = 0.0, None, 30, 0
    t0 = time.time()

    precision = tp / (np.arange(len(y)) + 1)
    return float(np.sum(precision * y_sorted) / max(y.sum(), 1))


def train(graph: DrugGraph, epochs: int = None, verbose: bool = True) -> Dict:
    """Train the full MEDISAFE-GNN model on the given DrugGraph."""
    cfg = config.CONFIG
    epochs = epochs or cfg["epochs"]
    torch.manual_seed(cfg["random_seed"])
    rng = np.random.default_rng(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(graph.features, dtype=torch.float32, device=device)
    A = torch.tensor(
        normalise_adjacency(graph.adjacency()), dtype=torch.float32, device=device
    )

    train_e, val_e, test_e = graph.split_edges()
    tr_idx = _match_indices(graph, train_e)
    te_idx = _match_indices(graph, test_e)
    y_type_tr = torch.tensor(graph.edge_type[tr_idx], dtype=torch.long, device=device)
    y_adv_tr = torch.tensor(graph.edge_adverse[tr_idx], dtype=torch.float32, device=device)

    model = MedisafeGNN(
        feature_dim=X.size(1), hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"], num_gnn_layers=cfg["num_gnn_layers"],
        heads=cfg["num_attention_heads"], gnn_type=cfg["gnn_type"],
        dropout=cfg["dropout"], decoder_hidden=cfg["decoder_hidden"],
        n_interaction_types=len(INTERACTION_TYPES),
        n_adverse_effects=len(graph.adverse_vocab),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    history = {"loss": [], "val_auc": [], "val_ap": []}
    best_val, best_state, patience, bad = 0.0, None, 30, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        neg_e = graph.sample_negative_edges(
            int(len(train_e) * cfg["negative_sampling_ratio"]), rng
        )
        neg_t = torch.tensor(neg_e, dtype=torch.long, device=device)
        edges = torch.cat(
            [torch.tensor(train_e, dtype=torch.long, device=device), neg_t], dim=0
        )
        y = torch.cat(
            [torch.ones(len(train_e)), torch.zeros(len(neg_e))]
        ).to(device)

        emb, p, t_logits, a_logits = model(X, A, edges)
        loss_link = F.binary_cross_entropy(p, y)
        loss_type = F.cross_entropy(t_logits[: len(train_e)], y_type_tr)
        loss_adv = F.binary_cross_entropy_with_logits(
            a_logits[: len(train_e)], y_adv_tr
        )
        loss = loss_link + 0.5 * loss_type + 0.5 * loss_adv
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z = model.encoder(X, A)
            p_val = model.decoder(z, torch.tensor(val_e, dtype=torch.long, device=device))[0]
            val_neg = graph.sample_negative_edges(len(val_e), rng)
            p_neg = model.decoder(
                z, torch.tensor(val_neg, dtype=torch.long, device=device)
            )[0]
            y_val = np.concatenate([np.ones(len(val_e)), np.zeros(len(p_neg))])
            s_val = np.concatenate([p_val.cpu().numpy(), p_neg.cpu().numpy()])
            v_auc, v_ap = roc_auc(y_val, s_val), average_precision(y_val, s_val)

        history["loss"].append(float(loss.item()))
        history["val_auc"].append(v_auc)
        history["val_ap"].append(v_ap)

        if v_auc > best_val:
            best_val, bad = v_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best val AUC {best_val:.4f})")
            break
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d} | loss {loss.item():.4f} "
                f"(link {loss_link.item():.3f} type {loss_type.item():.3f} "
                f"adv {loss_adv.item():.3f}) | val AUC {v_auc:.4f} AP {v_ap:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return _evaluate(model, X, A, graph, test_e, te_idx, best_val, t0, history, device)


def _evaluate(model, X, A, graph, test_e, te_idx, best_val, t0, history, device):
    model.eval()
    with torch.no_grad():
        z = model.encoder(X, A)
        p_test, t_logits = model.decoder(
            z, torch.tensor(test_e, dtype=torch.long, device=device)
        )
        rng = np.random.default_rng(config.CONFIG["random_seed"])
        test_neg = graph.sample_negative_edges(len(test_e), rng)
        p_neg = model.decoder(z, torch.tensor(test_neg, dtype=torch.long, device=device))[0]
        y_test = np.concatenate([np.ones(len(test_e)), np.zeros(len(p_neg))])
        s_test = np.concatenate([p_test.cpu().numpy(), p_neg.cpu().numpy()])
        metrics = {
            "test_auc": roc_auc(y_test, s_test),
            "test_ap": average_precision(y_test, s_test),
            "best_val_auc": best_val,
            "train_seconds": round(time.time() - t0, 1),
            "n_drugs": graph.n_drugs,
            "n_interactions": len(graph.edge_index),
            "n_adverse_effects": len(graph.adverse_vocab),
        }
        if len(test_e) and len(te_idx):
            t_pred = t_logits.argmax(dim=1).cpu().numpy()
            metrics["type_accuracy"] = float((t_pred == graph.edge_type[te_idx]).mean())
    return {"model": model, "metrics": metrics, "history": history,
            "device": str(device)}


def _match_indices(graph: DrugGraph, edges: np.ndarray) -> np.ndarray:
    """Recover original edge indices for a subset of edges (order preserved)."""
    lookup = {}
    for e, (i, j) in enumerate(graph.edge_index):
        lookup[(i, j)] = e
    out = []
    for i, j in edges:
        e = lookup.get((i, j), lookup.get((j, i)))
        if e is not None:
            out.append(e)
    return np.asarray(out, dtype=np.int64)


def save_checkpoint(result: Dict, graph: DrugGraph, path: str = None) -> str:
    path = path or config.CHECKPOINT_PATH
    payload = {
        "state_dict": result["model"].state_dict(),
        "config": config.CONFIG,
        "drug_names": graph.drug_names,
        "features": graph.features,
        "adverse_vocab": graph.adverse_vocab,
        "interaction_types": INTERACTION_TYPES,
        "metrics": result["metrics"],
        "edges": graph.edge_index,
        "edge_type": graph.edge_type,
        "edge_adverse": graph.edge_adverse,
        "ddi_records": graph.ddi.to_dict(orient="records"),
        "drug_records": graph.drugs.to_dict(orient="records"),
    }
    torch.save(payload, path)
    with open(path + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    return path
