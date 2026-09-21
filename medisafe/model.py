"""MEDISAFE-GNN model: drug encoder + multi-task interaction decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GCNLayer, GATLayer


class DrugEncoder(nn.Module):
    """Stacked message-passing encoder producing one embedding per drug."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, heads: int = 4,
                 gnn_type: str = "gat", dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        dims_in = [in_dim] + [hidden_dim] * (num_layers - 1)
        dims_out = [hidden_dim] * (num_layers - 1) + [out_dim]
        for idx, (d_in, d_out) in enumerate(zip(dims_in, dims_out)):
            if gnn_type == "gat":
                heads_use = heads if idx < num_layers - 1 else 1
                d_out_use = max(d_out // heads_use, 8) if idx < num_layers - 1 else d_out
                self.layers.append(
                    GATLayer(d_in, d_out_use, heads=heads_use, dropout=dropout)
                )
            else:
                self.layers.append(GCNLayer(d_in, d_out))

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = X
        for layer in self.layers:
            h = layer(h, A)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class InteractionDecoder(nn.Module):
    """Scores a candidate drug pair from its two node embeddings."""

    def __init__(self, emb_dim: int, hidden: int, n_types: int):
        super().__init__()
        # pair representation: concat + Hadamard product + abs difference
        in_dim = emb_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.interaction_head = nn.Linear(hidden, 1)   # P(interaction)
        self.type_head = nn.Linear(hidden, n_types)    # interaction category

    def forward(self, emb: torch.Tensor, edge_index: torch.Tensor):
        zi, zj = emb[edge_index[:, 0]], emb[edge_index[:, 1]]
        pair = torch.cat([zi, zj, zi * zj], dim=-1)
        h = self.mlp(pair)
        p_interaction = torch.sigmoid(self.interaction_head(h)).squeeze(-1)
        type_logits = self.type_head(h)
        return p_interaction, type_logits


class AdverseEffectHead(nn.Module):
    """Multi-label head predicting adverse effects for an interacting pair."""

    def __init__(self, emb_dim: int, hidden: int, n_effects: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_effects),
        )

    def forward(self, emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        zi, zj = emb[edge_index[:, 0]], emb[edge_index[:, 1]]
        pair = torch.cat([zi, zj, zi * zj], dim=-1)
        return self.mlp(pair)


class MedisafeGNN(nn.Module):
    """Full model: encoder + interaction decoder + adverse-effect head."""

    def __init__(self, feature_dim: int, hidden_dim: int, embedding_dim: int,
                 num_gnn_layers: int = 2, heads: int = 4, gnn_type: str = "gat",
                 dropout: float = 0.3, decoder_hidden: int = 64,
                 n_interaction_types: int = 4, n_adverse_effects: int = 24):
        super().__init__()
        self.encoder = DrugEncoder(
            in_dim=feature_dim, hidden_dim=hidden_dim, out_dim=embedding_dim,
            num_layers=num_gnn_layers, heads=heads, gnn_type=gnn_type,
            dropout=dropout,
        )
        self.decoder = InteractionDecoder(
            embedding_dim, decoder_hidden, n_interaction_types
        )
        self.adverse = AdverseEffectHead(
            embedding_dim, decoder_hidden, n_adverse_effects
        )

    def forward(self, X: torch.Tensor, A: torch.Tensor, edge_index: torch.Tensor):
        emb = self.encoder(X, A)
        p, t_logits = self.decoder(emb, edge_index)
        a_logits = self.adverse(emb, edge_index)
        return emb, p, t_logits, a_logits