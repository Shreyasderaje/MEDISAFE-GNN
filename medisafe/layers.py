"""Graph neural network layers implemented in pure PyTorch.

The message-passing layers support dense adjacency input, which is ideal for
the small-to-medium drug graphs (hundreds to a few thousand nodes) used by
MEDISAFE-GNN and avoids the platform-specific install of PyTorch Geometric.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Classic spectral-convolution layer (Kipf & Welling, 2017)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # A_hat must already be symmetric + self-loops + normalised.
        return self.linear(A_hat @ X)


class GATLayer(nn.Module):
    """Graph attention layer (Velickovic et al., 2018) with multi-head support.

    Attention coefficients are exposed in ``self.last_attention`` so the
    explainer can highlight which neighbourhood messages supported a
    prediction.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.dropout = dropout
        self.last_attention: torch.Tensor | None = None
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """X: (N, in_dim);  A: dense (N, N) 0/1 adjacency incl. self-loops."""
        N = X.size(0)
        H = self.W(X).view(N, self.heads, self.out_dim)          # (N, heads, D)

        e_src = (H * self.a_src).sum(-1)                         # (N, heads)
        e_dst = (H * self.a_dst).sum(-1)                         # (N, heads)
        # logits(i->j) = a_src.h + a_dst.h_j
        logits = e_src.unsqueeze(1) + e_dst.unsqueeze(0)         # (N, N, heads)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        mask = (A > 0).unsqueeze(-1)                             # (N, N, 1)
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = F.softmax(logits, dim=1)                          # normalise over j
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        self.last_attention = attn.detach()
        out = torch.einsum("ijh,jhd->ihd", attn, H)              # aggregate
        return out.reshape(N, self.heads * self.out_dim)
