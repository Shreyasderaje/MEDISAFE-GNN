"""Lightweight SMILES feature extraction (RDKit-free).

MEDISAFE-GNN intentionally avoids heavy optional dependencies such as RDKit.
This module tokenises a SMILES string and computes a compact physico-chemical
"fingerprint" describing the drug's composition (atom counts, bond orders,
rings, aromaticity, charge, functional-group heuristics).  These features are
the input node attributes of the drug interaction graph.
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

# Two-character elements must be matched before single-character ones.
_TWO_CHAR = ("Cl", "Br", "Se", "Si", "Li", "Na", "Mg", "Sn", "Zn", "As", "Fe")
_SINGLE_CHAR = ("B", "C", "N", "O", "S", "P", "F", "I")


def tokenize_smiles(smiles: str) -> List[str]:
    """Split a SMILES string into atoms / symbols / bracket tokens."""
    tokens: List[str] = []
    i = 0
    s = smiles.strip()
    while i < len(s):
        ch = s[i]
        if ch == "[":
            j = s.find("]", i)
            if j == -1:
                break
            tokens.append(s[i : j + 1])
            i = j + 1
            continue
        two = s[i : i + 2]
        if two in _TWO_CHAR:
            tokens.append(two)
            i += 2
            continue
        if ch in _SINGLE_CHAR or ch in "=#%()@+-./\\123456789":
            tokens.append(ch)
            i += 1
            continue
        if ch.islower():  # aromatic atoms b c n o s p
            tokens.append(ch)
            i += 1
            continue
        i += 1
    return tokens


def molecular_weight(smiles: str) -> float:
    """Approximate molecular weight from a SMILES string (g/mol)."""
    weights = {
        "B": 10.8, "C": 12.01, "N": 14.01, "O": 16.0, "S": 32.06, "P": 30.97,
        "F": 19.0, "Cl": 35.45, "Br": 79.9, "I": 126.9, "Si": 28.09,
        "Se": 78.96, "Na": 22.99, "K": 39.1, "Li": 6.94, "Mg": 24.3,
        "Ca": 40.08, "Zn": 65.4, "Fe": 55.85,
    }
    total = 0.0
    for tok in tokenize_smiles(smiles):
        if tok.startswith("["):
            inner = tok.strip("[]")
            m = re.match(r"[A-Z][a-z]?", inner)
            if m:
                total += weights.get(m.group(0), 0.0)
            continue
        total += weights.get(tok, 0.0)
    return total

def smiles_features(smiles: str, dim: int = 48) -> np.ndarray:
    """Compute a fixed-length feature vector for one drug.

    Layout (first 40 dims are counts / heuristics, remaining dims reserved):
      0-9   atom counts  (C, N, O, S, P, F, Cl, Br, I, other)
      10    aromatic atom count
      11    double bonds
      12    triple bonds
      13    rings
      14    branches
      15    molecular weight (scaled /100)
      16    H-bond donors heuristic
      17    H-bond acceptors heuristic
      18    charged atoms
      19    chiral centers
      20-39 fragment / structural heuristics
      40-47 reserved zeros
    """
    tokens = tokenize_smiles(smiles)
    s = smiles.strip()

    atoms = {
        "C": 0, "N": 0, "O": 0, "S": 0, "P": 0, "F": 0, "Cl": 0, "Br": 0,
        "I": 0, "other": 0,
    }
    aromatic = 0
    double_bonds = s.count("=")
    triple_bonds = s.count("#")
    rings = len({c for c in s if c.isdigit()})
    branches = s.count("(")
    charged = 0
    chiral = s.count("@")

    for tok in tokens:
        if tok.startswith("["):
            inner = tok.strip("[]")
            if "+" in inner or "-" in inner:
                charged += 1
            m = re.match(r"[A-Z][a-z]?", inner)
            el = m.group(0) if m else "other"
        else:
            el = tok
        if tok.islower() and len(tok) == 1:
            aromatic += 1
            el = tok.upper()
        if el in atoms:
            atoms[el] += 1
        else:
            atoms["other"] += 1

    n_total = sum(atoms.values()) or 1
    n_o = atoms["O"]
    n_n = atoms["N"]

    feats = np.zeros(dim, dtype=np.float32)
    base = [
        atoms["C"], atoms["N"], atoms["O"], atoms["S"], atoms["P"], atoms["F"],
        atoms["Cl"], atoms["Br"], atoms["I"], atoms["other"],
        aromatic, double_bonds, triple_bonds, rings, branches,
        molecular_weight(smiles) / 100.0,
        max(n_n + n_o - double_bonds * 0.5, 0.0),    # donors heuristic
        float(n_n + n_o),                             # acceptors heuristic
        charged, chiral,
    ]
    feats[: len(base)] = np.asarray(base, dtype=np.float32)

    frag = [
        1.0 if "C(=O)O" in s else 0.0,                       # carboxylic acid
        1.0 if "C(=O)N" in s else 0.0,                       # amide
        1.0 if "C(=O)OC" in s.replace("[", "").replace("]", "") else 0.0,  # ester
        1.0 if "=O" in s else 0.0,                           # any carbonyl
        1.0 if atoms["S"] > 0 else 0.0,                      # sulfur
        1.0 if "S(=O)" in s else 0.0,                        # sulfone/sulfonamide
        1.0 if "N+" in s else 0.0,                           # quaternary/nitro
        1.0 if "c1ccc" in s or "C1=CC=CC=C1" in s else 0.0,  # benzene
        1.0 if s.count("c1") + s.count("C1") >= 2 else 0.0,  # two+ rings
        1.0 if atoms["I"] > 0 else 0.0,                      # iodine
        1.0 if "N=C" in s or "C=N" in s else 0.0,            # imine
        1.0 if "OC" in s and "=O" not in s else 0.0,         # alcohol/ether
        1.0 if "c2" in s and "c1" in s else 0.0,             # poly-aromatic
        1.0 if n_o >= 3 else 0.0,                            # poly-oxygenated
        1.0 if n_n >= 3 else 0.0,                            # poly-nitrogenated
        1.0 if (n_n + n_o) / n_total > 0.25 else 0.0,        # heteroatom rich
        1.0 if atoms["C"] / n_total > 0.5 else 0.0,          # lipophilic core
        1.0 if rings >= 2 else 0.0,
        1.0 if len(tokens) > 40 else 0.0,                    # large molecule
        1.0 if atoms["F"] + atoms["Cl"] + atoms["Br"] > 0 else 0.0,  # halogen
    ]
    feats[20 : 20 + len(frag)] = np.asarray(frag, dtype=np.float32)

    feats[:10] = feats[:10] / 10.0
    feats[10:15] = feats[10:15] / 5.0
    return feats


def batch_smiles_features(smiles_list: List[str], dim: int = 48) -> np.ndarray:
    """Return an (N, dim) matrix of features for a list of SMILES."""
    if not smiles_list:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack([smiles_features(s, dim) for s in smiles_list])

