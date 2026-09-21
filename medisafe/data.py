"""Dataset loading utilities.

MEDISAFE-GNN ships with a curated demo dataset of real, well documented
drug-drug interactions compiled from public reference sources (public DrugBank
interaction descriptions, FDA labeling and published pharmacology literature).
It can also ingest the full TWOSIDES database (Tatonetti Lab) when the user
downloads it via ``scripts/download_real_data.py``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .features import batch_smiles_features

REQUIRED_DRUG_COLUMNS = [
    "name", "smiles", "atc_class", "cyp_substrate", "cyp_inhibitor",
]
REQUIRED_DDI_COLUMNS = [
    "drug_1", "drug_2", "interaction_type", "adverse_effects", "mechanism",
]
INTERACTION_TYPES = ["major", "moderate", "minor", "contraindicated"]


def load_drugs(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the drug table (name, smiles, atc_class, cyp flags)."""
    path = csv_path or config.DRUGS_SAMPLE_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Drug table not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in ["name", "smiles"] if c not in df.columns]
    if missing:
        raise ValueError(f"Drug table missing columns: {missing}")
    df["name"] = df["name"].str.strip()
    return df


def load_ddi(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the drug-drug interaction table."""
    path = csv_path or config.DDI_SAMPLE_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"DDI table not found: {path}")
    df = pd.read_csv(path)
    df["drug_1"] = df["drug_1"].str.strip()
    df["drug_2"] = df["drug_2"].str.strip()
    df["interaction_type"] = (
        df["interaction_type"].str.strip().str.lower()
    )
    df["adverse_effects"] = df["adverse_effects"].fillna("")
    df["mechanism"] = df["mechanism"].fillna("")
    return df


def load_dataset(drug_csv: Optional[str] = None, ddi_csv: Optional[str] = None):
    """Load and validate drugs + interactions, returning (drugs, ddi, features)."""
    drugs = load_drugs(drug_csv)
    ddi = load_ddi(ddi_csv)

    drug_names = set(drugs["name"])
    unknown = sorted((set(ddi["drug_1"]) | set(ddi["drug_2"])) - drug_names)
    if unknown:
        raise ValueError(
            f"DDI table references drugs missing from drug table: {unknown}"
        )

    feats = batch_smiles_features(
        drugs["smiles"].tolist(), dim=config.CONFIG["feature_dim"]
    )
    return drugs, ddi, feats


def extract_adverse_effect_vocabulary(ddi: pd.DataFrame, top_k: int = 24):
    """Build the adverse-effect label vocabulary from the DDI table.

    Adverse effects are stored as semicolon separated strings, e.g.
    ``"bleeding;gastrointestinal hemorrhage"``.
    """
    counts: Dict[str, int] = {}
    for effects in ddi["adverse_effects"]:
        for eff in str(effects).split(";"):
            eff = eff.strip().lower()
            if eff:
                counts[eff] = counts.get(eff, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = [name for name, _ in ranked[:top_k]]
    return vocab, counts


def load_twosides(csv_gz_path: str, min_prr: float = 5.0,
                  top_conditions: int = 100) -> pd.DataFrame:
    """Convert the full TWOSIDES database into the MEDISAFE DDI format.

    Only statistically significant signals (default PRR >= 5) for the most
    frequently reported conditions are kept, to make the data tractable on
    consumer hardware.
    """
    df = pd.read_csv(csv_gz_path)
    col_d1 = "drug_1_concept_name"
    col_d2 = "drug_2_concept_name"
    col_cond = "condition_concept_name"
    if col_cond not in df.columns:  # older releases use different casing
        col_cond = [c for c in df.columns if "condition" in c and "name" in c][0]

    df = df[df["PRR"] >= min_prr]
    top = df[col_cond].value_counts().head(top_conditions).index
    df = df[df[col_cond].isin(top)]

    agg = (
        df.groupby([col_d1, col_d2, col_cond])["PRR"].max().reset_index()
    )
    out = agg.rename(columns={
        col_d1: "drug_1", col_d2: "drug_2", col_cond: "adverse_effects",
        "PRR": "prr",
    })
    out["interaction_type"] = np.where(out["prr"] >= 15, "major", "moderate")
    out["mechanism"] = "pharmacovigilance signal (TWOSIDES, PRR>= %.1f)" % min_prr
    return out
