"""Global configuration for MEDISAFE-GNN."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REAL_DATA_DIR = os.path.join(DATA_DIR, "real")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

DRUGS_SAMPLE_CSV = os.path.join(DATA_DIR, "drugs_sample.csv")
DDI_SAMPLE_CSV = os.path.join(DATA_DIR, "ddi_sample.csv")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "medisafe_gnn.pt")

# ---------------------------------------------------------------------------
# External datasets (used by scripts/download_real_data.py)
# ---------------------------------------------------------------------------
TWOSIDES_URL = (
    "https://tatonettilab-resources.s3.us-west-1.amazonaws.com/nsides/TWOSIDES.csv.gz"
)
OFFSIDES_URL = (
    "https://tatonettilab-resources.s3.us-west-1.amazonaws.com/nsides/OFFSIDES.csv.gz"
)

# ---------------------------------------------------------------------------
# Model / training hyper-parameters
# ---------------------------------------------------------------------------
CONFIG = {
    # feature extraction
    "feature_dim": 48,
    # GNN encoder
    "hidden_dim": 64,
    "embedding_dim": 64,
    "num_gnn_layers": 2,
    "num_attention_heads": 4,
    "gnn_type": "gat",          # "gat" or "gcn"
    "dropout": 0.3,
    # link-prediction decoder
    "decoder_hidden": 64,
    # adverse-effect multi-label head
    "max_adverse_effects": 24,
    # training
    "learning_rate": 1e-2,
    "weight_decay": 5e-4,
    "epochs": 150,
    "negative_sampling_ratio": 1.0,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "random_seed": 42,
    # inference thresholds
    "interaction_threshold": 0.5,
    "adverse_effect_threshold": 0.35,
}

for _d in (DATA_DIR, REAL_DATA_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)
