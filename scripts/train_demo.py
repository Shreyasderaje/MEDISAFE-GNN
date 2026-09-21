#!/usr/bin/env python
"""Train the MEDISAFE-GNN demo model and save a checkpoint.

Usage:
    python scripts/train_demo.py [--epochs 150] [--data-dir data/real]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medisafe import config
from medisafe.data import load_dataset, load_twosides
from medisafe.graph import DrugGraph
from medisafe.train import save_checkpoint, train


def main() -> int:
    ap = argparse.ArgumentParser(description="Train MEDISAFE-GNN")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--real-data", action="store_true",
                    help="use TWOSIDES-derived data in data/real if available")
    args = ap.parse_args()

    if args.real_data:
        twosides = os.path.join(config.REAL_DATA_DIR, "TWOSIDES_medisafe.csv")
        if os.path.exists(twosides):
            print(f"Loading real data: {twosides}")
            ddi = load_twosides(twosides)
            drugs, _, feats = load_dataset()
            known = set(drugs["name"])
            ddi = ddi[ddi["drug_1"].isin(known) & ddi["drug_2"].isin(known)]
            graph = DrugGraph(drugs, ddi, feats)
        else:
            print("Real data not found - falling back to demo dataset.")
            drugs, ddi, feats = load_dataset()
            graph = DrugGraph(drugs, ddi, feats)
    else:
        drugs, ddi, feats = load_dataset()
        graph = DrugGraph(drugs, ddi, feats)

    print(f"Graph: {graph.n_drugs} drugs, {len(graph.edge_index)} interactions, "
          f"{len(graph.adverse_vocab)} adverse-effect labels")
    result = train(graph, epochs=args.epochs)
    print("Metrics:", result["metrics"])
    path = save_checkpoint(result, graph)
    print(f"Checkpoint saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())