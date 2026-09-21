#!/usr/bin/env python
"""Download the full TWOSIDES pharmacovigilance database.

TWOSIDES (Tatonetti Lab, Columbia University) contains millions of
drug-pair / adverse-event signals mined from the FDA FAERS database with
propensity-score-matched PRR statistics.

    Citation: Tatonetti NP, Ye PP, Daneshjou R, Altman RB. "Data-driven
    prediction of drug effects and interactions." Sci Transl Med 2012.

Usage:
    python scripts/download_real_data.py
"""
import gzip
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medisafe import config
from medisafe.data import load_twosides


def download(url: str, dest: str) -> None:
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main() -> int:
    os.makedirs(config.REAL_DATA_DIR, exist_ok=True)
    gz_path = os.path.join(config.REAL_DATA_DIR, "TWOSIDES.csv.gz")
    out_path = os.path.join(config.REAL_DATA_DIR, "TWOSIDES_medisafe.csv")

    if not os.path.exists(gz_path):
        try:
            download(config.TWOSIDES_URL, gz_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}\n"
                  "Download manually from "
                  "https://tatonettilab.org/resources/nsides/ and place "
                  f"TWOSIDES.csv.gz in {config.REAL_DATA_DIR}")
            return 1

    print("Converting to MEDISAFE DDI format (PRR >= 5, top conditions)...")
    ddi = load_twosides(gz_path, min_prr=5.0, top_conditions=100)
    ddi.to_csv(out_path, index=False)
    print(f"Saved {len(ddi)} interaction records to {out_path}")
    print("Re-run scripts/train_demo.py --real-data to train on this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())