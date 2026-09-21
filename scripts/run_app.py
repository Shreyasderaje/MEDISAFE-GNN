#!/usr/bin/env python
"""Launch the MEDISAFE-GNN web dashboard."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medisafe.app import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    print(f"Starting MEDISAFE-GNN dashboard at http://{args.host}:{args.port}")
    run(host=args.host, port=args.port, debug=False)