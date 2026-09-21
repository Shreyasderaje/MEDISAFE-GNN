# MEDISAFE-GNN

**Graph Neural Network–Based Drug–Drug Interaction and Adverse Effect Prediction System**

> ⚠️ **Research / decision-support tool.** Predictions are statistical inferences from
> biomedical data. They are **not** medical advice and must not replace pharmacist or
> physician review. The UI intentionally never claims a combination is "safe".

---

## What it does

MEDISAFE-GNN represents drugs as nodes in a graph (with molecular features as node
attributes) and documented interactions as edges. A **Graph Attention Network (GAT)**
learns drug embeddings by message passing, and multi-task decoder heads predict, for any
drug pair:

| Output | Description |
|---|---|
| **Interaction probability** | likelihood that the pair interacts (link prediction) |
| **Interaction type** | major / moderate / minor / contraindicated |
| **Adverse effects** | multi-label probabilities (bleeding, rhabdomyolysis, serotonin syndrome, …) |
| **Confidence** | decision-uncertainty score derived from the predicted probability |
| **Supporting evidence** | GAT attention paths + shared pharmacology (ATC class, CYP enzymes) |

```
   Drug database (SMILES, ATC, CYP enzymes)
                  │
      ┌───────────┴────────────┐
      ▼                        ▼
 molecular features      interaction graph (edges = DDIs)
      └───────────┬────────────┘
                  ▼
      GAT encoder (message passing)
                  ▼
      link-prediction decoder ──► P(interaction), interaction type
                  ▼
      adverse-effect head (multi-label)
                  ▼
      attention-based explanation ──► Flask web dashboard
```

## Quick start

```bash
# 1. install dependencies (CPU PyTorch is sufficient)
pip install -r requirements.txt

# 2. train the demo model (bundled dataset, ~1 minute on CPU)
python scripts/train_demo.py

# 3. launch the dashboard
python scripts/run_app.py
# open http://127.0.0.1:5000
```

Run the test suite:

```bash
python -m pytest tests/ -q
```

## Project structure

```
MEDISAFE-GNN/
├── medisafe/
│   ├── config.py        # paths + hyper-parameters
│   ├── features.py      # RDKit-free SMILES → physico-chemical fingerprint
│   ├── data.py          # dataset loading, TWOSIDES conversion
│   ├── graph.py         # DrugGraph: edges, negative sampling, splits
│   ├── layers.py        # GCN + GAT layers in pure PyTorch
│   ├── model.py         # encoder + link decoder + adverse-effect head
│   ├── train.py         # training loop, AUC/AP metrics, checkpoints
│   ├── explain.py       # attention paths + pharmacology evidence
│   ├── predict.py       # MedisafeEngine inference API
│   ├── app.py           # Flask dashboard + REST API
│   ├── templates/       # dashboard UI
│   └── static/          # CSS + dashboard JS
├── data/
│   ├── drugs_sample.csv     # 89 drugs (SMILES, ATC class, CYP substrate/inhibitor)
│   └── ddi_sample.csv       # 137 documented interactions (type, effects, mechanism)
├── scripts/
│   ├── train_demo.py        # train + save checkpoint
│   ├── run_app.py           # launch web dashboard
│   └── download_real_data.py# fetch full TWOSIDES database
└── tests/                   # 31 unit + integration tests
```
## Methods (as used in the DDI-prediction literature)

The pipeline follows the standard graph-based DDI approach (see Abbas et al.,
*Scientific Reports* 2025, "Graph neural network-based drug-drug interaction
prediction" and references therein):

1. **Node features** — drug molecular descriptors computed from SMILES (atom
   composition, rings, aromaticity, functional groups, H-bond heuristics).
2. **Encoder** — 2-layer GAT with multi-head attention (Kipf & Welling GCN also
   supported via `CONFIG["gnn_type"] = "gcn"`).
3. **Link prediction** — pair decoder on embedding concat + Hadamard product,
   trained with **binary cross-entropy + negative sampling**.
4. **Multi-task heads** — interaction-type cross-entropy and adverse-effect
   multi-label BCE, combined loss `L = L_link + 0.5·L_type + 0.5·L_adverse`.
5. **Explainability** — GAT attention coefficients aggregated over 1-hop and
   2-hop message paths, plus shared-ATC / shared-CYP pharmacological evidence.

## Scaling to the full datasets

The bundled dataset is a curated demo subset (real drugs and real documented
interactions, compiled from public reference sources — FDA labeling and published
pharmacology literature; SMILES strings are demo-grade, verify before production use).

```bash
python scripts/download_real_data.py   # TWOSIDES (~4.6M FAERS-derived pair/effect records)
python scripts/train_demo.py --real-data
```

- **TWOSIDES** — Tatonetti NP, Ye PP, Daneshjou R, Altman RB. *Data-driven prediction
  of drug effects and interactions.* Sci Transl Med. 2012. https://tatonettilab.org/resources/nsides/
- **DrugBank** — Wishart DS et al. Nucleic Acids Res. 2018 (requires license; convert
  its interaction tables to the `data/*.csv` schema yourself).

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | dashboard |
| `/api/drugs?q=war` | GET | drug name autocomplete |
| `/api/predict` | POST | `{"drugs": ["warfarin", "aspirin"]}` → pair predictions |
| `/api/model` | GET | model metadata + test metrics |

## Limitations

- Small curated knowledge base → the model only sees 75 drugs by default.
- Predictions on drug pairs absent from the knowledge base are extrapolations from
  graph structure and molecular similarity, not validated clinical findings.
- TWOSIDES signals are pharmacovigilance correlations (reporting-ratio based),
  not causal proof.

## License

MIT — see [LICENSE](LICENSE).
