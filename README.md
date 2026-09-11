# Illicit Bitcoin Detection

Graph-based illicit transaction detection on the Elliptic Bitcoin dataset, using GraphSAGE for
transductive node classification. Part of a broader portfolio demonstrating production-grade
classification and anomaly-detection systems across heterogeneous data types (time-series,
tabular, graph-structured, and image data).

Full methodology, results, and honest limitations disclosure: **[final_report.md](final_report.md)**

## Problem

Given a directed graph of ~204K Bitcoin transactions (~468K undirected edges after
preprocessing), each transaction node carries 165 anonymized features. Roughly 2% of nodes are
confirmed illicit, ~21% confirmed licit, and the remaining ~77% are unlabeled. The task is
transductive binary node classification (illicit vs. licit), evaluated under a temporal
train/validation/test split that simulates real deployment drift rather than a random split.

## Headline result

Sealed test evaluation (time steps 41-49, evaluated exactly once): **illicit-class F1 = 0.5553**
(precision 0.7793, recall 0.4313). This is reported alongside, not in place of, the validation
result used for model selection (F1 0.8182); the gap between the two and its most likely cause
(temporal class-distribution shift) is discussed in the final report, since disclosing that gap
honestly is a deliberate part of this project's methodology.

## Project structure

```
graph-ml-elliptic/
├── README.md                          # this file
├── final_report.md                    # full methodology, results, limitations
├── configs/
│   └── config.yaml                    # single source of truth for all pipeline parameters
├── data/
│   ├── raw/                           # Elliptic Bitcoin CSV files (not committed)
│   └── processed/                     # node ID mapping, persisted PyG Data artifact
├── notebooks/
│   ├── 00_calibration.ipynb           # environment / PyG API smoke test (Cora dataset)
│   ├── 01_eda.ipynb                   # Phase 0: integrity checks, class/temporal analysis
│   ├── 02_graph_construction.ipynb    # Phase 1: PyG Data object construction
│   ├── 03_graphsage_baseline.ipynb    # Phase 2: fixed-hyperparameter GraphSAGE baseline
│   └── 04_ablation_study.ipynb        # Phase 3: config-driven ablation, sealed test
├── outputs/
│   ├── figures/                       # generated plots (committed for traceability)
│   └── models/                        # model checkpoints (not committed)
└── requirements.txt
```

## Prerequisites

- Python 3.12 (the exact version this project was built and tested with; `requirements.txt`
  pins package versions against it)
- Roughly 800 MB free disk space for the raw CSVs, processed graph artifact, and model
  checkpoints combined — this is an estimate, not a verified figure; check locally with
  `du -sh data/ outputs/` before relying on it
- **CPU-only by design, not auto-detected.** PyTorch Geometric's message-passing extensions
  lack verified Apple Silicon (MPS) support, so `training.device` is fixed to `cpu` in
  `configs/config.yaml` and used as-is throughout the pipeline (see `final_report.md`, Section
  6, for the full rationale). A CUDA GPU would technically run this code faster but has not
  been tested against it.

## Setup

1. Clone the repository

   ```bash
   git clone https://github.com/BerkaySarmasoglu/graph-ml-elliptic 
   cd graph-ml-elliptic
   ```

2. Create and activate a virtual environment

   ```bash
   python3 -m venv venv_graph_ml
   source venv_graph_ml/bin/activate        # macOS/Linux
   # venv_graph_ml\Scripts\activate         # Windows
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

   Core dependencies (see `requirements.txt` for pinned versions): `torch`, `torch_geometric`,
   `scipy`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `jupyter`,
   `ipywidgets`, `tqdm`.

4. Download the dataset

   Download the Elliptic Bitcoin dataset's three CSV files
   (`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`) into
   `data/raw/`.

## Running the pipeline

Notebooks are numbered and must be run in order; each one's only input is the artifact
persisted by the notebook before it, not the raw files directly (except `01_eda.ipynb`, which
reads the raw CSVs):

1. `01_eda.ipynb` — verifies the raw data and produces `data/processed/node_id_mapping.csv`
2. `02_graph_construction.ipynb` — produces `data/processed/elliptic_pyg_data.pt`
3. `03_graphsage_baseline.ipynb` — produces `outputs/models/baseline_graphsage.pt`
4. `04_ablation_study.ipynb` — produces `outputs/models/final_locked_graphsage.pt` and the
   sealed test result

All hyperparameters and design decisions resolved by this pipeline are recorded in
`configs/config.yaml`, with inline comments documenting where and why each value was decided.

## Stack

Python 3.12, PyTorch 2.14.0, PyTorch Geometric 2.8.0.post1, scikit-learn, pandas, NumPy.
CPU-only by design (see `final_report.md` for the Apple Silicon / PyG support rationale).
