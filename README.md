# IA-DATA708 — Responsible AI on Social Graphs

Projet final du module **IADATA708 — Algorithmic Fairness, Interpretability and Robustness**, Mastère Spécialisé Big Data & IA, Télécom Paris, 2026.

## Objectif

Pipeline complet de Responsible AI sur le graphe social **Pokec-z** (~67K nœuds, ~882K arêtes) avec un GCN pour la classification de nœuds. Le projet suit une approche en 4 actes :

1. **Baseline** — Entraîner un GCN et mesurer performance + biais initial
2. **Diagnostic XAI** — GNNExplainer + GNNShap pour identifier les sources de biais
3. **Correction Fairness** — Reweighting + régularisation DP (from scratch)
4. **Validation** — MC Dropout + perturbations pour vérifier robustesse et incertitude

## Stack

PyTorch, PyTorch Geometric, scikit-learn, matplotlib, seaborn, NetworkX

## Installation

```bash
# Installer torch en premier (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Puis les autres dépendances
pip install -r requirements.txt
```

## Structure

```
├── notebooks/
│   ├── 01_eda.ipynb           # Exploration des données Pokec-z
│   └── 02_pipeline.ipynb      # Notebook principal : les 4 actes
├── report/
│   └── note_analyse.md        # Note d'analyse (1-2 pages)
├── references/                # Synthèse des cours et papiers
├── data/                      # Dataset Pokec-z (gitignored)
├── requirements.txt           # Dépendances Python
└── README.md
```

## Équipe

Tahiana Andriambahoaka, Younes, Mehdi, Amir — Mastère Spécialisé Big Data & IA, Télécom Paris
