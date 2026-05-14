# IA-DATA708 — Responsible AI on Social Graphs

Projet final du module IADATA708 — Algorithmic Fairness, Interpretability and Robustness, Mastère Spécialisé Big Data & IA, Télécom Paris, 2026.

## Objectif

Pipeline de Responsible AI sur le réseau social Pokec (sous-région Ceska republika, ~31K nœuds, ~108K arêtes) avec un GCN pour la prédiction de liens. Le projet suit 4 étapes :

1. Baseline — Entraîner un GCN et mesurer performance + biais initial (probing, amplification homophilie)
2. Diagnostic XAI — Saliency, GNNExplainer, SHAP, counterfactual pour identifier les sources de biais
3. Correction Fairness — Reweighting, régularisation DP, FairGNN (codés sans librairie externe)
4. Validation — MC Dropout, ECE, perturbations features + edge removal

## Stack

PyTorch, PyTorch Geometric, scikit-learn, SHAP, matplotlib, seaborn

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
│   ├── 01_eda_link_prediction.ipynb  # Exploration des données Pokec
│   ├── 02_pipeline.ipynb             # Notebook principal : les 4 étapes
│   └── ra_utils.py                   # Helpers réutilisables
├── report/
│   └── note_analyse.md               # Note d'analyse (1-2 pages)
├── references/                       # Synthèse des cours
├── data/                             # Dataset Pokec (gitignored)
├── requirements.txt
└── README.md
```

## Équipe

Tahiana Andriambahoaka, Younes, Mehdi, Amir — Mastère Spécialisé Big Data & IA, Télécom Paris
