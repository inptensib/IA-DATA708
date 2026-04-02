# CLAUDE.md — Projet Responsible AI (IADATA708)

## Identité du Projet

Ce repo contient le projet final du module **IADATA708 — Algorithmic Fairness, Interpretability and Robustness** du Mastère Spécialisé Big Data & IA de Télécom Paris.

**Objectif** : Construire un pipeline complet de Responsible AI sur des données de graphe social (Pokec-z), en suivant une approche scientifique en 4 actes : observer → diagnostiquer → corriger → valider.

## Équipe

- 4 membres (Mastère Spécialisé Big Data/IA, Télécom Paris)
- Chef de projet : Tahiana Andriambahoaka

## Livrables (exigences exactes du PDF)

Le projet ne demande que **2 livrables** :

1. **Un notebook reproductible** (`.ipynb`) — réplique l'ensemble des expériences (entraînement, évaluation, figures). Tout le code est DANS le notebook, pas de modules externes.
2. **Une note d'analyse** (1-2 pages max) — protocole expérimental, résultats clés, analyse d'au moins un compromis perf/équité/robustesse, limites.

Pas de structure `src/` ni de modules séparés. Le notebook est autonome et exécutable de bout en bout.

## Barème de Notation

| Critère | Poids | Ce que ça implique |
|---------|-------|-------------------|
| Implémentation correcte & reproductibilité | **40%** | Seeds fixées, notebook exécutable, résultats stables |
| Qualité du protocole expérimental | **20%** | Splits documentés, hyperparamètres listés, métriques multiples |
| Analyse des résultats & compromis | **30%** | Courbe Pareto, discussion compromis perf↔fairness↔robustesse |
| Clarté de la note d'analyse | **10%** | 2 pages max, figures intégrées, limites discutées |

**Remarque importante du PDF** : L'objectif n'est PAS d'atteindre les meilleures performances possibles, mais de proposer une **analyse rigoureuse et interprétable** des effets des contraintes d'équité et de robustesse.

## Contexte Technique

### Dataset : Pokec-z

- **Source** : Pokec (plus grand réseau social slovaque), sous-ensemble région de Zilina
- **Taille** : ~67K nœuds (utilisateurs), ~882K arêtes (amitiés)
- **Features des nœuds** : attributs démographiques (âge, éducation, région, etc.)
- **Attribut sensible** : genre (binaire, 0/1)
- **Variable cible** : domaine de travail (working field) — classification de nœuds
- **Format** : objet `torch_geometric.data.Data` avec `x`, `edge_index`, `y`, et un attribut `sens` pour le genre

### Architecture Modèle : GCN

- **Base** : GCN de Kipf & Welling (2017)
- **Architecture** : 2 couches GCNConv (features → 64 → num_classes), ReLU + Dropout(0.5) entre les couches
- **Entraînement** : CrossEntropyLoss, Adam (lr=0.01), 200 epochs
- **Split** : Train 50% / Val 25% / Test 25% (split aléatoire, seed fixée)
- **Seed globale** : `torch.manual_seed(42)`, `np.random.seed(42)` au début du notebook

### Pipeline en 4 Actes

Le projet suit l'approche recommandée par le professeur :

**Acte 1 — Baseline** : Entraîner le GCN et mesurer performance + biais initial
- Métriques : Accuracy, F1-macro, DP gap, EOpp gap, EO gap

**Acte 2 — Diagnostic XAI** : Utiliser l'interprétabilité pour comprendre d'où vient le biais
- GNNExplainer (Ying et al., NeurIPS 2019) pour l'explication locale
- GNNShap (Mastropietro et al., WWW 2024) pour l'explication globale via Shapley values
- Évaluation XAI (cours Interprétabilité p.93-97) : Fidelity+ (prédiction avec top-K features → doit rester stable), Fidelity- (prédiction sans top-K → doit chuter)

**Acte 3 — Correction Fairness** : Appliquer des méthodes de mitigation ciblées par le diagnostic
- Reweighting (pré-traitement) : pondérer la loss par groupe
- Régularisation DP (in-processing) : L_total = L_task + λ · L_fair avec λ ∈ {0, 0.1, 0.5, 1, 5, 10}
- Courbe de Pareto accuracy vs DP gap
- **CODER FROM SCRATCH** (exigence de la prof, pas d'import Fairlearn)

**Acte 4 — Validation & Incertitude** : Vérifier les corrections, quantifier l'incertitude
- MC Dropout (T=30 forward passes), entropie prédictive, information mutuelle
- Perturbations : suppression d'arêtes (5%, 10%, 20%) + bruit gaussien (σ=0.1, 0.5, 1.0)
- Analyse du compromis 3D : Performance ↔ Fairness ↔ Robustesse

## Structure du Repository

```
IA-DATA708/
├── CLAUDE.md              # Ce fichier — contexte pour Claude Code
├── README.md              # README du projet
├── requirements.txt       # Dépendances Python
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb       # Exploration des données Pokec-z
│   └── 02_pipeline.ipynb  # Notebook principal : les 4 actes
├── report/
│   └── note_analyse.md    # Note d'analyse (1-2 pages)
└── data/
    └── pokec/             # Dataset Pokec-z (gitignored)
```

## Métriques de Fairness — Formules de Référence

Toujours utiliser ces formules exactes dans le code.

**Demographic Parity (DP) Gap** :
```
DP_gap = |P(Ŷ=1|A=0) − P(Ŷ=1|A=1)|
```

**Equality of Opportunity (EOpp) Gap** :
```
EOpp_gap = |TPR_{A=0} − TPR_{A=1}|
où TPR_a = P(Ŷ=1|Y=1, A=a)
```

**Equalized Odds (EO) Gap** :
```
EO_gap = |TPR_{A=0} − TPR_{A=1}| + |FPR_{A=0} − FPR_{A=1}|
où FPR_a = P(Ŷ=1|Y=0, A=a)
```

**Seuil** : DP_gap < 0.05 est considéré "fair" (80% rule inversée).

## Métriques d'Incertitude — Formules de Référence (Acte 4)

Formules exactes vérifiées depuis le cours Franchi (p.101-110).

**MC Dropout** (T forward passes, dropout activé à l'inférence) :
```
P(Y=k|x) ≈ (1/T) Σ_{t=1}^{T} P(Y=k|ω ⊙ b^t, x)
b^t ~ Bernoulli(p), T=30, p=0.5
```

**Entropie prédictive** (incertitude totale) :
```
H(P(Y|x)) = −Σ_k P(Y=k|x) log P(Y=k|x)
```

**Information mutuelle** (incertitude épistémique) :
```
I(Y|x) = H(P(Y=k|x)) − E_{P(ω|D)}[H(P(Y=k|ω,x))]
        = Entropie totale − Moyenne des entropies individuelles
```

**ECE — Expected Calibration Error** :
```
ECE = Σ_{i=1}^{m} (|B_i|/N) × |accuracy(B_i) − confidence(B_i)|
m = 10 ou 15 bins, ECE=0 → calibration parfaite
```
Visualisation : **reliability diagram** (confiance vs accuracy par bin).

**Reweighting** (formule exacte, cours Mitigation p.4-10) :
```
w(a, y) = P(A=a) × P(Y=y) / P(A=a, Y=y)
```

## Conventions dans le Notebook

### Organisation du notebook

Le notebook est structuré en **sections claires** correspondant aux 4 actes :

```
# 0. Setup & Data Loading
# 1. Acte 1 — Baseline GCN
# 2. Acte 2 — Diagnostic XAI
# 3. Acte 3 — Correction Fairness
# 4. Acte 4 — Validation & Incertitude
# 5. Conclusion & Résumé des résultats
```

### Règles de code

- **Première cellule** : imports + seeds + device setup
- **Fonctions helper** : définies au début de chaque section qui en a besoin (pas de module externe)
- **Reproductibilité** : `set_seed(42)` avant chaque entraînement
- **Visualisations** : chaque figure a un titre, labels d'axes, légende (affichée inline dans le notebook)
- **Résultats** : chaque expérience log ses métriques dans un DataFrame (affiché dans le notebook)
- **Markdown cells** : entre chaque section et sous-section, expliquer l'objectif et commenter les résultats

### Style Python

- Docstrings Google style pour les fonctions helper
- Type hints sur les signatures
- Nommage `snake_case` pour variables/fonctions, `PascalCase` pour classes
- `print()` OK dans le notebook pour afficher les résultats

## Fichiers PDF des Cours (chemins)

| Cours | Fichier | Pages |
|-------|---------|-------|
| Introduction Générale | `/mnt/PDFs/general intro-lecture-ms.pdf` | 21 |
| Fairness MS 1 | `/mnt/PDFs/fairness-ms-1.pdf` | 68 |
| Fairness Mitigation | `/mnt/PDFs/fairness-mitigation.pdf` | 27 |
| Interprétabilité | `/mnt/PDFs/interpretability-lecture-ms-2026.pdf` | 97 |
| Uncertainty | `/mnt/PDFs/_Uncertainty_presentation_MS.pdf` | 127 |
| Sujet du projet | `/mnt/projet/responsibleAI-5.pdf` | 2 |

**Synthèse complète vérifiée** : `references/course_synthesis.md`

## Références Cours (avec pages)

| Cours | Professeur | Pages clés | Usage |
|-------|-----------|------------|-------|
| Intro Générale | d'Alché-Buc | p.15-17, 21 | Cadre Trustworthy AI, piliers |
| Fairness MS 1 | Laclau | p.46-57, 63-68 | Métriques fairness, impossibilité |
| Fairness Mitigation | Laclau | p.4-10, 17-22, 33 | Reweighting, régularisation |
| Interprétabilité | d'Alché-Buc | p.30-36, 69-72, 93-97 | SHAP, CAV/TCAV, évaluation XAI |
| Uncertainty | Franchi | p.25-26, 65, 101-110 | MC Dropout, entropie, calibration |

## Références Papiers Clés

### Fondamentaux (utilisés directement dans le projet)

| Papier | Année | Usage | Lien |
|--------|-------|-------|------|
| Kipf & Welling, GCN | ICLR 2017 | Architecture baseline | [arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907) |
| Ying et al., GNNExplainer | NeurIPS 2019 | XAI locale | [hf.co/papers/1903.03894](https://hf.co/papers/1903.03894) |
| Lundberg & Lee, SHAP | NeurIPS 2017 | Shapley values | [hf.co/papers/1705.07874](https://hf.co/papers/1705.07874) |
| Mastropietro et al., GNNShap | WWW 2024 | XAI globale GNN | [arxiv.org/abs/2401.04829](https://arxiv.org/abs/2401.04829) |
| Gal & Ghahramani, MC Dropout | ICML 2016 | Incertitude bayésienne | [hf.co/papers/1506.02142](https://hf.co/papers/1506.02142) |
| Chouldechova, Fair prediction | Big Data 2017 | Théorèmes impossibilité | [arxiv.org/abs/1703.09207](https://arxiv.org/abs/1703.09207) |
| Kleinberg et al., Inherent Trade-Offs | 2016 | Incompatibilité métriques | [arxiv.org/abs/1609.05807](https://arxiv.org/abs/1609.05807) |
| Dai & Wang, FairGNN | WSDM 2021 | Source dataset Pokec-z + adversarial fairness | [hf.co/papers/2009.01454](https://hf.co/papers/2009.01454) |

### État de l'art (2023-2025, pour discussion et comparaison)

| Papier | Année | Idée clé | Lien |
|--------|-------|----------|------|
| Chen et al., Fair GNN Survey | 2023 | Panorama complet fairness GNN | [hf.co/papers/2307.03929](https://hf.co/papers/2307.03929) |
| FairSIN | AAAI 2024 | Projection orthogonale pour neutraliser info sensible | [arxiv.org/abs/2403.12474](https://arxiv.org/abs/2403.12474) |
| BeMap | 2023 | Message passing équilibré entre groupes | [hf.co/papers/2306.04107](https://hf.co/papers/2306.04107) |
| FUGNN | 2024 | Framework unifié fairness + utilité GNN | [arxiv.org/abs/2405.17034](https://arxiv.org/abs/2405.17034) |
| Zheng et al., Robust Fidelity XAI | 2023 | Évaluation robuste de la fidélité des explications GNN | [hf.co/papers/2310.01820](https://hf.co/papers/2310.01820) |
| Hsu et al., Structured Uncertainty GNN | 2022 | ECE edgewise pour incertitude sur graphes | [hf.co/papers/2210.15575](https://hf.co/papers/2210.15575) |
| Trivedi et al., G-DeltaUQ | 2023 | Incertitude épistémique scalable pour GNN | [hf.co/papers/2309.10976](https://hf.co/papers/2309.10976) |
| Abbahaddou et al., GCORN | 2024 | Robustesse certifiée GCN via orthonormalité | [hf.co/papers/2404.17947](https://hf.co/papers/2404.17947) |
| Zhang et al., Adversarial Attacks on Fair GNN | 2023 | Attaques adversariales sur la fairness des GNN | [hf.co/papers/2310.13822](https://hf.co/papers/2310.13822) |

## Instructions pour Claude Code

### Principes généraux

- **Français avec accents** : tous les textes (markdown cells, commentaires Python, strings affichées) doivent utiliser les accents français corrects (données, arêtes, nœuds, métriques, corrélation, etc.). Ne jamais écrire de mots français sans accents.
- **Tout dans le notebook** : pas de fichiers `.py` séparés, le notebook est autonome
- **Reproductibilité d'abord** : seeds fixées, hyperparamètres documentés
- **Visualisations parlantes** : chaque figure est auto-explicative (titre, labels, légende)
- **Pas de magie noire** : expliquer chaque choix dans des markdown cells
- **Résultats comparables** : les mêmes fonctions de métriques sont réutilisées partout

### Quand on te demande de travailler sur un acte

1. Vérifier d'abord ce qui existe déjà dans le notebook
2. Respecter l'ordre des actes (1 → 2 → 3 → 4), chaque acte dépend du précédent
3. Définir les fonctions helper au début de la section concernée
4. Produire au moins une visualisation par acte
5. Commenter les résultats dans une markdown cell après chaque expérience
6. Les figures s'affichent inline, les résultats dans des DataFrames affichés dans le notebook

### Erreurs courantes à éviter

- Ne PAS utiliser Fairlearn/AIF360 pour les méthodes de base (coder from scratch)
- Ne PAS oublier de mesurer les métriques de fairness dès l'Acte 1
- Ne PAS changer le split train/val/test entre les actes (même split partout)
- Ne PAS entraîner le modèle sur le test set
- Ne PAS ignorer les nœuds sans label dans les métriques de fairness
- L'attribut sensible (genre) NE DOIT PAS être retiré des features du baseline (c'est l'XAI qui le détecte)
- Mention explicite de l'utilisation de Claude dans la note d'analyse (exigence du PDF)
- Toujours utiliser `encoding='utf-8'` dans les `open()` Python pour lire/écrire des fichiers (Windows utilise cp1252 par défaut, ce qui corrompt les accents)

### Recherche état de l'art

Un MCP `paper_search` (HuggingFace) est disponible pour chercher des papiers récents.
Utiliser pour enrichir la discussion dans la note d'analyse avec des références 2023-2025.

Requêtes utiles :
- `paper_search("GNN fairness social network")` → méthodes récentes fairness GNN
- `paper_search("GNN explainability faithfulness")` → évaluation des explications
- `paper_search("uncertainty quantification graph neural network")` → incertitude sur graphes
- `paper_search("adversarial robustness GNN")` → robustesse adversariale

Les références trouvées s'ajoutent dans la section "État de l'art" de la note d'analyse,
pas dans le code. Elles servent à positionner notre approche par rapport aux avancées récentes.
