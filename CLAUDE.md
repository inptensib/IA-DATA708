# CLAUDE.md — Projet Responsible AI (IADATA708)

## Identité du Projet

Ce repo contient le projet final du module **IADATA708 — Algorithmic Fairness, Interpretability and Robustness** du Mastère Spécialisé Big Data & IA de Télécom Paris.

**Objectif** : Construire un pipeline complet de Responsible AI sur des données de graphe social (Pokec-z), en suivant une approche scientifique en 4 actes : observer → diagnostiquer → corriger → valider.

**Tâche choisie** : **Prédiction de liens** (link prediction) — prédire si deux utilisateurs du réseau social sont amis, à partir de leurs attributs et de la structure du graphe. C'est la tâche la plus naturelle pour un graphe social, et elle expose directement le biais d'homophilie de genre.

## Équipe

- 4 membres (Mastère Spécialisé Big Data/IA, Télécom Paris)
- Chef de projet : Tahiana Andriambahoaka

## Livrables (exigences exactes du PDF)

Le projet ne demande que **2 livrables** :

1. **Un notebook reproductible** (`.ipynb`) **+ un module helpers `notebooks/ra_utils.py`** — réplique l'ensemble des expériences (entraînement, évaluation, figures). Le notebook importe `ra_utils` (situé dans le même dossier) qui contient les helpers réutilisables (constantes, set_seed, classify_pairs, fairness, helpers de viz, verdicts). Le notebook reste exécutable de bout en bout tant que `ra_utils.py` est livré dans le même dossier.
2. **Une note d'analyse** (1-2 pages max) — protocole expérimental, résultats clés, analyse d'au moins un compromis perf/équité/robustesse, limites.

Module unique `notebooks/ra_utils.py` pour les helpers communs aux 4 actes. Le notebook reste autonome via `from ra_utils import *` dans la première cell.

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
- **Tâche** : **Link prediction** — prédire l'existence d'un lien (amitié) entre deux nœuds
- **Format** : objet `torch_geometric.data.Data` avec `x`, `edge_index`, et un attribut `sens` pour le genre

### Architecture Modèle : GCN Encoder + Décodeur de Liens

L'architecture suit le paradigme standard encoder-decoder pour la link prediction sur graphes :

**Encodeur** (GCN) :
- 2 couches GCNConv (features → 128 → 64), ReLU + Dropout(0.5) entre les couches
- Produit un embedding z_u ∈ ℝ^64 pour chaque nœud u
- Base : Kipf & Welling (ICLR 2017)

**Décodeur** (produit scalaire + sigmoïde) :
```
score(u, v) = σ(z_u · z_v)
```
- σ = sigmoïde → score ∈ [0, 1] interprété comme probabilité de lien
- Simple, efficace, et standard pour la link prediction (Kipf & Welling, VGAE 2016)

**Entraînement** :
- **Loss** : BCEWithLogitsLoss (binary cross-entropy sur les logits z_u · z_v)
- **Positive edges** : arêtes existantes du train set
- **Negative sampling** : échantillonner autant d'arêtes négatives (paires non connectées) que de positives
- **Optimiseur** : Adam (lr=0.01), 200 epochs
- **Seed globale** : `torch.manual_seed(42)`, `np.random.seed(42)` au début du notebook

**Split des arêtes** (crucial — on split les ARÊTES, pas les nœuds) :
- Train 70% / Val 15% / Test 15% des arêtes positives
- Mêmes proportions pour les arêtes négatives (échantillonnées)
- Le GCN à l'entraînement ne voit QUE les arêtes train dans `edge_index`
- Les arêtes val/test servent uniquement à l'évaluation
- Utiliser `torch_geometric.transforms.RandomLinkSplit` pour le split

### Métriques de Performance (Link Prediction)

- **AUC-ROC** : mesure principale, capacité à discriminer vrais liens des faux
- **Average Precision (AP)** : utile si les classes sont déséquilibrées
- **Hits@K** (optionnel) : parmi les top-K prédictions, combien sont des vrais liens

### Pipeline en 4 Actes (Link Prediction)

Le projet suit l'approche recommandée par le professeur, adaptée à la prédiction de liens :

**Acte 1 — Baseline** : Entraîner le GCN encoder-decoder et mesurer performance + biais initial
- Métriques de performance : AUC-ROC, AP
- Métriques de fairness : DP gap (inter vs intra-genre), EOpp gap, Disparate Impact
- Analyse de la distribution des scores par type de paire (H-H, F-F, H-F)

**Acte 2 — Diagnostic XAI** : Utiliser l'interprétabilité pour comprendre d'où vient le biais
- GNNExplainer adapté aux arêtes : quelles features/sous-graphes expliquent la prédiction d'un lien
- Importance des features globale : quelles features contribuent le plus aux prédictions intra-genre vs inter-genre
- Évaluation XAI (cours Interprétabilité p.93-97) : Fidelity+/- adaptées aux liens

**Acte 3 — Correction Fairness** : Appliquer des méthodes de mitigation
- Reweighting (pré-traitement) : pondérer la loss selon le type de paire (intra/inter-genre)
- Régularisation DP (in-processing) : L_total = L_task + λ · L_fair avec λ ∈ {0, 0.1, 0.5, 1, 5, 10}
- Courbe de Pareto : AUC vs DP gap pour chaque λ
- **CODER FROM SCRATCH** (exigence de la prof, pas d'import Fairlearn)

**Acte 4 — Validation & Incertitude** : Vérifier les corrections, quantifier l'incertitude
- MC Dropout (T=30 forward passes), entropie prédictive sur les scores de liens
- Perturbations : suppression d'arêtes (5%, 10%, 20%) + bruit gaussien sur les features (σ=0.1, 0.5, 1.0)
- Comparaison incertitude par type de paire (intra vs inter-genre)
- Analyse du compromis 3D : Performance ↔ Fairness ↔ Robustesse

## Structure du Repository

```
IA-DATA708/
├── CLAUDE.md              # Ce fichier — contexte pour Claude Code
├── README.md              # README du projet
├── requirements.txt       # Dépendances Python
├── .gitignore
├── notebooks/
│   ├── 01_eda_link_prediction.ipynb   # Exploration Pokec-z (Ceska republika)
│   ├── 02_pipeline.ipynb              # Notebook principal : les 4 actes
│   └── ra_utils.py                    # Helpers réutilisables (viz, fairness, verdicts)
├── report/
│   └── note_analyse.md    # Note d'analyse (1-2 pages)
├── references/
│   └── course_synthesis.md # Synthèse vérifiée des cours
└── data/
    └── pokec/             # Dataset Pokec-z (gitignored)
```

## Métriques de Fairness pour Link Prediction — Formules de Référence

La fairness en link prediction s'évalue sur les **types de paires** :
- **Intra-genre** : paires (u, v) où genre(u) = genre(v) — même groupe
- **Inter-genre** : paires (u, v) où genre(u) ≠ genre(v) — groupes croisés

**Demographic Parity (DP) Gap** — le modèle prédit-il des liens de manière équitable ?
```
DP_gap = |P(ŷ=1 | paire intra-genre) − P(ŷ=1 | paire inter-genre)|
```
Un DP_gap élevé signifie que le modèle favorise les liens intra-genre (renforce l'homophilie).

**Equality of Opportunity (EOpp) Gap** — le modèle détecte-t-il aussi bien les vrais liens ?
```
EOpp_gap = |TPR_intra − TPR_inter|
où TPR_type = P(ŷ=1 | y=1, paire de type)
```

**Equalized Odds (EO) Gap** :
```
EO_gap = |TPR_intra − TPR_inter| + |FPR_intra − FPR_inter|
```

**Disparate Impact (DI)** :
```
DI = min(P(ŷ=1|intra), P(ŷ=1|inter)) / max(P(ŷ=1|intra), P(ŷ=1|inter))
```
DI ≥ 0.8 = fair (80% rule).

**Alternative : fairness par groupe de nœuds** (complémentaire) :
```
DP_gap_node = |mean(score des liens prédits pour nœuds A=0) − mean(score pour nœuds A=1)|
```

**Seuil** : DP_gap < 0.05 est considéré "fair".

## Métriques d'Incertitude — Formules de Référence (Acte 4)

Formules exactes vérifiées depuis le cours Franchi (p.101-110).
Adaptées pour la link prediction (prédiction binaire par paire).

**MC Dropout** (T forward passes, dropout activé à l'inférence) :
```
P(lien|u,v) ≈ (1/T) Σ_{t=1}^{T} σ(z_u^t · z_v^t)
z_u^t = GCN(x, edge_index; ω ⊙ b^t), b^t ~ Bernoulli(p)
T=30, p=0.5
```

**Entropie prédictive** (incertitude totale, binaire) :
```
H(u,v) = −p·log(p) − (1−p)·log(1−p)
où p = P(lien|u,v) (moyenne MC Dropout)
```
H max = log(2) ≈ 0.693 quand p = 0.5 (incertitude maximale).

**Variance prédictive** (incertitude épistémique, simple pour le binaire) :
```
Var(u,v) = (1/T) Σ_{t=1}^{T} (σ(z_u^t · z_v^t) − p̄)²
```

**Information mutuelle** (incertitude épistémique) :
```
I(u,v) = H(P(lien|u,v)) − (1/T) Σ_{t=1}^{T} H(σ(z_u^t · z_v^t))
        = Entropie totale − Moyenne des entropies individuelles
```

**ECE — Expected Calibration Error** :
```
ECE = Σ_{i=1}^{m} (|B_i|/N) × |accuracy(B_i) − confidence(B_i)|
m = 10 ou 15 bins, ECE=0 → calibration parfaite
```
Visualisation : **reliability diagram** (confiance vs accuracy par bin).

**Reweighting** (formule adaptée à la link prediction) :
```
w(type) = N_total / (2 × N_type)
où type ∈ {intra-genre, inter-genre}
```
Pondère la BCELoss pour équilibrer les types de paires.

## Conventions dans le Notebook

### Organisation du notebook

Le notebook est structuré en **sections claires** correspondant aux 4 actes :

```
# 0. Setup & Data Loading + Edge Split
# 1. Acte 1 — Baseline Link Prediction
# 2. Acte 2 — Diagnostic XAI
# 3. Acte 3 — Correction Fairness
# 4. Acte 4 — Validation & Incertitude
# 5. Conclusion & Résumé des résultats
```

### Règles de code

- **Première cellule** : imports + `from ra_utils import *` + device setup + `set_seed(42)`
- **Fonctions helper communes** : dans `notebooks/ra_utils.py` (importé via `from ra_utils import *` dans la première cell). Les fonctions spécifiques à un seul acte restent locales à la section concernée.
- **Reproductibilité** : `set_seed(42)` avant chaque entraînement (importé depuis `ra_utils`)
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
| Interprétabilité | d'Alché-Buc | p.30-36, 69-72, 93-97 | SHAP, GNNExplainer, évaluation XAI |
| Uncertainty | Franchi | p.25-26, 65, 101-110 | MC Dropout, entropie, calibration |

## Références Papiers Clés

### Fondamentaux (utilisés directement dans le projet)

| Papier | Année | Usage | Lien |
|--------|-------|-------|------|
| Kipf & Welling, GCN | ICLR 2017 | Architecture encodeur | [arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907) |
| Kipf & Welling, VGAE | 2016 | Paradigme encoder-decoder link prediction | [arxiv.org/abs/1611.07308](https://arxiv.org/abs/1611.07308) |
| Ying et al., GNNExplainer | NeurIPS 2019 | XAI locale (nœuds et arêtes) | [hf.co/papers/1903.03894](https://hf.co/papers/1903.03894) |
| Lundberg & Lee, SHAP | NeurIPS 2017 | Shapley values | [hf.co/papers/1705.07874](https://hf.co/papers/1705.07874) |
| Gal & Ghahramani, MC Dropout | ICML 2016 | Incertitude bayésienne | [hf.co/papers/1506.02142](https://hf.co/papers/1506.02142) |
| Chouldechova, Fair prediction | Big Data 2017 | Théorèmes impossibilité | [arxiv.org/abs/1703.09207](https://arxiv.org/abs/1703.09207) |
| Dai & Wang, FairGNN | WSDM 2021 | Source dataset Pokec-z | [hf.co/papers/2009.01454](https://hf.co/papers/2009.01454) |

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

## Instructions pour Claude Code

### Principes généraux

- **Français avec accents — SYSTÉMATIQUE** : Les accents français corrects (é, è, ê, à, ç, î, ô, œ, etc.) sont **obligatoires partout** :
  - Markdown cells : accents obligatoires (rendu HTML, aucun risque)
  - Commentaires Python : accents obligatoires
  - `print()` et strings affichées : accents obligatoires (Jupyter rend en HTML/UTF-8)
  - Matplotlib titres, labels, légendes : accents obligatoires (les polices standard supportent Latin-1)
  - **Exception** : variables Python et clés de dict restent en ASCII (convention Python)
  - **Exception** : les symboles spéciaux Unicode (→ × ⁻⁵ ≈) doivent être remplacés par ASCII (`->`, `x`, `x1e-5`, `~`) dans le code Python uniquement (problèmes de rendu matplotlib sur Windows)
  - Exemples corrects : `print(f"Nœuds : {n:,}")`, `ax.set_title('Densité du graphe')`, `# Arêtes réciproques`
  - Exemples incorrects : `print("Noeuds")`, `ax.set_title('Densite')`, `# Aretes`
- **Tout dans le notebook** : pas de fichiers `.py` séparés, le notebook est autonome
- **Reproductibilité d'abord** : seeds fixées, hyperparamètres documentés
- **Visualisations parlantes** : chaque figure est auto-explicative (titre, labels, légende)
- **Pas de magie noire** : expliquer chaque choix dans des markdown cells
- **Résultats comparables** : les mêmes fonctions de métriques sont réutilisées partout

### Spécificités Link Prediction

- **Split des arêtes** : utiliser `RandomLinkSplit` de PyG, avec `num_val=0.15, num_test=0.15`
- **Negative sampling** : toujours fixer la seed pour la reproductibilité
- **Types de paires** : toujours tracker si une paire est intra-genre (same) ou inter-genre (cross)
- **Décodeur** : dot product + sigmoïde comme baseline, pas de MLP complexe
- **Évaluation** : AUC et AP sur les arêtes val/test (jamais sur les arêtes train)
- **Fairness** : toujours calculer les métriques séparément pour intra/inter-genre

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
- Ne PAS utiliser les arêtes val/test dans le `edge_index` du GCN à l'entraînement (fuite de données !)
- Ne PAS oublier le negative sampling pour l'évaluation
- L'attribut sensible (genre) NE DOIT PAS être retiré des features du baseline (c'est l'XAI qui le détecte)
- Mention explicite de l'utilisation de Claude dans la note d'analyse (exigence du PDF)
- Toujours utiliser `encoding='utf-8'` dans les `open()` Python pour lire/écrire des fichiers (Windows utilise cp1252 par défaut, ce qui corrompt les accents)

### Recherche état de l'art

Un MCP `paper_search` (HuggingFace) est disponible pour chercher des papiers récents.
Utiliser pour enrichir la discussion dans la note d'analyse avec des références 2023-2025.

Requêtes utiles :
- `paper_search("fair link prediction graph neural network")` → fairness en link prediction
- `paper_search("GNN explainability link prediction")` → XAI pour les liens
- `paper_search("uncertainty quantification graph neural network")` → incertitude sur graphes
- `paper_search("adversarial robustness GNN link prediction")` → robustesse

Les références trouvées s'ajoutent dans la section "État de l'art" de la note d'analyse,
pas dans le code. Elles servent à positionner notre approche par rapport aux avancées récentes.
