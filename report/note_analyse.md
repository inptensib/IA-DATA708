# Note d'Analyse — Responsible AI sur Pokec-z

Projet IADATA708 — Algorithmic Fairness, Interpretability and Robustness.
Mastère Spécialisé Big Data & IA, Télécom Paris, 2026.

## 1. Protocole Expérimental

**Dataset** : Pokec-z, ~67K nœuds, ~882K arêtes, attribut sensible =
genre (binaire), cible = working field (multiclass).

**Modèle** : GCN 2 couches (Kipf & Welling, 2017).

| Hyperparamètre      | Valeur          |
|:---------------------|:----------------|
| Hidden dim           | 64              |
| Dropout              | 0.5             |
| Learning rate        | 0.01            |
| Epochs               | 200             |
| Split                | 50/25/25        |
| Seed                 | 42              |
| Optimizer            | Adam            |

**Métriques** : Accuracy, F1-macro, DP gap, EOpp gap, EO gap.

**Méthodes de correction** : reweighting (pré-processing),
régularisation DP avec λ ∈ {0, 0.1, 0.5, 1, 5, 10} (in-processing).
Toutes codées from scratch.

**Diagnostic XAI** : GNNExplainer (local), GNNShap simplifié (global).

**Incertitude** : MC Dropout (T=30), entropie prédictive, information
mutuelle. Perturbations : suppression d'arêtes (5/10/20%), bruit
gaussien (σ=0.1/0.5/1.0).

## 2. Résultats Clés

<!-- TODO : remplir avec les résultats du notebook -->

**Baseline** :

| Métrique   | Valeur |
|:-----------|:-------|
| Accuracy   |        |
| F1-macro   |        |
| DP gap     |        |
| EOpp gap   |        |
| EO gap     |        |

**Diagnostic XAI** : le genre apparaît dans le top-K features
identifiées par GNNExplainer/GNNShap, confirmant le biais structurel
amplifié par la propagation de messages sur un graphe homophile.

**Après correction** :

| Modèle        | Accuracy | DP gap | EOpp gap |
|:--------------|:---------|:-------|:---------|
| Baseline      |          |        |          |
| Reweighting   |          |        |          |
| DP-reg λ=0.1  |          |        |          |
| DP-reg λ=1    |          |        |          |
| DP-reg λ=10   |          |        |          |

<!-- Insérer figure : Courbe de Pareto accuracy vs DP gap -->

## 3. Analyse du Compromis Performance ↔ Fairness ↔ Robustesse

<!-- TODO : analyse détaillée -->

La courbe de Pareto montre que [décrire le compromis observé].

Sous perturbations (suppression d'arêtes, bruit features), le modèle
fair [est plus/moins] robuste que le baseline. L'entropie prédictive
du groupe minoritaire [augmente/diminue] après correction.

**Impossibilité** (Chouldechova, 2017 ; Kleinberg et al., 2016) : les
taux de base diffèrent sur Pokec-z (P(Y=1|A=0) ≠ P(Y=1|A=1)), donc
satisfaire simultanément DP, EOpp et Predictive Parity est impossible.
Notre choix de DP comme métrique principale se justifie par [raison].

## 4. Limites

- GCN à 2 couches : architecture simple, pas de GAT ni GraphSAGE
- Seed unique (42) : pas de variance inter-runs quantifiée
- Attribut sensible binaire : ne capture pas l'intersectionnalité
- GNNShap simplifié : approximation par permutation, pas la version
  exacte de Mastropietro et al. (2024)
- Dataset spécifique : Pokec-z (Slovaquie), généralisation non testée
- Pas de fairness causale : seules les métriques observationnelles

## 5. Utilisation d'Outils d'IA

Ce projet a utilisé **Claude** (Anthropic) comme assistant de
développement pour : la structuration du pipeline, le debugging,
la recherche de papiers état de l'art, et la rédaction de la note.
Toutes les décisions méthodologiques et l'analyse des résultats sont
le fruit du travail de l'équipe.

## Références

- Kipf & Welling, *Semi-Supervised Classification with GCN*, ICLR 2017
- Ying et al., *GNNExplainer*, NeurIPS 2019
- Mastropietro et al., *GNNShap*, WWW 2024
- Gal & Ghahramani, *Dropout as Bayesian Approximation*, ICML 2016
- Chouldechova, *Fair Prediction with Disparate Impact*, Big Data 2017
- Kleinberg et al., *Inherent Trade-Offs in Fair Determination*, 2016
- Dai & Wang, *FairGNN*, WSDM 2021
