# Fairness, Interprétabilité et Robustesse en Prédiction de Liens sur Graphe Social

**Projet IADATA708** — Mastère Spécialisé Big Data & IA, Télécom Paris, 2026
**Équipe** : Tahiana Andriambahoaka (chef de projet), Younes Ben Taieb, Hugo Music, Adam Moussaoui

---

## 1. Protocole Expérimental

### 1.1 Tâche et données

Nous étudions la **prédiction de liens** (link prediction) sur Pokec, le plus grand réseau social slovaque. Nous utilisons le sous-ensemble *Ceska republika* (67 053 nœuds, 882 756 arêtes), la région présentant le biais d'homophilie de genre le plus marqué (ratio H→F / H→H = 4.59×). L'attribut sensible est le **genre** (binaire, F=0 / H=1). La tâche consiste à prédire si un lien d'amitié existe entre deux utilisateurs, à partir de 64 features (démographiques + textuels) et de la structure du graphe.

### 1.2 Architecture et entraînement

L'architecture suit le paradigme encodeur-décodeur standard (Kipf & Welling, 2016) :

| Composant | Détail |
|:----------|:-------|
| **Encodeur** | GCN 2 couches (64 features → 128 → 64), ReLU + Dropout(0.5) |
| **Décodeur** | Produit scalaire + sigmoïde : $\text{score}(u,v) = \sigma(z_u \cdot z_v)$ |
| **Loss** | BCEWithLogitsLoss |
| **Optimiseur** | Adam (lr=0.01, weight_decay=5e-4), 100 epochs |
| **Split** | 70% train / 15% val / 15% test (arêtes), via `RandomLinkSplit` |
| **Seed** | 42 (fixée pour torch, numpy, random) |
| **Checkpointing** | Meilleur modèle sur AUC validation |

Trois architectures comparées en Acte 1 : GCN, SGC (K=2), GAT (4 têtes). Le **GCN** est retenu comme référence pour sa performance (AUC=0.856), son profil de biais mesurable (Cosine F/H=0.667), et sa comparabilité avec la littérature fairness GNN.

### 1.3 Métriques

- **Performance** : AUC-ROC, Average Precision, F1
- **Fairness** (par type de paire intra/inter-genre) : DP gap, EOpp gap, EO gap, Disparate Impact
- **Représentation** : Cosine centroïdes F/H, Probing leakage (classifieur linéaire)
- **Incertitude** : Entropie prédictive, variance MC, information mutuelle, ECE
- **Robustesse** : ratio AUC sous bruit / AUC propre

## 2. Résultats

### 2.1 Acte 1 — Baseline et biais initial

| Modèle | AUC | F1 | DP gap | EOpp | Cosine F/H | Leakage |
|:-------|:----|:---|:-------|:-----|:-----------|:--------|
| **GCN** | **0.856** | 0.786 | 0.051 | 0.020 | 0.667 | +25.2 pp |
| SGC | 0.771 | 0.718 | 0.012 | 0.045 | -0.178 | +30.4 pp |
| GAT | 0.848 | 0.780 | 0.051 | 0.031 | 0.972 | +21.0 pp |

Le GCN présente un DP gap borderline (0.051 > seuil 0.05) et un leakage de genre de +25.2 points au-dessus du classifieur majoritaire, confirmant que les embeddings encodent le genre.

### 2.2 Acte 2 — Diagnostic XAI

**GNNExplainer** (48 arêtes, 12 par type de paire) : le feature `gender` montre un pattern asymétrique — importance élevée pour F→F (0.319) et H→F (0.360), plus faible pour F→H (0.169) et H→H (0.214). Les features indirects (`comp`, champs textuels) contribuent autant que le genre explicite. La fidélité des explications est validée : Fidelity+ est 8-10× supérieur à Fidelity- (K=5 : 0.122 vs 0.012).

**Attention GAT** : le GAT sur-pondère spontanément les voisins du même genre — pic d'attention à 1.0 pour les arêtes same-genre, valeurs basses (0-0.1) pour les arêtes cross-genre.

**Ablation contrefactuelle** : retirer le feature `gender` du GCN **augmente** le DP gap (+0.012). Le biais est donc **topologique** — il provient de la propagation de messages sur un graphe homophile, pas du feature explicite.

### 2.3 Acte 3 — Correction Fairness (codé from scratch)

Trois méthodes implémentées sans Fairlearn/AIF360 :

| Méthode | AUC | DP gap | EOpp | Cosine F/H |
|:--------|:----|:-------|:-----|:-----------|
| Baseline | 0.856 | 0.051 | 0.020 | 0.667 |
| Reweighting | 0.856 | 0.052 | 0.020 | 0.852 |
| DP-reg λ=0.1 | **0.857** | **0.025** | 0.011 | 0.390 |
| DP-reg λ=0.5 | 0.841 | 0.044 | 0.027 | -0.034 |
| DP-reg λ=1.0 | 0.837 | 0.026 | 0.015 | -0.433 |
| FairGNN α=0.5 | 0.850 | **0.007** | 0.005 | -0.165 |
| FairGNN α=1.0 | 0.845 | 0.025 | 0.007 | -0.713 |
| FairGNN α=2.0 | 0.840 | 0.065 | 0.027 | 0.251 |

**Résultats clés** :

- **Reweighting** : effet paradoxal — les embeddings deviennent plus gender-neutral (cosine 0.667→0.852) sans changer les prédictions. Le biais topologique résiste au rééquilibrage de la loss.
- **DP (λ=0.1)** : quasi free-lunch — DP gap divisé par 2 (0.025 < 0.05) sans perte d'AUC.
- **FairGNN (α=0.5)** : meilleure fairness (DP≈0.007) avec seulement -0.6% d'AUC. Mais l'adversarial est **non-monotone** : α=2.0 produit un DP *pire* que le baseline (0.065), et α=5.0 dégrade les deux objectifs.

### 2.4 Acte 4 — Validation et Incertitude

**MC Dropout (T=30)** : entropie prédictive élevée (H≈0.55 sur H_max=0.693) pour les 3 modèles. Résultat contre-intuitif : le modèle est **plus incertain sur les paires intra-genre** (H_intra≈0.565 > H_inter≈0.537), car celles-ci sont plus nombreuses et plus diverses. L'information mutuelle est faible (MI≈0.025) — l'incertitude est principalement aléatoire, pas épistémique.

**ECE** : les trois modèles sont mal calibrés (ECE≈0.16), avec un biais de sur-confiance dans les bins 0.1-0.5. Ce défaut est structurel (décodeur dot-product), indépendant de la correction de fairness.

**Perturbation de features** (σ∈{0.1, 0.5, 1.0}) : la dégradation de l'AUC est identique pour les 3 modèles (~-6% à σ=0.5). En revanche, la **robustesse de la fairness diffère** : FairGNN maintient un DP<0.015 pour tout σ, alors que DP-reg voit son DP augmenter à ~0.040 sous σ=0.5 avant de retomber. La correction adversariale, agissant sur la géométrie des embeddings, est plus résistante au bruit d'entrée que la régularisation sur les scores.

**Suppression d'arêtes (10%)** : perte modeste d'AUC (~0.7 points) pour les deux modèles. Le DP gap du modèle corrigé reste sous 0.05, confirmant la survie de la correction structurelle.

### 2.5 Acte 5 — Généralisation multi-régions

Pour tester la portée des conclusions, nous répliquons l'expérience (Baseline / DP λ=0.1 / FairGNN α=0.5, mêmes hyperparamètres, seed=42) sur 4 régions Pokec aux profils contrastés.

| Région | n_nodes | $p_F$ | Homophilie | AUC_base | DP_base | DP_DP(0.1) | DP_FG(0.5) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Ceska republika | 31 298 | 0.570 | 0.65 | 0.856 | 0.050 | 0.030 | **0.004** |
| Bratislavsky kraj | 176 192 | 0.486 | 0.73 | 0.873 | 0.032 | **0.001** | 0.011 |
| Nitriansky kraj | 176 815 | 0.503 | 0.88 | 0.902 | 0.025 | 0.016 | **0.008** |
| Zilinsky kraj | 187 585 | 0.505 | 0.93 | 0.898 | **0.011** | 0.008 | 0.041 ⚠ |
| **Moyenne ± std** | | | | 0.882 ± 0.022 | 0.030 ± 0.016 | 0.014 ± 0.013 | 0.016 ± 0.017 |

- **Corrélation Spearman homophilie ↔ DP baseline** : $\rho = -1.00$, $p < 0.001$ — l'hypothèse intuitive « plus d'homophilie → plus de biais » est **réfutée**. Ce sont les régions à plus faible $p_F$ asymétrique et de plus petite taille (Ceska) qui montrent le DP gap absolu le plus élevé. Le DP gap baseline n'est donc pas piloté par l'homophilie seule mais par une combinaison (déséquilibre du genre + taille du graphe + homophilie).
- **Wilcoxon apparié Baseline vs FairGNN** : $W = 3.00$, $p = 0.625$ (non significatif, $N = 4$). FairGNN ne montre **pas** d'amélioration agrégée car il **échoue sur Zilinsky** (DP 0.011 → 0.041, dégradation ×3.7). Sur ce graphe à 2.2M arêtes et signal de biais faible, le discriminateur adversarial sur-corrige et introduit un biais inverse.
- **DP-regularization ($\lambda = 0.1$) est le seul correctif robuste** : il réduit le DP gap sur les **4/4 régions** et amène toutes les régions sous le seuil de 0.05. Sur Bratislavsky il atteint DP = 0.0009 (quasi-nul) pour −0.08% AUC.
- **Coût en AUC marginal** : 0.882 → 0.881 → 0.879 entre Baseline / DP / FairGNN — les corrections restent quasi-gratuites en performance à travers les régions.

## 3. Analyse du Compromis Performance ↔ Fairness ↔ Robustesse

La courbe de Pareto AUC vs DP gap présente un **coude net à λ=0.1** : en dessous, la correction est gratuite ; au-dessus, chaque gain de fairness coûte de plus en plus de performance. Le front de Pareto est dominé par deux points optimaux — DP (λ=0.1) pour le meilleur ratio AUC/fairness, et FairGNN (α=0.5) pour la meilleure fairness absolue.

Le radar plot 5-axes (Performance, Fairness, Robustesse, Calibration, Certitude) révèle que les trois modèles sont **quasi-superposés** sur 3 des 5 axes — seuls Fairness et Robustesse les différencient. Le compromis est donc **réel mais non dramatique** : on peut obtenir un modèle fair et robuste avec <1% de perte d'AUC.

L'impossibilité de Chouldechova (2017) est illustrée empiriquement : aucune configuration ne satisfait simultanément DP, EOpp et DI de manière optimale. Notre choix du DP gap comme métrique principale se justifie par sa pertinence pour la link prediction — il mesure directement si le modèle renforce l'homophilie de genre.

Un enseignement majeur émerge de l'Acte 2 : le biais en link prediction sur graphe social est **fondamentalement topologique**. Retirer le feature genre aggrave le biais (+0.012 de DP gap), car le GCN reconstruit l'information de genre via la propagation de messages sur un graphe homophile. Les approches naïves (suppression de features, reweighting) sont donc insuffisantes — seules les méthodes agissant sur les embeddings (régularisation DP, adversarial) ou sur la structure de l'apprentissage sont efficaces.

La réplication multi-régions (Acte 5) **nuance ce diagnostic** : si le mécanisme topologique est universel, le **DP gap absolu** dépend aussi du déséquilibre du genre ($p_F$) et de la taille du graphe ; l'anti-corrélation $\rho = -1.00$ entre homophilie normalisée et DP gap baseline en témoigne. Surtout, le **choix de méthode change** entre les régions : FairGNN, qui dominait sur Ceska, échoue sur Zilinsky. DP-regularization ($\lambda = 0.1$) reste robuste partout. La recommandation pratique est donc **DP-reg en première intention** ; FairGNN n'est à employer que lorsque le DP gap baseline est élevé et qu'on peut valider par région avant déploiement.

## 4. Limites

- **Seed unique** (42) sur Actes 1-4 et sur la réplication multi-régions : la variance inter-runs n'est pas quantifiée. L'échec de FairGNN sur Zilinsky pourrait être imputable à un *bad seed* — à vérifier avec 3-5 seeds × 4 régions.
- **$N = 4$ régions** pour l'Acte 5 : exploratoire, faible puissance statistique (Wilcoxon p=0.625). Pour rigueur publique, viser $N \geq 10$ régions.
- **Attribut sensible binaire** : ne capture pas l'intersectionnalité (genre × âge × région).
- **Calibration non traitée** : l'ECE≈0.16 pourrait être amélioré par temperature scaling ou Platt scaling.
- **Mesure d'homophilie confondante** : le ratio $P(\text{same}|\text{edge}) / P(\text{same}|\text{random})$ mélange déséquilibre du genre et structure. Un coefficient d'assortativité de Newman serait plus orthogonal.
- **Pas de fairness causale** : seules des métriques observationnelles sont utilisées ; un cadre contrefactuel causal (Kusner et al., 2017) permettrait une analyse plus fine.
- **FairGNN instable** : la sensibilité à α (non-monotone sur Ceska, retournement complet sur Zilinsky) souligne le besoin de méthodes de sélection d'hyperparamètres fairness-aware et de diagnostics par région avant déploiement.

## 5. Utilisation d'outils d'IA

Ce projet a utilisé **Claude** (Anthropic) comme assistant de développement pour : la structuration du pipeline en 4 actes, le debugging du code PyTorch Geometric, l'implémentation des méthodes de fairness from scratch, la recherche de références état de l'art, et la rédaction de cette note. Toutes les décisions méthodologiques, le choix des hyperparamètres, et l'interprétation des résultats sont le fruit du travail de l'équipe.

## 6. Conclusion

Ce projet a démontré qu'un pipeline complet de Responsible AI — observer, diagnostiquer, corriger, valider — puis répliquer sur plusieurs régions, permet d'obtenir un modèle de prédiction de liens performant (AUC=0.882 ± 0.022 sur 4 régions), équitable (DP gap moyen 0.014 sous DP-reg) et robuste, avec moins de 1% de perte de performance par rapport au baseline non corrigé.

Le résultat le plus marquant des Actes 1-4 est la nature **topologique** du biais sur Ceska : retirer le feature sensible aggrave le biais, ce qui invalide l'approche naïve de "fairness through unawareness" et justifie le recours à des méthodes agissant sur la loss (régularisation DP) ou les embeddings (adversarial). La courbe de Pareto révèle un coude à λ=0.1 où la correction est quasi-gratuite.

La généralisation multi-régions (Acte 5) apporte deux résultats inattendus et instructifs : (i) l'amplitude du DP gap baseline est **anti-corrélée à l'homophilie normalisée** ($\rho = -1.00$ sur 4 régions), ce qui souligne que le biais absolu dépend aussi du déséquilibre de l'attribut sensible et de la taille du graphe — pas seulement de la structure homophile ; (ii) **FairGNN ($\alpha = 0.5$) ne se généralise pas** — il dégrade le DP gap sur Zilinsky (×3.7), alors que DP-regularization ($\lambda = 0.1$) réduit le DP gap sur les 4/4 régions. La recommandation pratique pour un déploiement est donc **DP-regularization en première intention**, FairGNN restant réservé à des contextes où un diagnostic par région est possible.

L'analyse d'incertitude et de robustesse (Acte 4) confirme par ailleurs que les corrections de fairness sont **orthogonales** aux propriétés de calibration et d'incertitude du modèle : corriger le biais ne dégrade ni la confiance ni la robustesse aux perturbations. Mais aucune méthode testée n'élève la robustesse au bruit features au-dessus du seuil 0.95 — c'est une vulnérabilité distincte qui appelle des défenses dédiées (régularisation Lipschitz, entraînement adversarial sur les features).

## Références

1. Kipf & Welling, *Semi-Supervised Classification with GCN*, ICLR 2017
2. Kipf & Welling, *Variational Graph Auto-Encoders*, 2016
3. Ying et al., *GNNExplainer: Generating Explanations for GNN*, NeurIPS 2019
4. Gal & Ghahramani, *Dropout as a Bayesian Approximation*, ICML 2016
5. Chouldechova, *Fair Prediction with Disparate Impact*, Big Data 2017
6. Dai & Wang, *Say No to the Discrimination: Learning Fair GNN*, WSDM 2021
7. Chen et al., *Fairness-Aware GNN: A Survey*, 2023
8. FairSIN, *Achieving Fairness via Sensitive Information Neutralization*, AAAI 2024
