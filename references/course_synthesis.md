# Synthèse Vérifiée des Cours — IADATA708

Ce document compile le contenu vérifié de chaque PDF de cours,
avec les formules exactes, les pages de référence, et les concepts
directement applicables au projet Pokec-z.

---

## 1. Introduction Générale (d'Alché-Buc) — 21 slides

**Fichier** : `/mnt/PDFs/general intro-lecture-ms.pdf`

### Concepts clés

- **Trustworthy AI** : cadre européen (EU AI Act), 7 piliers dont
  fairness, transparency, robustness, accountability
- **Responsible AI** = sous-ensemble opérationnel : fairness +
  interpretability + robustness (les 3 piliers du projet)
- **Bias types** (p.15-17) : historical bias, representation bias,
  measurement bias, aggregation bias, evaluation bias
- **Feedback loops** (p.21) : les biais des modèles renforcent les
  biais sociétaux (cercle vicieux)

### Application au projet

Le cadre Trustworthy AI structure les 4 actes : observer le biais
(Acte 1), l'expliquer (Acte 2), le corriger (Acte 3), valider la
robustesse (Acte 4).

---

## 2. Fairness MS 1 (Laclau) — 68 slides

**Fichier** : `/mnt/PDFs/fairness-ms-1.pdf`

### Métriques de Fairness (p.46-57) — Formules exactes vérifiées

**Demographic Parity (DP)** :
```
DP_gap = |P(Ŷ=1|A=0) − P(Ŷ=1|A=1)|
```
Seuil : DP_gap < 0.05 (80% rule inversée)

**Equality of Opportunity (EOpp)** :
```
EOpp_gap = |TPR_{A=0} − TPR_{A=1}|
où TPR_a = P(Ŷ=1|Y=1, A=a)
```

**Equalized Odds (EO)** :
```
EO_gap = |TPR_{A=0} − TPR_{A=1}| + |FPR_{A=0} − FPR_{A=1}|
où FPR_a = P(Ŷ=1|Y=0, A=a)
```

**Predictive Parity** :
```
PP = |PPV_{A=0} − PPV_{A=1}|
où PPV_a = P(Y=1|Ŷ=1, A=a)
```

### Théorèmes d'impossibilité (p.63-68)

- **Chouldechova (2017)** : si les taux de base diffèrent entre
  groupes (P(Y=1|A=0) ≠ P(Y=1|A=1)), alors on NE PEUT PAS
  satisfaire simultanément DP + EOpp + Predictive Parity
- **Kleinberg et al. (2016)** : incompatibilité formelle entre
  calibration et classification parity (sauf cas dégénéré)
- **Implication projet** : sur Pokec-z, les taux de base diffèrent
  → il faut CHOISIR une métrique (DP comme métrique principale)
  et documenter le compromis

### Autres concepts

- **Individual fairness** (p.38-42) : traiter similairement les
  individus similaires, D(f(x), f(x')) ≤ L·d(x, x')
- **Disparate impact** vs **disparate treatment** (p.25-30)
- **Simpson's paradox** : attention aux analyses agrégées vs
  sous-groupes

---

## 3. Fairness Mitigation (Laclau) — 27 slides

**Fichier** : `/mnt/PDFs/fairness-mitigation.pdf`

### Pré-processing (p.4-10)

**Reweighting** : pondérer les échantillons pour équilibrer la
représentation par groupe × label :
```
w(a, y) = P(A=a) × P(Y=y) / P(A=a, Y=y)
```
Les poids sont appliqués à la loss (CrossEntropy pondérée).

**Autres méthodes pré-processing** (mentionnées mais non requises) :
- Suppression de l'attribut sensible (naïf, ne marche pas à cause
  des proxies)
- Transformation des données (Zemel et al., 2013)
- Relabeling (Kamiran & Calders, 2012)

### In-processing (p.17-22)

**Régularisation DP** :
```
L_total = L_task + λ × L_fair
L_fair = |mean(ŷ[A=0]) − mean(ŷ[A=1])|
```
- λ ∈ {0, 0.1, 0.5, 1, 5, 10} pour sweep
- λ=0 = baseline, λ grand = forte contrainte fairness
- Courbe de Pareto : accuracy vs DP_gap pour chaque λ

**Adversarial debiasing** (mentionné p.22, pas requis dans le
projet) : entraîner un adversaire qui prédit A depuis les
représentations, et entraîner le modèle pour le tromper.

### Post-processing (p.33)

**Threshold adjustment** : ajuster les seuils de décision par
groupe pour égaliser les taux de sélection.
```
Pour chaque groupe a :
  threshold_a = argmin_t |P(Ŷ=1|A=a, threshold=t) − target_rate|
```

### Exigence du projet

**CODER FROM SCRATCH** — pas d'import Fairlearn ni AIF360. Le code
doit être dans le notebook, implémentation manuelle.

---

## 4. Interprétabilité (d'Alché-Buc) — 97 slides

**Fichier** : `/mnt/PDFs/interpretability-lecture-ms-2026.pdf`

### Méthodes de Saliency (p.30-36)

**Saliency map** (gradient-based) :
```
smap(x, f, c) = ∇_x f_c(x)|_{x=x_0}
```
Sensibilité de la sortie de classe c par rapport à chaque feature.

**SmoothGrad** (améliore la stabilité) :
```
SmoothGrad(x) = (1/N) Σ_{i=1}^{N} ∇_x F_c(x + ε_i)
ε_i ~ N(0, σ²I)
```

**GradCAM** (pour réseaux convolutionnels) :
```
α_k^c = (1/Z) ΣΣ ∂F_c/∂A^k_{i,j}
L^c = ReLU(Σ_k α_k^c × A^k)
```

### LIME (p.40-45)

**Local Interpretable Model-agnostic Explanations** :
```
ξ(x) = argmin_{g ∈ G} L(f, g, π_{x'}) + Ω(g)
```
- f = modèle boîte noire
- g = modèle local interprétable (régression linéaire)
- π_{x'} = noyau de proximité
- Ω(g) = régularisation (complexité de g)

Variante K-Lasso : g(z') = ⟨w_g, z'⟩

### LRP — Layer-wise Relevance Propagation (p.50-55)

**Loi de conservation** :
```
Σ_j R_j^{(l)} = Σ_k R_k^{(l+1)} = ... = f(x)
```

**Règles de propagation** :
- LRP-0 : R_j = Σ_k (a_j × w_{jk} / z_k) × R_k
- LRP-ε : R_j = Σ_k (a_j × w_{jk} / (z_k + ε)) × R_k
- LRP-γ : favorise les contributions positives
- **Composite LRP** (best practice) : différentes règles par couche

### CAV / TCAV (Kim et al., ICML 2018) — p.69-72

**Concept Activation Vectors** :
```
v^l_C = normal au hyperplan séparant les exemples du concept C
        vs exemples aléatoires dans l'espace des activations de
        la couche l
```

**TCAV Score** :
```
S_{C,k,l}(x) = ∇_{h_l} f_k(x) · v^l_C
TCAV_Q = fraction des exemples avec sensibilité positive
```

### Concept Bottleneck Models (Koh et al., ICML 2020) — p.75-80

```
ŷ = f(g(x))
```
- g(x) → concepts interprétables (ex: "a des ailes", "est rouge")
- f(concepts) → prédiction finale
- 4 variantes : independent, sequential, joint, standard

### SENN (Meilis et al., 2018) — p.81-85

Self-Explaining Neural Network : le modèle produit ses propres
explications by-design via concept encoder + relevance parametrizer
+ aggregator.

### FLINT (Parekh et al., NeurIPS 2021) — p.86-90

Full SLI (Sparse Linear Interpretable) :
```
Objectif = L_pred + β×L_of + γ×L_if + δ×L_cnc
```
- L_of : orthogonal fidelity
- L_if : input fidelity
- L_cnc : concept coherence

### Évaluation de l'interprétabilité (p.93-97)

**Propriétés d'une bonne explication** :
- **Faithfulness** (fidélité) : l'explication reflète vraiment le
  raisonnement du modèle
- **Stability** : entrées similaires → explications similaires
- **Plausibility** : l'explication est compréhensible pour un humain
  (≠ faithfulness !)
- **Correctness**, **Consistency**, **Completeness**

**Tests de fidélité** :
- Perturbation test : masquer les features "importantes" →
  la prédiction doit changer significativement
- Insertion/Deletion metrics : ajouter/retirer les features par
  ordre d'importance
- Gradient alignment : cohérence entre explications et gradients

**Pour le projet (Acte 2)** :
- Fidelity+ : prédiction avec uniquement les top-K features
  identifiées → doit rester proche de l'original
- Fidelity- : prédiction SANS les top-K features → doit chuter

---

## 5. Uncertainty (Franchi) — 127 slides

**Fichier** : `/mnt/PDFs/_Uncertainty_presentation_MS.pdf`

### Types d'incertitude (p.10-15)

- **Aléatoire (aleatoric)** : irréductible, due au bruit dans les
  données. Ne diminue PAS avec plus de données.
- **Épistémique (epistemic)** : réductible, due au manque de
  connaissance du modèle. Diminue avec plus de données/modèle.

### Deep Ensembles (p.25-26)

```
P(y*|x*) = (1/M) Σ_{j=1}^{M} P(y*|ω^j, x*)
```
- M modèles entraînés indépendamment (différentes init + shuffles)
- Approximation simple de l'incertitude épistémique
- Coût : M × entraînement et M × inférence

### MC Dropout (Gal & Ghahramani, 2016) — p.65

```
P(y*|x*) ≈ (1/T) Σ_{t=1}^{T} P(y*|ω ⊙ b^t, x*)
b^t ~ Bernoulli(p)
```
- T forward passes avec dropout ACTIVÉ à l'inférence
- Approximation variationnelle de l'inférence bayésienne
- **Pour le projet** : T=30, dropout=0.5

### Weak Ensembles (p.70-80)

**BatchEnsemble** :
```
W_j = W ⊙ (r_j × s_j^T)
```
- W partagé, seuls r_j et s_j sont spécifiques à chaque membre
- Coût mémoire ≈ 1 modèle, performances ≈ ensemble

**MIMO** : Multiple-Input Multiple-Output
**Packed-Ensembles** (Laurent et al., ICLR 2023)

### BNN — Bayesian Neural Networks (p.85-100)

**Prédiction marginalisée** :
```
P(Y|X) = ∫ P(Y|X,ω) P(ω|D) dω ≈ (1/M) Σ P(Y|X,ω_i)
ω_i ~ P(ω|D)
```

**Inférence variationnelle** : approximer P(ω|D) par q_λ(ω)
```
λ* = argmin_λ KL(q_λ(ω) || P(ω|D))
```

**ELBO** :
```
ELBO = E_q[log P(D|ω)] − KL(q_λ(ω) || P(ω))
```

**Reparameterization trick** : ω = μ + σ ⊙ ε, ε ~ N(0,I)

**TRADI** (p.95) : Kalman filtering pour tracker la distribution
postérieure des poids du DNN au cours de l'entraînement.

**ABNN** (p.98-100) : convertir un DNN classique en BNN via
Bayesian Normalization Layers (BNL) :
```
BNL(h_j) = ((h_j − μ̂_j) / σ̂_j) × γ_j(1 + ε_j) + β_j
ε_j ~ N(0, 1)
```

### Critères d'incertitude (p.101) — Formules exactes

**MCP (Maximum Class Probability)** :
```
MCP = max_k P(Y=k|x, w)
```
MCP faible → incertitude élevée.

**Entropie prédictive** (single DNN) :
```
H(P(Y|x,w)) = −Σ_k P(Y=k|x,w) log P(Y=k|x,w)
```
Entropie élevée → distribution uniforme → incertitude élevée.
Liée à l'incertitude aléatoire.

**Entropie prédictive** (ensemble/MC Dropout) :
```
H(P(Y|x)) = −Σ_k P(Y=k|x) log P(Y=k|x)
```
Où P(Y=k|x) est la prédiction marginalisée sur les modèles.
Capture incertitude totale (aléatoire + épistémique).

**Information Mutuelle** (décomposition épistémique) :
```
I(Y|x) = H(P(Y=k|x)) − E_{P(ω|D)}[H(P(Y=k|ω,x))]
        = Entropie totale − Entropie aléatoire moyenne
        = Incertitude épistémique
```

### ECE — Expected Calibration Error (p.106-108)

```
ECE = Σ_{i=1}^{m} (|B_i|/N) × |accuracy(B_i) − confidence(B_i)|
```
- m bins (partition des prédictions par confiance)
- B_i = ensemble des échantillons dans le bin i
- ECE = 0 → calibration parfaite
- Sensible au nombre de bins (m=10 ou m=15 typique)
- Visualisation : **reliability diagram** (confiance vs accuracy
  par bin, avec diagonale = calibration parfaite)

### Sparsification Error (p.109-110)

```
SE = Σ_{i=1}^{N} |error(concat(B_i,...,B_N)) − error(concat(O_i,...,O_N))|
```
- Mesure la correspondance entre incertitude prédite et erreur réelle
- SE = 0 → l'incertitude est parfaitement corrélée avec l'erreur

### Évaluation par type d'incertitude (p.111-116)

**Aléatoire (classification)** : NLL, ECE, Accuracy
**Aléatoire (régression)** : Accuracy, Sparsification Error
**Épistémique (classification)** : OOD Detection → AUROC, AUPR, FPR95
**Distribution shift** : Accuracy + ECE + Sparsification Error

### Application au projet (Acte 4)

1. MC Dropout (T=30) sur le modèle baseline ET le modèle fair
2. Calculer pour chaque : entropie prédictive, information mutuelle
3. Comparer les distributions d'incertitude par groupe (A=0 vs A=1)
4. ECE + reliability diagram pour évaluer la calibration
5. Perturbations (arêtes + features) pour tester la robustesse
6. Compromis 3D : performance ↔ fairness ↔ robustesse

---

## 6. Sujet du Projet (responsibleAI-5) — 2 pages

**Fichier** : `/mnt/projet/responsibleAI-5.pdf`

### Exigences exactes

- 2 livrables : notebook reproductible + note d'analyse (1-2 pages)
- Tâche : **classification ou prédiction de liens** (le sujet autorise les deux)
- Coder from scratch (pas Fairlearn/AIF360)
- Mention obligatoire de l'utilisation d'IA
- Barème : 40% implémentation, 20% protocole, 30% analyse, 10% note
- L'objectif n'est PAS la performance max mais l'ANALYSE RIGOUREUSE

### Choix du projet : Link Prediction

La prédiction de liens a été choisie car :
- C'est la tâche la plus naturelle pour un graphe social (prédire les amitiés)
- Le biais d'homophilie (les gens se lient avec ceux qui leur ressemblent)
  se manifeste directement dans les prédictions de liens
- L'angle fairness est riche : le modèle renforce-t-il la ségrégation ?

---

## Mapping Cours → Actes du Projet (Link Prediction)

| Acte | Cours principal | Cours support | Formules clés |
|------|----------------|---------------|---------------|
| 1 — Baseline LP | Fairness MS 1 | Intro Générale | AUC, AP, DP gap (intra/inter), EOpp gap |
| 2 — XAI | Interprétabilité | — | GNNExplainer (arêtes), Fidelity+/- |
| 3 — Fairness | Fairness Mitigation | Fairness MS 1 | Reweighting par type de paire, L_total = L_task + λ×L_fair |
| 4 — Uncertainty | Uncertainty | — | H binaire, Var MC, ECE, MC Dropout sur scores de liens |

### Adaptation des formules de fairness pour la link prediction

Les métriques de fairness des cours (conçues pour la classification) sont
adaptées en remplaçant les groupes de nœuds par les types de paires :

- **Groupe A=0/A=1** → **Type de paire intra/inter-genre**
- **P(Ŷ=1|A=a)** → **P(lien prédit | paire intra/inter)**
- **TPR_a** → **TPR par type de paire**
- **Reweighting w(a,y)** → **w(type_paire) = N_total / (2 × N_type)**

L'incertitude s'adapte naturellement : MC Dropout produit T scores par
paire, et l'entropie binaire H = -p·log(p) - (1-p)·log(1-p) remplace
l'entropie multiclasse.
