# ROADMAP — Responsible AI Pipeline for Link Prediction on Pokec

## Project Overview

**Task:** Link prediction (predicting friendships) on Pokec-z social graph  
**Sensitive attribute:** Gender (binary)  
**Baseline model:** GCN encoder + dot-product decoder  
**Structure:** 4-act scientific pipeline: Observe → Diagnose → Correct → Validate

---

## Act 1 — Baseline Link Prediction ✅ (90% complete)

### Done
- [x] Data loading (Pokec, Ceska republika region — most biased subregion)
- [x] Edge split: train 70% / val 15% / test 15% via `RandomLinkSplit`
- [x] GCN baseline trained (2-layer GCNConv, 128→64, dropout 0.5)
- [x] SGC baseline trained (linear GNN, K=2)
- [x] GAT baseline trained (GATv2Conv, 4 heads, concat→mean)
- [x] Per-pair evaluation (F1 + AUC for F→F, F→H, H→F, H→H)
- [x] Fairness dashboard (DP gap, EOpp gap, EO gap, Disparate Impact)
- [x] Representation bias analysis (centroid cosine similarity + gender probing)
- [x] Pair distribution analysis (data vs predictions, homophily amplification)

### Remaining
- [ ] **Unified comparison table:** Train all 3 models with distinct names (`model_gcn`, `model_sgc`, `model_gat`), produce one DataFrame with AUC, AP, F1, DP, EOpp, EO, DI, Leakage
- [ ] **3-panel comparison plot:** Performance / Fairness / Leakage side by side
- [ ] **Baseline selection justification:** Markdown cell explaining why GCN is selected for correction (worst leakage, standard in fairness literature, simplest architecture = clearest tradeoff)
- [ ] **Cleanup:** Remove duplicate `probe_gender` / `plot_representation_bias` definitions (already in `ra_utils.py`)

### Key Findings So Far
| Model | Cosine F/H | Leakage | DP gap |
|-------|-----------|---------|--------|
| GCN   | 0.67      | +25 pp  | 0.051  |
| GAT   | 0.979     | +20.6 pp| TBD    |
| SGC   | TBD       | TBD     | TBD    |

- GCN amplifies homophily by +7.7 percentage points (intra-genre over-represented)
- All models encode gender in embeddings despite no explicit supervision

---

## Act 2 — Diagnostic XAI 🔲

**Goal:** Understand *where* bias comes from, not just *that* it exists.

### Method 1 — GNNExplainer on edges (local XAI)
- [ ] Apply `torch_geometric.explain.GNNExplainer` to GCN baseline
- [ ] Select ~50 test edges balanced across 4 pair types
- [ ] Extract feature importance masks per edge
- [ ] Compare average feature importance: intra-genre vs inter-genre edges
- [ ] **Key question:** Does the model use different features for different pair types?

### Method 2 — GAT attention analysis (built-in XAI)
- [ ] Use already-extracted attention weights (`alphas` from `model_gat`)
- [ ] Compute mean attention for same-gender vs cross-gender neighbors
- [ ] **Key question:** Does GAT attend more to same-gender neighbors? (explains homophily amplification)

### Method 3 — Counterfactual feature ablation
- [ ] Zero out gender feature column in `data.x`, re-encode with GCN, re-evaluate
- [ ] Compare AUC and DP gap: with vs without gender feature
- [ ] **Key question:** If DP gap barely changes → bias comes from graph structure (topology), not the feature itself

### XAI Evaluation — Fidelity+/- (course requirement)
- [ ] Fidelity+: remove top-K important features → prediction should drop
- [ ] Fidelity-: remove bottom-K features → prediction should be stable
- [ ] Report Fidelity scores to validate GNNExplainer faithfulness

### Deliverables
- 3-4 figures (one per method)
- One markdown interpretation cell per method

---

## Act 3 — Correction Fairness 🔲

**Goal:** Reduce bias on GCN baseline, measure performance↔fairness tradeoff.

### Method 1 — Reweighting (pre-processing, coded from scratch)
- [ ] Classify train edges into intra-genre vs inter-genre
- [ ] Compute weights: `w(type) = N_total / (2 × N_type)`
- [ ] Pass weights to `BCEWithLogitsLoss` during training
- [ ] Train one GCN with reweighting, evaluate

### Method 2 — DP Regularization (in-processing, coded from scratch)
- [ ] Loss: `L_total = L_task + λ × L_fair`
- [ ] `L_fair = |mean(ŷ_intra) - mean(ŷ_inter)|` (differentiable DP gap)
- [ ] Sweep λ ∈ {0, 0.1, 0.5, 1, 5, 10}
- [ ] For each λ: train GCN, evaluate AUC + DP gap on test set

### Method 3 — FairGNN (adversarial debiasing, already coded)
- [ ] Evaluate with same metrics pipeline as other methods
- [ ] Compare against reweighting and DP regularization

### Key Deliverables
- [ ] **Pareto curve:** AUC vs DP gap (one point per λ + baseline + reweighted + FairGNN) — *this is the centerpiece, 30% of grade*
- [ ] **Leakage comparison:** Re-run `probe_gender` on best corrected model vs baseline
- [ ] **Summary table:** All methods × all metrics (AUC, AP, F1, DP, EOpp, DI, Leakage)

### Important Constraints
- **No Fairlearn/AIF360** — all fairness methods coded from scratch (project requirement)
- Same train/val/test split across all experiments (no data leakage)

---

## Act 4 — Validation & Uncertainty 🔲

**Goal:** Verify corrections are robust, quantify prediction uncertainty.

### Part A — MC Dropout (uncertainty quantification)
- [ ] T=30 forward passes with `model.train()` (dropout active at inference)
- [ ] For each test edge: mean prediction p̄ and predictive entropy H = -p̄·log(p̄) - (1-p̄)·log(1-p̄)
- [ ] Compare entropy distributions: intra vs inter-genre pairs
- [ ] Plot: entropy histograms/boxplots by pair type
- [ ] ECE reliability diagram (calibration)
- [ ] **Key question:** Is the model more uncertain about cross-gender friendships?

### Part B — Perturbation robustness
- [ ] **Edge removal:** randomly drop 5%, 10%, 20% of train edges → retrain → measure AUC + DP gap
- [ ] **Feature noise:** Gaussian noise N(0, σ²) with σ ∈ {0.1, 0.5, 1.0} → re-evaluate (no retraining)
- [ ] Run on both baseline GCN and best corrected model from Act 3
- [ ] **Key question:** Does fairness correction survive perturbations?

### Key Deliverables
- [ ] Entropy boxplot by pair type (baseline vs corrected)
- [ ] Robustness table: AUC + DP gap under each perturbation × each model
- [ ] **3D tradeoff summary:** Performance ↔ Fairness ↔ Robustness for all models

---

## Section 5 — Conclusion

- [ ] Final summary table of all results across all acts
- [ ] Discussion of the main tradeoff observed
- [ ] Limitations (dataset size, binary gender, single graph, etc.)

---

## Priority Order (if time is short)

| Priority | Task | Justification |
|----------|------|---------------|
| **1** | Act 3: Pareto curve (λ sweep) | 30% of grade = tradeoff analysis |
| **2** | Act 2: GNNExplainer + Fidelity | Required by course (interpretability) |
| **3** | Act 4: MC Dropout + entropy | Required by course (uncertainty) |
| **4** | Act 4: Perturbation robustness | Completes the 3D tradeoff |
| **5** | Act 2: GAT attention + counterfactual | Adds depth to XAI diagnostic |
| **6** | Act 1: Final comparison cleanup | Polish |

---

## Architecture Summary

```
Input: Pokec graph (N nodes, E edges, D features per node)
       Sensitive attribute: gender (binary)

Encoder (GCN):
  x ∈ ℝ^(N×D) → GCNConv(D, 128) → ReLU → Dropout(0.5) → GCNConv(128, 64) → z ∈ ℝ^(N×64)

Decoder (dot product):
  score(u, v) = σ(z_u · z_v)    ∈ [0, 1]

Loss:
  L_task = BCEWithLogitsLoss(z_u · z_v, y)
  L_fair = |mean(ŷ_intra) - mean(ŷ_inter)|   (Act 3 only)
  L_total = L_task + λ · L_fair

Evaluation:
  - Performance: AUC-ROC, Average Precision, F1 (per pair type)
  - Fairness: DP gap, EOpp gap, EO gap, Disparate Impact (intra vs inter-genre)
  - Representation: Centroid cosine similarity, Gender probing accuracy
  - Uncertainty: Predictive entropy (MC Dropout), ECE
  - Robustness: AUC/DP under edge removal + feature noise
```

---

## References

- Kipf & Welling, GCN (ICLR 2017)
- Kipf & Welling, VGAE (2016) — encoder-decoder link prediction paradigm
- Ying et al., GNNExplainer (NeurIPS 2019)
- Gal & Ghahramani, MC Dropout (ICML 2016)
- Dai & Wang, FairGNN (WSDM 2021) — adversarial debiasing + Pokec dataset source
- Chouldechova, Fair prediction impossibility (Big Data 2017)
