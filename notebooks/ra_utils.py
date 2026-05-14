"""Helpers pour le notebook 02_pipeline.ipynb  - Responsible AI sur Pokec."""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


# -- Constantes
PAIR_ORDER = ['F->F', 'F->H', 'H->F', 'H->H']
PAIRS = [(name, idx) for idx, name in enumerate(PAIR_ORDER)]
C_INTRA, C_INTER = '#2196F3', '#FF9800'
# Liste des 64 features produites par load_pokec (4 démographiques + 30 has_text + 30 length_text)
FEATURE_NAMES = ['comp', 'age', 'public', 'gender'] + [f'h{i}' for i in range(30)] + [f'l{i}' for i in range(30)]


# -- Setup
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# -- Chargement données
def load_pokec(data_dir, region='ceska republika'):
    """Charge le sous-ensemble Pokec filtré par région.

    Returns:
        data: torch_geometric.data.Data avec x, edge_index, sens
        sens: tenseur 1D des genres (0=F, 1=H)
        p: DataFrame pandas des profils (pour analyses descriptives)
    """
    TC_IDX = list(range(9, 29)) + list(range(31, 37)) + [38, 39, 40, 42]  # 30 colonnes texte
    USECOLS = [0, 1, 2, 3, 4, 7] + TC_IDX  # user_id, public, completion, gender, region, AGE + texte

    p = pd.read_csv(os.path.join(data_dir, 'soc-pokec-profiles.txt'), sep='\t', header=None,
                    na_values='null', usecols=USECOLS, low_memory=False)
    p.columns = ['user_id', 'public', 'comp', 'gender', 'region', 'age'] + [f't{i}' for i in range(30)]
    p = p[p['region'].astype(str).str.startswith(region)].copy()
    for c in ['gender', 'age', 'comp', 'public']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    p = p[p['gender'].isin([0, 1])].copy()
    p['gender'] = p['gender'].astype(int)
    p['age'] = p['age'].fillna(0).clip(0, 100)
    p['comp'] = p['comp'].fillna(0)
    p['public'] = p['public'].fillna(0)
    for i in range(30):
        p[f'h{i}'] = p[f't{i}'].notna().astype(int)
        p[f'l{i}'] = p[f't{i}'].fillna('').astype(str).str.len()
    FC = FEATURE_NAMES

    ids = sorted(p['user_id'].values)
    id2i = {u: i for i, u in enumerate(ids)}
    p['idx'] = p['user_id'].map(id2i)
    p = p.sort_values('idx').reset_index(drop=True)
    x = torch.tensor(StandardScaler().fit_transform(p[FC].values.astype(float)), dtype=torch.float)
    sens = torch.tensor(p['gender'].values, dtype=torch.long)

    e = pd.read_csv(os.path.join(data_dir, 'soc-pokec-relationships.txt'), sep='\t', header=None,
                    names=['s', 't'], dtype='int32')
    ns = set(p['user_id'].values)
    e = e[e['s'].isin(ns) & e['t'].isin(ns)]
    ei = to_undirected(torch.stack([torch.tensor(e['s'].map(id2i).values, dtype=torch.long),
                                     torch.tensor(e['t'].map(id2i).values, dtype=torch.long)]))
    data = Data(x=x, edge_index=ei, sens=sens)
    data.num_nodes = x.shape[0]
    return data, sens, p


# -- Training
def train_with_best(model, train_data, val_data, device, epochs=100, lr=0.01, weight_decay=5e-4,
                    eval_every=10, criterion=None, desc='Entraînement'):
    """Train link prediction model avec best-model checkpointing sur val AUC.

    Le modèle est restauré in-place avec le state_dict du meilleur checkpoint.

    Returns:
        best_auc: meilleur AUC de validation observé
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    xd = train_data.x.to(device)
    eid = train_data.edge_index.to(device)
    elid = train_data.edge_label_index.to(device)
    eld = train_data.edge_label.float().to(device)
    vx = val_data.x.to(device)
    vei = val_data.edge_index.to(device)
    veli = val_data.edge_label_index.to(device)
    vl = val_data.edge_label.numpy()

    best_auc, best_state = 0, None
    for ep in tqdm(range(1, epochs + 1), desc=desc):
        model.train()
        opt.zero_grad()
        loss = criterion(model(xd, eid, elid), eld)
        loss.backward()
        opt.step()
        if ep % eval_every == 0:
            model.eval()
            with torch.no_grad():
                va = roc_auc_score(vl, torch.sigmoid(model(vx, vei, veli)).cpu().numpy())
            if va > best_auc:
                best_auc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc


# -- Evaluation
def evaluate_link_pred(model, test_data, sens, device):
    """Evalue le modèle sur le test set et retourne tous les artefacts utiles.

    Returns:
        tp: predictions sigmoid (proba), shape (n,)
        tl: labels binaires, shape (n,)
        tpt: codes de paires (0..3), shape (n,)
        auc, ap, f1_global: métriques scalaires globales
        f: dict de fairness (sortie de fairness())
    """
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(test_data.x.to(device),
                                  test_data.edge_index.to(device),
                                  test_data.edge_label_index.to(device))).cpu().numpy()
    tl = test_data.edge_label.numpy()
    tpt = classify_pairs(test_data.edge_label_index, sens).numpy()
    auc = roc_auc_score(tl, tp)
    ap = average_precision_score(tl, tp)
    f = fairness(tp, tl, tpt)
    f1_global = f1_score(tl, (tp > 0.5).astype(float), zero_division=0)
    return tp, tl, tpt, auc, ap, f, f1_global


# -- Fairness
def classify_pairs(ei, sens):
    return sens[ei[0]] * 2 + sens[ei[1]]


def fairness(preds, labels, pt):
    """Toutes les métriques de fairness pour link prediction par type de paire.

    Args:
        preds: scores de prédiction (proba) shape (n,)
        labels: labels binaires shape (n,)
        pt: codes de paires shape (n,) avec valeurs dans {0,1,2,3}

    Returns:
        dict avec DP, EOpp, EO, DI (métriques canoniques),
        ps/pa/pf1 (par paire : score moyen, AUC, F1),
        ps_b (taux de pred binaires par paire),
        sf/sh/tf/th (par source/cible et version binaire _b).
    """
    bp = (preds > 0.5).astype(float)
    M = lambda m: bp[m].mean() if m.sum() else 0
    intra, inter = (pt == 0) | (pt == 3), (pt == 1) | (pt == 2)
    pr_i, pr_e = M(intra), M(inter)
    eopp = abs(M(intra & (labels == 1)) - M(inter & (labels == 1)))
    return {
        'DP': abs(pr_i - pr_e),
        'EOpp': eopp,
        'EO': eopp + abs(M(intra & (labels == 0)) - M(inter & (labels == 0))),
        'DI': min(pr_i, pr_e) / max(pr_i, pr_e) if max(pr_i, pr_e) > 0 else 1,
        # Scores raw (probas)  - pour chart descriptif
        'ps': {n: preds[pt == c].mean() for n, c in PAIRS},
        'pa': {n: roc_auc_score(labels[pt == c], preds[pt == c]) for n, c in PAIRS
               if (labels[pt == c] > 0).any() and (labels[pt == c] == 0).any()},
        # F1 par paire (performance threshold-based)
        'pf1': {n: f1_score(labels[pt == c], bp[pt == c], zero_division=0) for n, c in PAIRS if (pt == c).any()},
        'sf': preds[(pt == 0) | (pt == 1)].mean(), 'sh': preds[(pt == 2) | (pt == 3)].mean(),
        'tf': preds[(pt == 0) | (pt == 2)].mean(), 'th': preds[(pt == 1) | (pt == 3)].mean(),
        # Taux prédiction positive (binaire)  - pour dashboard DP cohérent
        'ps_b': {n: bp[pt == c].mean() for n, c in PAIRS},
        'sf_b': bp[(pt == 0) | (pt == 1)].mean(), 'sh_b': bp[(pt == 2) | (pt == 3)].mean(),
        'tf_b': bp[(pt == 0) | (pt == 2)].mean(), 'th_b': bp[(pt == 1) | (pt == 3)].mean(),
    }


# -- Probing
def probe_gender(z, sens, test_size=0.3, random_state=42):
    """Probe attaque pour évaluer si les embeddings encodent le genre.

    Returns:
        cos_fh: cosine similarity entre centroïdes F et H
        gen_acc_tr: accuracy train du classifieur linéaire
        gen_acc_te: accuracy test (mesure honnête de la généralisation)
        prop_majo: proportion de la classe majoritaire (baseline trivial)
    """
    sn = sens.numpy() if hasattr(sens, 'numpy') else np.asarray(sens)
    zn = z.numpy() if hasattr(z, 'numpy') else np.asarray(z)
    cos_fh = F.cosine_similarity(z[sn == 0].mean(0).unsqueeze(0),
                                  z[sn == 1].mean(0).unsqueeze(0)).item()

    ztr, zte, str_, ste = train_test_split(zn, sn, test_size=test_size,
                                            random_state=random_state, stratify=sn)
    clf = LogisticRegression(max_iter=500).fit(ztr, str_)
    gen_acc_tr = clf.score(ztr, str_)
    gen_acc_te = clf.score(zte, ste)
    prop_majo = max((sn == 0).mean(), (sn == 1).mean())
    return cos_fh, gen_acc_tr, gen_acc_te, prop_majo


# -- Distribution paires
def pair_distribution(tp, tl, tpt, threshold=0.5):
    pp_mask = (tp > threshold)
    pos_mask = (tl == 1)
    data_v = [100 * ((tpt == c) & pos_mask).sum() / pos_mask.sum() for _, c in PAIRS]
    pred_v = [100 * ((tpt == c) & pp_mask).sum() / pp_mask.sum() for _, c in PAIRS]
    diffs = [p - d for d, p in zip(data_v, pred_v)]
    return data_v, pred_v, diffs


# -- XAI
def saliency_link_pred(model, x, edge_index, edge_label_index, device):
    """Saliency par feature : moyenne de |grad_x score(u,v)| sur les paires fournies.

    Args:
        edge_label_index: tensor (2, n_pairs)  - paires sur lesquelles calculer

    Returns:
        np.array (n_features,)  - importance moyenne par feature
    """
    model.eval()
    x_in = x.to(device).detach().clone().requires_grad_(True)
    ei = edge_index.to(device)
    eli = edge_label_index.to(device)
    scores = torch.sigmoid(model(x_in, ei, eli))
    scores.sum().backward()
    grads = x_in.grad.abs()  # (n_nodes, n_features)
    pair_grads = grads[eli[0]] + grads[eli[1]]  # (n_pairs, n_features)
    return pair_grads.mean(dim=0).cpu().numpy()


def select_extreme_pairs(tp, tl, tpt, target_code=3):
    """Sélectionne 1 paire bien prédite et 1 paire mal prédite parmi les vraies amitiés du type ciblé.

    Args:
        target_code: 0=F->F, 1=F->H, 2=H->F, 3=H->H (défaut : H->H)

    Returns:
        (idx_well, score_well, idx_mis, score_mis)
    """
    mask = (tpt == target_code) & (tl == 1)
    indices = np.where(mask)[0]
    scores = tp[indices]
    best = int(indices[np.argmax(scores)])
    worst = int(indices[np.argmin(scores)])
    return best, float(tp[best]), worst, float(tp[worst])


class _LinkPredWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_label_index):
        return self.model(x, edge_index, edge_label_index)


def explain_pair_gnn(model, x, edge_index, edge_label_index, pair_idx, device, epochs=200):
    """Lance GNNExplainer sur une paire et retourne l'importance par feature (mean des 2 nœuds).

    Returns:
        np.array (n_features,)  - importance par feature pour la paire (u, v)
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    set_seed(42)
    wrapped = _LinkPredWrapper(model).to(device)
    wrapped.eval()
    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=epochs, lr=0.01),
        explanation_type='model',
        edge_mask_type='object',
        node_mask_type='attributes',
        model_config=dict(mode='binary_classification', task_level='edge', return_type='raw'),
    )
    pair = edge_label_index[:, pair_idx:pair_idx + 1].to(device)
    explanation = explainer(x=x.to(device), edge_index=edge_index.to(device), edge_label_index=pair)
    node_mask = explanation.node_mask.detach().cpu().numpy()  # (n_nodes, n_features)
    u, v = int(pair[0, 0]), int(pair[1, 0])
    return (node_mask[u] + node_mask[v]) / 2


def fidelity_pos_neg(model, test_data, sens, importance, device, top_k=10):
    """Fidelity+ (sufficiency) et Fidelity- (necessity) en masquant les features.

    Args:
        importance: dict avec clés 'intra' et 'inter' (chacune np.array de taille n_features)

    Returns:
        dict {'intra': {auc_full, auc_only_topk, auc_without_topk}, 'inter': {...}}
    """
    model.eval()
    x = test_data.x.to(device)
    ei = test_data.edge_index.to(device)
    eli = test_data.edge_label_index.to(device)
    tl = test_data.edge_label.numpy()
    tpt = classify_pairs(test_data.edge_label_index, sens).numpy()
    mean_per_feat = x.mean(dim=0)

    with torch.no_grad():
        tp_full = torch.sigmoid(model(x, ei, eli)).cpu().numpy()

    results = {}
    for kind, codes in [('intra', [0, 3]), ('inter', [1, 2])]:
        mask = np.isin(tpt, codes)
        topk_idx = np.argsort(importance[kind])[-top_k:]
        keep = torch.zeros(x.shape[1], dtype=torch.bool, device=device)
        keep[topk_idx] = True
        x_only, x_drop = x.clone(), x.clone()
        x_only[:, ~keep] = mean_per_feat[~keep]   # Fid+ : garde top-K seulement
        x_drop[:, keep] = mean_per_feat[keep]     # Fid- : masque top-K
        with torch.no_grad():
            tp_only = torch.sigmoid(model(x_only, ei, eli)).cpu().numpy()
            tp_drop = torch.sigmoid(model(x_drop, ei, eli)).cpu().numpy()
        results[kind] = {
            'auc_full': roc_auc_score(tl[mask], tp_full[mask]),
            'auc_only_topk': roc_auc_score(tl[mask], tp_only[mask]),
            'auc_without_topk': roc_auc_score(tl[mask], tp_drop[mask]),
        }
    return results


def plot_top_features(importances, feature_names=None, top_k=10, gender_idx=3, figsize=(15, 5)):
    if feature_names is None:
        feature_names = FEATURE_NAMES
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    items = list(importances.items())
    for i, (lbl, imp) in enumerate(items):
        topk_idx = np.argsort(imp)[-top_k:]
        topk_names = [feature_names[j] for j in topk_idx]
        topk_vals = imp[topk_idx]
        colors = ['#d62728' if j == gender_idx else '#2196F3' for j in topk_idx]
        bars = ax[i].barh(topk_names, topk_vals, color=colors, edgecolor='black')
        for b, v in zip(bars, topk_vals):
            ax[i].text(v + max(topk_vals) * .02, b.get_y() + b.get_height() / 2,
                       f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
        ax[i].set(xlabel='Importance', title=lbl)
    fig.suptitle(f'Top-{top_k} features  (rouge = gender)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig, ax


def shap_link_pred(model, x, edge_index, ref_node, target_node, device,
                   n_background=20, n_explain=30, nsamples=64, random_state=42):
    """SHAP KernelExplainer pour link prediction.

    Pour la paire (ref_node, target_node), perturbe les features de ref_node
    et mesure l'impact sur score(ref_node, target_node).

    Returns:
        mean_abs: np.array (n_features,)  - importance moyenne par feature
        ranking:  np.array (n_features,)  - indices triés par importance décroissante
    """
    import shap
    set_seed(random_state)

    x_dev = x.to(device)
    edge_index_dev = edge_index.to(device)
    pair = torch.tensor([[ref_node], [target_node]], dtype=torch.long).to(device)
    model.eval()

    def shap_predict(features):
        """Score de reco quand on remplace les features de ref_node par chaque ligne de `features`."""
        features_t = torch.tensor(features, dtype=torch.float).to(device)
        scores = np.zeros(len(features))
        for i in range(len(features)):
            xm = x_dev.clone()
            xm[ref_node] = features_t[i]
            with torch.no_grad():
                pred = torch.sigmoid(model(xm, edge_index_dev, pair))
            scores[i] = pred.item()
        return scores

    rng = np.random.default_rng(random_state)
    n_nodes = x.shape[0]
    bg_idx = rng.choice(n_nodes, n_background, replace=False)
    explain_idx = rng.choice(n_nodes, n_explain, replace=False)
    X_bg = x[bg_idx].cpu().numpy()
    X_explain = x[explain_idx].cpu().numpy()

    explainer = shap.KernelExplainer(shap_predict, X_bg)
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = np.argsort(mean_abs)[::-1]
    return mean_abs, ranking


def plot_shap_topk(mean_abs, ranking, feature_names=None, gender_idx=3, top_k=20, figsize=(10, 7)):
    if feature_names is None:
        feature_names = FEATURE_NAMES
    top_idx = ranking[:top_k]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]
    colors = ['#d62728' if i == gender_idx else '#3498db' for i in top_idx]
    gender_rank = int(np.where(ranking == gender_idx)[0][0]) + 1

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(top_k), top_vals[::-1], color=colors[::-1], edgecolor='black')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set(xlabel='Mean |SHAP value|',
           title=f'Top {top_k} features SHAP')
    gender_pos = top_k - 1 - list(top_idx).index(gender_idx) if gender_idx in top_idx else None
    if gender_pos is not None:
        ax.annotate(f'rang {gender_rank}/{len(feature_names)}',
                    xy=(top_vals[list(top_idx).index(gender_idx)], gender_pos),
                    xytext=(15, 0), textcoords='offset points',
                    fontsize=9, color='#d62728', fontweight='bold')
    plt.tight_layout()
    return fig, ax, gender_rank


def counterfactual_shift(model, x, edge_index, sens, device, gender_idx=3, n_samples=500, random_state=42):
    """Counterfactual probing : flipper la feature `gender` et mesurer le shift de l'embedding.

    Si le modèle est désentangled, ||z_orig - z_cf|| ~ 0 et cos(z_orig, z_cf) ~ 1.

    Returns:
        shifts:    np.array (n_samples,)  - distance L2 entre z_orig et z_cf
        cosines:   np.array (n_samples,)  - cosine similarity entre z_orig et z_cf
        sens_idx:  np.array (n_samples,)  - genre original de chaque nœud sample (0=F, 1=H)
    """
    set_seed(random_state)
    model.eval()
    rng = np.random.default_rng(random_state)
    indices = rng.choice(x.shape[0], min(n_samples, x.shape[0]), replace=False)

    x_dev = x.to(device)
    edge_index_dev = edge_index.to(device)
    sens_np = sens.numpy() if hasattr(sens, 'numpy') else np.asarray(sens)

    # Valeurs z-scored moyennes de gender pour chaque genre (pour le swap)
    val_F = x_dev[sens_np == 0, gender_idx].mean().item()
    val_H = x_dev[sens_np == 1, gender_idx].mean().item()

    # Embeddings originaux
    with torch.no_grad():
        z_orig = model.encode(x_dev, edge_index_dev)

    # Counterfactual : flipper le gender pour les nœuds sample
    x_cf = x_dev.clone()
    for i in indices:
        x_cf[i, gender_idx] = val_H if sens_np[i] == 0 else val_F

    with torch.no_grad():
        z_cf = model.encode(x_cf, edge_index_dev)

    z_o = z_orig[indices].cpu().numpy()
    z_c = z_cf[indices].cpu().numpy()

    shifts = np.linalg.norm(z_o - z_c, axis=1)
    norms_o = np.linalg.norm(z_o, axis=1) + 1e-12
    norms_c = np.linalg.norm(z_c, axis=1) + 1e-12
    cosines = (z_o * z_c).sum(axis=1) / (norms_o * norms_c)
    return shifts, cosines, sens_np[indices]


def plot_counterfactual_shift(shifts, cosines, sens_idx, figsize=(13, 5)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].hist(shifts[sens_idx == 0], bins=40, alpha=0.65, label='Source = F', color=C_INTER, edgecolor='black')
    ax[0].hist(shifts[sens_idx == 1], bins=40, alpha=0.65, label='Source = H', color=C_INTRA, edgecolor='black')
    ax[0].axvline(shifts.mean(), color='red', ls='--', lw=1.5, label=f'moyenne = {shifts.mean():.3f}')
    ax[0].axvline(0, color='green', ls=':', lw=1.5, label='Idéal : 0 (invariant)')
    ax[0].set(xlabel='||z_orig - z_cf||', ylabel='Nombre de nœuds',
              title='Shift d\'embedding après flip de gender (distance L2)')
    ax[0].legend()

    ax[1].hist(cosines[sens_idx == 0], bins=40, alpha=0.65, label='Source = F', color=C_INTER, edgecolor='black')
    ax[1].hist(cosines[sens_idx == 1], bins=40, alpha=0.65, label='Source = H', color=C_INTRA, edgecolor='black')
    ax[1].axvline(1.0, color='green', ls='--', lw=1.5, label='Idéal : 1 (invariant)')
    ax[1].set(xlabel='cos(z_orig, z_cf)', ylabel='Nombre de nœuds',
              title='Similarité cosine après flip de gender')
    ax[1].legend()
    plt.tight_layout()
    return fig, ax


# -- Bias detection
def plot_score_distribution_by_pair(tp, tl, tpt, figsize=(13, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    pos = (tl == 1)
    palette = {'F->F': '#2196F3', 'F->H': '#FF9800', 'H->F': '#9467bd', 'H->H': '#d62728'}
    for n, c in PAIRS:
        mask = pos & (tpt == c)
        if mask.sum() == 0:
            continue
        scores = tp[mask]
        ax.hist(scores, bins=40, alpha=0.55, label=f'{n} (n={len(scores)}, mean={scores.mean():.3f})',
                color=palette[n], edgecolor='black', linewidth=0.4)
    ax.axvline(0.5, color='black', ls='--', lw=1.5, label='Seuil 0.5')
    ax.set(xlabel='Score prédit (sigmoid)', ylabel='Nombre de vrais positifs',
           title='Distribution des scores prédits par type de paire (vrais positifs)')
    ax.legend(fontsize=9); plt.tight_layout()
    return fig, ax


def top_k_bias(tp, tl, tpt, k=100):
    """Top-K liens les plus probables : quel % est intra-genre ? Comparé au % réel.

    Returns:
        dict {k, top_intra_pct, top_inter_pct, data_intra_pct, amplification_pp}
    """
    idx_top = np.argsort(tp)[-k:]
    top_codes = tpt[idx_top]
    top_intra = ((top_codes == 0) | (top_codes == 3)).mean() * 100
    top_inter = ((top_codes == 1) | (top_codes == 2)).mean() * 100
    pos = (tl == 1)
    data_codes = tpt[pos]
    data_intra = ((data_codes == 0) | (data_codes == 3)).mean() * 100
    return {
        'k': k,
        'top_intra_pct': top_intra,
        'top_inter_pct': top_inter,
        'data_intra_pct': data_intra,
        'amplification_pp': top_intra - data_intra,
    }


def plot_top_k_bias(results, figsize=(8, 4)):
    if isinstance(results, dict):
        results = [results]
    labels = ['Réalité\n(vrais positifs)']
    values = [results[0]['data_intra_pct']]
    for r in results:
        labels.append(f'Top-{r["k"]}\nprédictions')
        values.append(r['top_intra_pct'])
    colors = ['#1f77b4'] + ['#d62728'] * len(results)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=colors, edgecolor='black')
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}%', ha='center', fontweight='bold')
    ax.axhline(values[0], color='red', ls='--', lw=1.5, alpha=0.7, label=f'Réalité = {values[0]:.1f}% intra')
    ax.set(ylabel='% intra-genre', ylim=(0, 105),
           title='Top-K bias : amplification de l\'homophilie dans les prédictions confiantes')
    ax.legend(); plt.tight_layout()
    return fig, ax


def plot_embeddings_2d(z, sens, method='tsne', n_sample=2000, random_state=42, figsize=(8, 7)):
    sn = sens.numpy() if hasattr(sens, 'numpy') else np.asarray(sens)
    zn = z.numpy() if hasattr(z, 'numpy') else np.asarray(z)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(zn), min(n_sample, len(zn)), replace=False)
    zs, ss = zn[indices], sn[indices]

    used = method
    if method == 'umap':
        try:
            import umap
            z_2d = umap.UMAP(n_components=2, random_state=random_state).fit_transform(zs)
        except ImportError:
            used = 'tsne'
    if used == 'tsne':
        from sklearn.manifold import TSNE
        z_2d = TSNE(n_components=2, random_state=random_state, perplexity=30, init='pca').fit_transform(zs)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(z_2d[ss == 0, 0], z_2d[ss == 0, 1], c='#FF9800', alpha=0.5, s=12, label='Femme', edgecolor='none')
    ax.scatter(z_2d[ss == 1, 0], z_2d[ss == 1, 1], c='#2196F3', alpha=0.5, s=12, label='Homme', edgecolor='none')
    ax.set(title=f'Embeddings GCN colorés par genre ({used.upper()}, n={len(zs)})',
           xlabel=f'{used.upper()} dim 1', ylabel=f'{used.upper()} dim 2')
    ax.legend(); plt.tight_layout()
    return fig, ax


def compare_feature_ranks(importances_dict, feature_indices=(1, 3), feature_names=None):
    """Pour chaque condition, retourne le rang et la valeur des features ciblées.

    Args:
        importances_dict: {label: importance_array}
        feature_indices: indices à tracker (défaut : age=1, gender=3)

    Returns:
        pd.DataFrame avec colonnes (feature, condition, rank, value)
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    rows = []
    for label, imp in importances_dict.items():
        ranking = np.argsort(imp)[::-1]
        for idx in feature_indices:
            rank = int(np.where(ranking == idx)[0][0]) + 1
            rows.append({'feature': feature_names[idx], 'condition': label,
                         'rank': rank, 'value': float(imp[idx])})
    return pd.DataFrame(rows)


def counterfactual_collective(model, x, edge_index, sens, device, gender_idx=3, random_state=42):
    """Counterfactual collectif : flipper TOUS les genres en même temps.
    Compare au counterfactual local (1 nœud)  - révèle le paradoxe local/global.

    Returns:
        (shift_collective_mean, cosine_collective_mean) sur tous les nœuds
    """
    set_seed(random_state)
    model.eval()
    x_dev = x.to(device)
    edge_index_dev = edge_index.to(device)
    sens_np = sens.numpy() if hasattr(sens, 'numpy') else np.asarray(sens)

    val_F = x_dev[sens_np == 0, gender_idx].mean().item()
    val_H = x_dev[sens_np == 1, gender_idx].mean().item()

    with torch.no_grad():
        z_orig = model.encode(x_dev, edge_index_dev)

    x_cf = x_dev.clone()
    x_cf[sens_np == 0, gender_idx] = val_H
    x_cf[sens_np == 1, gender_idx] = val_F

    with torch.no_grad():
        z_cf = model.encode(x_cf, edge_index_dev)

    z_o = z_orig.cpu().numpy()
    z_c = z_cf.cpu().numpy()
    shifts = np.linalg.norm(z_o - z_c, axis=1)
    norms_o = np.linalg.norm(z_o, axis=1) + 1e-12
    norms_c = np.linalg.norm(z_c, axis=1) + 1e-12
    cosines = (z_o * z_c).sum(axis=1) / (norms_o * norms_c)
    return float(shifts.mean()), float(cosines.mean())


def plot_fidelity(fid_results, figsize=(11, 4)):
    types = list(fid_results.keys())
    full_v = [fid_results[t]['auc_full'] for t in types]
    only_v = [fid_results[t]['auc_only_topk'] for t in types]
    drop_v = [fid_results[t]['auc_without_topk'] for t in types]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(types))
    w = 0.27
    b1 = ax.bar(x - w, full_v, w, label='AUC complète',                color='#2ca02c', edgecolor='black')
    b2 = ax.bar(x,     only_v, w, label='AUC top-K seulement (Fid+)',   color='#1f77b4', edgecolor='black')
    b3 = ax.bar(x + w, drop_v, w, label='AUC sans top-K (Fid-)',        color='#d62728', edgecolor='black')
    for bars, vals in [(b1, full_v), (b2, only_v), (b3, drop_v)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                    ha='center', fontweight='bold', fontsize=9)
    ax.set_xticks(x, [f'{t.title()}-genre' for t in types])
    ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5, label='Hasard (0.5)')
    ax.set(ylabel='AUC', ylim=(0, 1.1),
           title='Fidelity+/- : les top-K features sont-elles vraiment causales ?')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig, ax


# -- Viz
pair_color = lambda n: C_INTRA if n in ('F->F', 'H->H') else C_INTER


def annot_bars(ax, bars, vals=None, fmt='{:.3f}', dy=0.015, fontsize=10):
    vals = [b.get_height() for b in bars] if vals is None else vals
    for b, v in zip(bars, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(v), ha='center', fontweight='bold', fontsize=fontsize)


def bar_per_pair(ax, vals_dict, title, ylabel, hline=None, hline_label=None, ylim=(0, 1), extras=None):
    vals = [vals_dict.get(n, np.nan) for n in PAIR_ORDER]
    bars = ax.bar(PAIR_ORDER, vals, color=[pair_color(n) for n in PAIR_ORDER], edgecolor='black')
    annot_bars(ax, bars, vals)
    has_legend = False
    if hline is not None:
        ax.axhline(hline, color='red', ls='--', lw=1.5, label=hline_label or f'Référence = {hline:.3f}')
        has_legend = True
    if extras:
        for v, lbl, st in extras:
            ax.axhline(v, label=lbl, **{**{'color': 'gray', 'ls': ':', 'lw': 1, 'alpha': 0.7}, **(st or {})})
        has_legend = True
    ax.set(ylabel=ylabel, title=title, ylim=ylim)
    if has_legend:
        ax.legend(loc='upper right', fontsize=9)
    return bars


def barh_dashboard(metrics, title='Dashboard', xlabel='Écart', threshold_lines=None, figsize=(12, 4)):
    metrics_sorted = sorted(metrics, key=lambda m: m[1])
    names, vals, thrs = zip(*metrics_sorted)
    colors = ['#d62728' if v > t else '#2ca02c' for v, t in zip(vals, thrs)]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, vals, color=colors, edgecolor='black', height=0.6)
    xmax = max(max(vals), max(thrs)) * 1.7
    for b, v, t in zip(bars, vals, thrs):
        ax.text(v + xmax * .015, b.get_y() + b.get_height() / 2,
                f'{v:.3f}  (seuil {t:.2f})', va='center', fontweight='bold', fontsize=10)
    ax.axvline(0, color='green', lw=2.5, label='Idéal (0)')
    if threshold_lines:
        for v, lbl, c in threshold_lines:
            ax.axvline(v, color=c, ls='--', lw=1.5, alpha=0.7, label=lbl)
    ax.set(xlabel=xlabel, xlim=(0, xmax), title=title)
    ax.legend(loc='lower right', fontsize=9)
    return fig, ax


def fairness_legend(fig, loc='lower center', bbox=(0.5, -0.02)):
    fig.legend(handles=[Patch(facecolor=C_INTRA, edgecolor='black', label='Intra-genre (F->F, H->H)'),
                        Patch(facecolor=C_INTER, edgecolor='black', label='Inter-genre (F->H, H->F)')],
               loc=loc, ncol=2, bbox_to_anchor=bbox, fontsize=10)


def plot_representation_bias(cos_fh, gen_acc_tr, gen_acc_te, prop_majo, figsize=(15, 5)):
    """Affiche les 2 subplots de Representation Bias : cosine F/H + probing genre."""
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=.35)

    # Cosine F vs H (idéal fairness = centroïdes confondus, cos = 1)
    ax[0].bar(['Centroïdes F vs H'], [cos_fh], color='#9467bd', edgecolor='black', width=0.4)
    ax[0].axhline(1, color='green', ls='--', lw=2.5, label='Idéal fairness : centroïdes confondus (cos = 1)')
    ax[0].axhline(0, color='gray',  ls=':',  lw=1.5, label='Centroïdes orthogonaux (cos = 0)')
    ax[0].text(0, cos_fh + 0.04, f'{cos_fh:.3f}', ha='center', fontweight='bold', fontsize=14)
    ax[0].set(ylabel='Cosine similarity', ylim=(min(-0.1, cos_fh - 0.1), 1.15),
              title='Cosine entre centroïdes embeddings F et H')
    ax[0].legend(loc='lower right', fontsize=9)

    # Probing accuracy
    b1 = ax[1].bar(['Train', 'Test'], [gen_acc_tr, gen_acc_te],
                   color=['#7f7f7f', '#9467bd'], edgecolor='black', width=0.5)
    ax[1].axhline(prop_majo, color='gray',  ls=':',  lw=1.5, label=f'Baseline majoritaire ({prop_majo:.3f})')
    ax[1].axhline(0.5,       color='green', ls='--', lw=2,   label='Hasard pur (0.5)')
    ax[1].axhline(1.0,       color='red',   ls=':',  lw=1.5, label='Parfaitement identifiable (1.0)')
    annot_bars(ax[1], b1)
    margin = gen_acc_te - prop_majo
    ax[1].annotate(f'+{100*margin:.0f} pts vs majoritaire',
                   xy=(1, gen_acc_te + 0.06), fontsize=10, color='#d62728',
                   ha='center', va='bottom')
    ax[1].set(ylabel='Accuracy du probing', ylim=(0.4, 1.05),
              title='Probing : prédire le genre depuis les embeddings')
    ax[1].legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_pair_distribution(data_v, pred_v, diffs, figsize=(13, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    x, w = np.arange(4), 0.35
    b1 = ax.bar(x - w / 2, data_v, w, label='Données (vraies amitiés)',       color='#1f77b4', edgecolor='black')
    b2 = ax.bar(x + w / 2, pred_v, w, label='Modèle (prédictions positives)', color='#ff7f0e', edgecolor='black')
    [ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
     for b, v in list(zip(b1, data_v)) + list(zip(b2, pred_v))]
    [ax.text(i, -3, f'{d:+.1f} pp', ha='center', fontsize=10,
             color='#d62728' if abs(d) > 3 else '#2ca02c', fontweight='bold')
     for i, d in enumerate(diffs)]
    ax.set_xticks(x, PAIR_ORDER)
    ax.axhline(25, color='gray', ls=':', lw=1, alpha=0.5, label='Uniforme (25%)')
    ax.set(ylabel='% des liens',
           xlabel='Type de paire   - delta (modèle - données) en points de % sous chaque type',
           title='Distribution des paires : données réelles vs prédictions du modèle',
           ylim=(-7, max(data_v + pred_v) * 1.18))
    total_intra = sum(d for d in diffs if d > 0)
    ax.text(0.03, 0.95, f'Intra-genre : +{total_intra:.1f} pp',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=11, bbox=dict(boxstyle='round', fc='#fff3cd', ec='orange'))
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    return fig, ax


# -- Multi-architecture
class SGC(nn.Module):
    """Simple Graph Convolution (Wu et al., 2019)  - GCN sans non-linéarité."""
    def __init__(self, inc, out=64, K=2):
        super().__init__()
        from torch_geometric.nn import SGConv
        self.conv = SGConv(inc, out, K=K)

    def encode(self, x, ei): return self.conv(x, ei)
    def decode(self, z, ei): return (z[ei[0]] * z[ei[1]]).sum(dim=-1)
    def forward(self, x, ei, eli): return self.decode(self.encode(x, ei), eli)


# -- Correction Fairness
def train_fair(model, train_data, val_data, sens, device,
               epochs=100, lr=0.01, weight_decay=5e-4, eval_every=10,
               reweight=False, lam_dp=0.0, desc='Entraînement'):
    """Entraînement GCN avec reweighting et/ou régularisation DP (from scratch).

    Args:
        reweight: pondère la loss par type de paire (w = N_total / 2*N_type)
        lam_dp: coefficient lambda de régularisation DP (0 = désactivé)
    """
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    xd = train_data.x.to(device)
    eid = train_data.edge_index.to(device)
    elid = train_data.edge_label_index.to(device)
    eld = train_data.edge_label.float().to(device)
    pt_train = classify_pairs(train_data.edge_label_index, sens).to(device)
    intra_mask = (pt_train == 0) | (pt_train == 3)
    inter_mask = (pt_train == 1) | (pt_train == 2)

    weights = torch.ones(len(eld), device=device)
    if reweight:
        n_intra, n_inter = intra_mask.sum().item(), inter_mask.sum().item()
        weights[intra_mask] = len(eld) / (2 * n_intra)
        weights[inter_mask] = len(eld) / (2 * n_inter)

    vx, vei = val_data.x.to(device), val_data.edge_index.to(device)
    veli, vl = val_data.edge_label_index.to(device), val_data.edge_label.numpy()
    bce = nn.BCEWithLogitsLoss(reduction='none')
    best_auc, best_state = 0, None

    for ep in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        model.train(); opt.zero_grad()
        logits = model(xd, eid, elid)
        task_loss = (bce(logits, eld) * weights).mean()
        if lam_dp > 0:
            scores = torch.sigmoid(logits)
            dp_penalty = torch.abs(scores[intra_mask].mean() - scores[inter_mask].mean())
            loss = task_loss + lam_dp * dp_penalty
        else:
            loss = task_loss
        loss.backward(); opt.step()
        if ep % eval_every == 0:
            model.eval()
            with torch.no_grad():
                va = roc_auc_score(vl, torch.sigmoid(model(vx, vei, veli)).cpu().numpy())
            if va > best_auc:
                best_auc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc


class GenderDiscriminator(nn.Module):
    """Adversaire qui prédit le genre depuis les embeddings (pour FairGNN)."""
    def __init__(self, in_dim, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def train_fairgnn(model, train_data, val_data, sens, device,
                  alpha=0.5, epochs=100, lr=0.01, weight_decay=5e-4, eval_every=10,
                  desc='FairGNN'):
    """Entraînement adversarial (FairGNN, Dai & Wang 2021, from scratch).

    Le GCN est entraîné contre un adversaire qui prédit le genre depuis z.
    Loss = L_task - alpha * L_disc (le GCN essaie de tromper le discriminateur).
    """
    disc = GenderDiscriminator(64).to(device)
    opt_enc = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_disc = Adam(disc.parameters(), lr=lr)

    xd = train_data.x.to(device)
    eid = train_data.edge_index.to(device)
    elid = train_data.edge_label_index.to(device)
    eld = train_data.edge_label.float().to(device)
    sens_d = sens.float().to(device)
    vx, vei = val_data.x.to(device), val_data.edge_index.to(device)
    veli, vl = val_data.edge_label_index.to(device), val_data.edge_label.numpy()

    bce_task = nn.BCEWithLogitsLoss()
    bce_disc = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    for ep in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        model.train(); disc.train()
        z = model.encode(xd, eid)

        # Discriminateur : apprend à prédire le genre
        opt_disc.zero_grad()
        disc_loss = bce_disc(disc(z.detach()), sens_d)
        disc_loss.backward(); opt_disc.step()

        # Encodeur : minimise L_task - alpha * L_disc (trompe le discriminateur)
        opt_enc.zero_grad()
        logits = model.decode(z, elid)
        task_loss = bce_task(logits, eld)
        adv_loss = bce_disc(disc(z), sens_d)
        loss = task_loss - alpha * adv_loss
        loss.backward(); opt_enc.step()

        if ep % eval_every == 0:
            model.eval()
            with torch.no_grad():
                va = roc_auc_score(vl, torch.sigmoid(model(vx, vei, veli)).cpu().numpy())
            if va > best_auc:
                best_auc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc


def collect_results(model, test_data, sens, device):
    """Évalue un modèle et collecte toutes les métriques (performance + fairness + leakage)."""
    tp, tl, tpt, auc, ap, f, f1g = evaluate_link_pred(model, test_data, sens, device)
    model.eval()
    with torch.no_grad():
        z = model.encode(test_data.x.to(device), test_data.edge_index.to(device)).cpu()
    cos_fh, _, acc_te, prop_majo = probe_gender(z, sens)
    return {'auc': auc, 'ap': ap, 'f1': f1g, 'dp': f['DP'], 'eopp': f['EOpp'],
            'eo': f['EO'], 'di': f['DI'], 'cos_fh': cos_fh, 'leakage': acc_te - prop_majo,
            'tp': tp, 'tl': tl, 'tpt': tpt, 'f': f}


def plot_pareto(results_dict, figsize=(10, 6)):
    """Courbe de Pareto AUC vs DP gap."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, r in results_dict.items():
        marker = 'o' if 'Baseline' in name else ('s' if 'Reweight' in name else ('^' if 'DP' in name else 'D'))
        color = '#d62728' if r['dp'] > 0.05 else '#2ca02c'
        ax.scatter(r['dp'], r['auc'], s=120, marker=marker, color=color, edgecolor='black', zorder=5)
        ax.annotate(name, (r['dp'], r['auc']), fontsize=8, textcoords='offset points', xytext=(5, 5))
    ax.axvline(0.05, color='orange', ls='--', lw=1.5, label='Seuil fair (DP=0.05)')
    fair = {n: r for n, r in results_dict.items() if r['dp'] < 0.05}
    if fair:
        best_n = max(fair, key=lambda n: fair[n]['auc'])
        b = fair[best_n]
        ax.annotate('meilleur compromis', xy=(b['dp'], b['auc']),
                    xytext=(b['dp'] + 0.005, b['auc'] - 0.008),
                    fontsize=9, ha='left', va='top',
                    bbox=dict(boxstyle='round', fc='#d4edda', ec='green', alpha=0.9))
    ax.set(xlabel='DP gap (plus bas = plus fair)', ylabel='AUC (plus haut = plus performant)',
           title='Courbe de Pareto : Performance vs Fairness')
    ax.legend(); plt.tight_layout()
    return fig, ax


# -- Validation & Incertitude
def mc_dropout_predict(model, test_data, device, T=30):
    """MC Dropout : T forward passes avec dropout activé.

    Returns:
        p_bar: prédiction moyenne (n_edges,)
        entropy: entropie prédictive H = -p log p - (1-p) log(1-p)
        variance: variance des prédictions
    """
    model.train()
    x, ei = test_data.x.to(device), test_data.edge_index.to(device)
    eli = test_data.edge_label_index.to(device)
    all_probs = []
    with torch.no_grad():
        for _ in range(T):
            probs = torch.sigmoid(model(x, ei, eli)).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.stack(all_probs)  # (T, n_edges)
    p_bar = all_probs.mean(axis=0)
    variance = all_probs.var(axis=0)
    eps = 1e-10
    entropy = -(p_bar * np.log(p_bar + eps) + (1 - p_bar) * np.log(1 - p_bar + eps))
    model.eval()
    return p_bar, entropy, variance


def compute_ece(preds, labels, n_bins=15):
    """Expected Calibration Error.

    Returns:
        ece: float, calibration error (0 = parfait)
        bin_accs, bin_confs, bin_counts: arrays pour reliability diagram
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0); bin_confs.append(0); bin_counts.append(0)
            continue
        bin_accs.append(labels[mask].mean())
        bin_confs.append(preds[mask].mean())
        bin_counts.append(mask.sum())
    bin_accs, bin_confs, bin_counts = np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)
    ece = (bin_counts / bin_counts.sum() * np.abs(bin_accs - bin_confs)).sum()
    return ece, bin_accs, bin_confs, bin_counts


def plot_reliability(models_preds, n_bins=15, figsize=(15, 4)):
    """Reliability diagrams pour plusieurs modèles.

    Args:
        models_preds: dict {name: (preds, labels)}
    """
    n = len(models_preds)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1: axes = [axes]
    for ax, (name, (preds, labels)) in zip(axes, models_preds.items()):
        ece, accs, confs, counts = compute_ece(preds, labels, n_bins)
        bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
        ax.bar(bin_centers, accs, width=1 / n_bins, alpha=0.7, color='#1f77b4', edgecolor='black', label='Accuracy')
        ax.plot([0, 1], [0, 1], 'r--', label='Calibration parfaite')
        ax.set(xlabel='Confiance', ylabel='Accuracy', title=f'{name}\nECE={ece:.3f}', xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def perturb_and_eval(model, test_data, sens, device, sigmas=(0.1, 0.5, 1.0)):
    """Évalue le modèle sous bruit gaussien sur les features.

    Returns:
        list of dicts {sigma, auc, dp}
    """
    results = []
    for sigma in sigmas:
        x_noisy = test_data.x + torch.randn_like(test_data.x) * sigma
        model.eval()
        with torch.no_grad():
            tp = torch.sigmoid(model(x_noisy.to(device), test_data.edge_index.to(device),
                                      test_data.edge_label_index.to(device))).cpu().numpy()
        tl = test_data.edge_label.numpy()
        tpt = classify_pairs(test_data.edge_label_index, sens).numpy()
        auc = roc_auc_score(tl, tp)
        f = fairness(tp, tl, tpt)
        results.append({'sigma': sigma, 'auc': auc, 'dp': f['DP']})
    return results


def plot_radar(results_dict, figsize=(7, 7)):
    """Radar plot 5-axes : Performance, Fairness, Robustesse, Calibration, Certitude.

    Args:
        results_dict: dict {name: {perf, fairness, robustness, calibration, certainty}} (valeurs 0-1)
    """
    categories = ['Performance', 'Fairness', 'Robustesse', 'Calibration', 'Certitude']
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    for i, (name, vals) in enumerate(results_dict.items()):
        values = [vals[k] for k in ['perf', 'fairness', 'robustness', 'calibration', 'certainty']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Synthèse 3D : Performance / Fairness / Robustesse', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    return fig, ax


# -- Verdicts
verdict_str = lambda v, t, op='lt': 'fair' if (op == 'lt' and v <= t) or (op == 'ge' and v >= t) or (op == 'gt' and v > t) else 'biaisé'
verdict_mark = lambda v, t, op='lt': 'OK' if (op == 'lt' and v <= t) or (op == 'ge' and v >= t) or (op == 'gt' and v > t) else 'KO'
