"""Helpers réutilisables pour 02_pipeline.ipynb (Responsible AI sur Pokec).

Importé via `from ra_utils import *` au début du notebook.

Sections :
    - Constantes      : PAIR_ORDER, PAIRS, C_INTRA, C_INTER
    - Setup           : set_seed
    - Data            : load_pokec
    - Training        : train_with_best
    - Evaluation      : evaluate_link_pred
    - Fairness        : classify_pairs, fairness
    - Probing         : probe_gender
    - Pair distrib    : pair_distribution
    - Viz génériques  : pair_color, annot_bars, bar_per_pair, barh_dashboard, fairness_legend
    - Viz spécifiques : plot_representation_bias, plot_pair_distribution
    - Verdicts        : verdict_str, verdict_mark
"""
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


# === Constantes ===
PAIR_ORDER = ['F->F', 'F->H', 'H->F', 'H->H']
PAIRS = [(name, idx) for idx, name in enumerate(PAIR_ORDER)]
C_INTRA, C_INTER = '#2196F3', '#FF9800'


# === Setup ===
def set_seed(s=42):
    """Seed all random sources for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# === Data ===
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
    FC = ['comp', 'age', 'public', 'gender'] + [f'h{i}' for i in range(30)] + [f'l{i}' for i in range(30)]

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


# === Training ===
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


# === Evaluation ===
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


# === Fairness ===
def classify_pairs(ei, sens):
    """Encode chaque arête par le code de paire (0=F->F, 1=F->H, 2=H->F, 3=H->H)."""
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
        # Scores raw (probas) — pour chart descriptif
        'ps': {n: preds[pt == c].mean() for n, c in PAIRS},
        'pa': {n: roc_auc_score(labels[pt == c], preds[pt == c]) for n, c in PAIRS
               if (labels[pt == c] > 0).any() and (labels[pt == c] == 0).any()},
        # F1 par paire (performance threshold-based)
        'pf1': {n: f1_score(labels[pt == c], bp[pt == c], zero_division=0) for n, c in PAIRS if (pt == c).any()},
        'sf': preds[(pt == 0) | (pt == 1)].mean(), 'sh': preds[(pt == 2) | (pt == 3)].mean(),
        'tf': preds[(pt == 0) | (pt == 2)].mean(), 'th': preds[(pt == 1) | (pt == 3)].mean(),
        # Taux prédiction positive (binaire) — pour dashboard DP cohérent
        'ps_b': {n: bp[pt == c].mean() for n, c in PAIRS},
        'sf_b': bp[(pt == 0) | (pt == 1)].mean(), 'sh_b': bp[(pt == 2) | (pt == 3)].mean(),
        'tf_b': bp[(pt == 0) | (pt == 2)].mean(), 'th_b': bp[(pt == 1) | (pt == 3)].mean(),
    }


# === Probing ===
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


# === Pair distribution ===
def pair_distribution(tp, tl, tpt, threshold=0.5):
    """Distribution des 4 paires : vraies amitiés (data) vs prédictions positives (pred).

    Returns:
        data_v: % par paire dans les vraies amitiés (ordre PAIR_ORDER)
        pred_v: % par paire dans les prédictions positives (ordre PAIR_ORDER)
        diffs: pred - data en points de pourcentage
    """
    pp_mask = (tp > threshold)
    pos_mask = (tl == 1)
    data_v = [100 * ((tpt == c) & pos_mask).sum() / pos_mask.sum() for _, c in PAIRS]
    pred_v = [100 * ((tpt == c) & pp_mask).sum() / pp_mask.sum() for _, c in PAIRS]
    diffs = [p - d for d, p in zip(data_v, pred_v)]
    return data_v, pred_v, diffs


# === Viz génériques ===
pair_color = lambda n: C_INTRA if n in ('F->F', 'H->H') else C_INTER


def annot_bars(ax, bars, vals=None, fmt='{:.3f}', dy=0.015, fontsize=10):
    """Annotations centrées sur chaque barre avec offset vertical."""
    vals = [b.get_height() for b in bars] if vals is None else vals
    for b, v in zip(bars, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(v), ha='center', fontweight='bold', fontsize=fontsize)


def bar_per_pair(ax, vals_dict, title, ylabel, hline=None, hline_label=None, ylim=(0, 1), extras=None):
    """Bar plot par PAIR_ORDER avec coloration intra/inter et ligne(s) de référence."""
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
    """Bar horizontal trié avec coloration par seuil. metrics = [(name, value, threshold), ...]."""
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
    """Légende globale intra/inter en bas de figure."""
    fig.legend(handles=[Patch(facecolor=C_INTRA, edgecolor='black', label='Intra-genre (F->F, H->H)'),
                        Patch(facecolor=C_INTER, edgecolor='black', label='Inter-genre (F->H, H->F)')],
               loc=loc, ncol=2, bbox_to_anchor=bbox, fontsize=10)


# === Viz spécifiques ===
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
    ax[1].set(ylabel='Accuracy du probing', ylim=(0.4, 1.05),
              title='Probing : prédire le genre depuis les embeddings')
    ax[1].legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_pair_distribution(data_v, pred_v, diffs, figsize=(13, 5)):
    """Affiche la distribution data vs pred par paire avec deltas annotés."""
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
           xlabel='Type de paire   —   delta (modèle - données) en points de % sous chaque type',
           title='Distribution des paires : données réelles vs prédictions du modèle',
           ylim=(-7, max(data_v + pred_v) * 1.18))
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    return fig, ax


# === Verdicts ===
verdict_str = lambda v, t, op='lt': 'fair' if (op == 'lt' and v <= t) or (op == 'ge' and v >= t) or (op == 'gt' and v > t) else 'biaisé'
verdict_mark = lambda v, t, op='lt': '✓' if (op == 'lt' and v <= t) or (op == 'ge' and v >= t) or (op == 'gt' and v > t) else '✗'
