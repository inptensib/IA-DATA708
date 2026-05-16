"""Métriques de performance, fairness et incertitude pour la link prediction.

Ce module contient :
    - Classification des paires : classify_pairs
    - Métriques de fairness : fairness
    - Évaluation complète : evaluate_link_pred, collect_results
    - Probing de genre : probe_gender
    - Distribution des paires : pair_distribution
    - Incertitude (MC Dropout) : mc_dropout_predict
    - Calibration : compute_ece
    - XAI : run_gnnexplainer, compute_fidelity, ablation_feature
    - Robustesse : evaluate_noise_robustness
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import PAIR_ORDER, PAIRS, FEAT_NAMES, set_seed


# ═══════════════════════════════════════════════════════════════════════════════
# Classification des paires
# ═══════════════════════════════════════════════════════════════════════════════

def classify_pairs(ei, sens):
    """Encode chaque arête par le code de paire (0=F->F, 1=F->H, 2=H->F, 3=H->H)."""
    return sens[ei[0]] * 2 + sens[ei[1]]


# ═══════════════════════════════════════════════════════════════════════════════
# Fairness
# ═══════════════════════════════════════════════════════════════════════════════

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
        'ps': {n: preds[pt == c].mean() for n, c in PAIRS},
        'pa': {n: roc_auc_score(labels[pt == c], preds[pt == c]) for n, c in PAIRS
               if (labels[pt == c] > 0).any() and (labels[pt == c] == 0).any()},
        'pf1': {n: f1_score(labels[pt == c], bp[pt == c], zero_division=0) for n, c in PAIRS if (pt == c).any()},
        'sf': preds[(pt == 0) | (pt == 1)].mean(), 'sh': preds[(pt == 2) | (pt == 3)].mean(),
        'tf': preds[(pt == 0) | (pt == 2)].mean(), 'th': preds[(pt == 1) | (pt == 3)].mean(),
        'ps_b': {n: bp[pt == c].mean() for n, c in PAIRS},
        'sf_b': bp[(pt == 0) | (pt == 1)].mean(), 'sh_b': bp[(pt == 2) | (pt == 3)].mean(),
        'tf_b': bp[(pt == 0) | (pt == 2)].mean(), 'th_b': bp[(pt == 1) | (pt == 3)].mean(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Évaluation complète
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_link_pred(model, test_data, sens, device):
    """Évalue le modèle sur le test set et retourne tous les artefacts utiles.

    Returns:
        tp: prédictions sigmoid (proba), shape (n,)
        tl: labels binaires, shape (n,)
        tpt: codes de paires (0..3), shape (n,)
        auc: AUC-ROC globale
        ap: Average Precision globale
        f: dict de fairness (sortie de fairness())
        f1_global: F1 score global (seuil 0.5)
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


def collect_results(model, test_data, sens, device):
    """Évalue un modèle et collecte toutes les métriques (performance + fairness + leakage).

    Returns:
        dict avec auc, ap, f1, f (fairness), cos_fh, leakage, tp, tl, tpt
    """
    tp, tl, tpt, auc, ap, f, f1 = evaluate_link_pred(model, test_data, sens, device)
    model.eval()
    with torch.no_grad():
        z = model.encode(test_data.x.to(device), test_data.edge_index.to(device)).cpu()
    cos_fh, _, acc_te, prop_majo = probe_gender(z, sens)
    return {
        'auc': auc, 'ap': ap, 'f1': f1, 'f': f,
        'cos_fh': cos_fh, 'leakage': acc_te - prop_majo,
        'tp': tp, 'tl': tl, 'tpt': tpt,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Probing de genre
# ═══════════════════════════════════════════════════════════════════════════════

def probe_gender(z, sens, test_size=0.3, random_state=42):
    """Probe attaque pour évaluer si les embeddings encodent le genre.

    Args:
        z: embeddings des noeuds, shape (n_nodes, dim)
        sens: attribut sensible (genre), shape (n_nodes,)
        test_size: proportion du test set pour le probing
        random_state: seed pour le split

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


# ═══════════════════════════════════════════════════════════════════════════════
# Distribution des paires
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Incertitude — MC Dropout
# ═══════════════════════════════════════════════════════════════════════════════

def mc_dropout_predict(model, test_data, device, T=30):
    """T forward passes avec dropout activé (model.train()).

    Args:
        model: modèle GNN avec dropout
        test_data: données de test (PyG Data avec edge_label_index)
        device: device torch
        T: nombre de passes MC

    Returns:
        p_bar: prédiction moyenne, shape (n_edges,)
        entropy: entropie prédictive H(p_bar)
        variance: variance des prédictions MC
        mi: information mutuelle (incertitude épistémique)
        all_probs: toutes les prédictions, shape (T, n_edges)
    """
    x = test_data.x.to(device)
    ei = test_data.edge_index.to(device)
    eli = test_data.edge_label_index.to(device)

    all_probs = []
    for _ in range(T):
        model.train()  # Active le dropout
        with torch.no_grad():
            probs = torch.sigmoid(model(x, ei, eli)).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.stack(all_probs)  # (T, n_edges)
    p_bar = all_probs.mean(axis=0)

    eps = 1e-10
    entropy = -p_bar * np.log(p_bar + eps) - (1 - p_bar) * np.log(1 - p_bar + eps)
    variance = all_probs.var(axis=0)

    # Information mutuelle = H(p_bar) - moyenne des H(p_t)
    h_individual = -all_probs * np.log(all_probs + eps) - (1 - all_probs) * np.log(1 - all_probs + eps)
    mi = entropy - h_individual.mean(axis=0)

    return p_bar, entropy, variance, mi, all_probs


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration — ECE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece(preds, labels, n_bins=15):
    """Expected Calibration Error (cours Franchi p.108).

    Args:
        preds: scores de prédiction (proba), shape (n,)
        labels: labels binaires, shape (n,)
        n_bins: nombre de bins pour la discrétisation

    Returns:
        ece: valeur ECE scalaire
        bin_accs: accuracy par bin
        bin_confs: confiance moyenne par bin
        bin_counts: nombre d'échantillons par bin
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
            continue
        bin_accs.append(labels[mask].mean())
        bin_confs.append(preds[mask].mean())
        bin_counts.append(mask.sum())
    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)
    total = bin_counts.sum()
    ece = np.sum(bin_counts / total * np.abs(bin_accs - bin_confs)) if total > 0 else 0
    return ece, bin_accs, bin_confs, bin_counts


# ═══════════════════════════════════════════════════════════════════════════════
# XAI — GNNExplainer
# ═══════════════════════════════════════════════════════════════════════════════

def run_gnnexplainer(model, test_data, sens, device, n_per_type=12, epochs=200, lr=0.01):
    """Exécute GNNExplainer sur des arêtes test équilibrées par type de paire.

    Args:
        model: modèle GCN (encodeur-décodeur)
        test_data: données de test PyG
        sens: attribut sensible (genre)
        device: device torch
        n_per_type: nombre d'arêtes par type de paire
        epochs: epochs pour GNNExplainer
        lr: learning rate pour GNNExplainer

    Returns:
        feat_imp: dict {pair_name: list of masks}, chaque masque shape (n_features,)
        explain_edges: arêtes expliquées, shape (2, n_total)
        explain_pt: types de paires des arêtes expliquées
    """
    from torch_geometric.explain import Explainer, GNNExplainer as GNNExplainerAlgo
    from models import GCNLinkWrapper
    from tqdm.notebook import tqdm

    wrapper = GCNLinkWrapper(model).to(device)
    wrapper.eval()

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainerAlgo(epochs=epochs, lr=lr),
        explanation_type='model',
        node_mask_type='common_attributes',
        edge_mask_type='object',
        model_config=dict(mode='binary_classification', task_level='edge', return_type='raw'),
    )

    # Sélection d'arêtes test positives, équilibrées par type de paire
    set_seed(42)
    pos_mask = test_data.edge_label == 1
    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
    eli_pos = test_data.edge_label_index[:, pos_idx]
    pt_pos = classify_pairs(eli_pos, sens)

    selected = []
    for _, code in PAIRS:
        candidates = (pt_pos == code).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(candidates))[:n_per_type]
        selected.append(candidates[perm])
    selected = torch.cat(selected)
    explain_edges = eli_pos[:, selected]
    explain_pt = pt_pos[selected]
    print(f"{len(selected)} arêtes : " + " | ".join(f"{n}={(explain_pt==c).sum().item()}" for n, c in PAIRS))

    # Exécution GNNExplainer
    feat_imp = {name: [] for name in PAIR_ORDER}
    x_dev = test_data.x.to(device)
    ei_dev = test_data.edge_index.to(device)

    for i in tqdm(range(len(selected)), desc='GNNExplainer'):
        eli_single = explain_edges[:, i:i+1].to(device)
        pt_name = PAIR_ORDER[explain_pt[i].item()]
        explanation = explainer(x=x_dev, edge_index=ei_dev, edge_label_index=eli_single)
        mask = explanation.node_mask.detach().cpu().numpy().flatten()
        feat_imp[pt_name].append(mask)

    return feat_imp, explain_edges, explain_pt


# ═══════════════════════════════════════════════════════════════════════════════
# XAI — Fidelity
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fidelity(model, test_data, feat_imp, explain_edges, device, K_values=(3, 5, 10)):
    """Calcule Fidelity+/- pour chaque K en masquant les features top/bottom.

    Args:
        model: modèle GCN
        test_data: données de test PyG
        feat_imp: dict de feature importances (sortie de run_gnnexplainer)
        explain_edges: arêtes à évaluer, shape (2, n)
        device: device torch
        K_values: tuple des valeurs de K à tester

    Returns:
        DataFrame avec colonnes K, Fidelity+, Fidelity-
    """
    import pandas as pd

    global_imp = np.mean(np.concatenate([np.array(v) for v in feat_imp.values()]), axis=0)
    model.eval()
    x_dev = test_data.x.to(device)
    ei_dev = test_data.edge_index.to(device)
    eli_dev = explain_edges.to(device)

    with torch.no_grad():
        scores_orig = torch.sigmoid(model(x_dev, ei_dev, eli_dev)).cpu().numpy()

    fidelity_results = []
    for K in K_values:
        top_k = np.argsort(global_imp)[-K:]
        bot_k = np.argsort(global_imp)[:K]
        with torch.no_grad():
            x_top = test_data.x.clone(); x_top[:, top_k] = 0
            scores_top = torch.sigmoid(model(x_top.to(device), ei_dev, eli_dev)).cpu().numpy()
            x_bot = test_data.x.clone(); x_bot[:, bot_k] = 0
            scores_bot = torch.sigmoid(model(x_bot.to(device), ei_dev, eli_dev)).cpu().numpy()
        fidelity_results.append({
            'K': K,
            'Fidelity+': np.mean(np.abs(scores_orig - scores_top)),
            'Fidelity-': np.mean(np.abs(scores_orig - scores_bot)),
        })

    return pd.DataFrame(fidelity_results).set_index('K')


# ═══════════════════════════════════════════════════════════════════════════════
# XAI — Ablation contrefactuelle
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_feature(model, test_data, sens, device, feature_idx=None):
    """Compare les prédictions avec et sans un feature (par défaut : gender).

    Args:
        model: modèle GCN
        test_data: données de test PyG
        sens: attribut sensible
        device: device torch
        feature_idx: index du feature à ablater (défaut : index de 'gender' dans FEAT_NAMES)

    Returns:
        dict avec auc_orig, auc_ablated, f_orig, f_ablated
    """
    if feature_idx is None:
        feature_idx = FEAT_NAMES.index('gender')

    model.eval()
    with torch.no_grad():
        tp_orig = torch.sigmoid(model(
            test_data.x.to(device), test_data.edge_index.to(device),
            test_data.edge_label_index.to(device))).cpu().numpy()
        x_ablated = test_data.x.clone()
        x_ablated[:, feature_idx] = 0
        tp_ablated = torch.sigmoid(model(
            x_ablated.to(device), test_data.edge_index.to(device),
            test_data.edge_label_index.to(device))).cpu().numpy()

    tl = test_data.edge_label.numpy()
    tpt = classify_pairs(test_data.edge_label_index, sens).numpy()
    return {
        'auc_orig': roc_auc_score(tl, tp_orig),
        'auc_ablated': roc_auc_score(tl, tp_ablated),
        'f_orig': fairness(tp_orig, tl, tpt),
        'f_ablated': fairness(tp_ablated, tl, tpt),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Robustesse — Perturbation de features
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_noise_robustness(models_dict, test_data, sens, device, sigmas=(0.1, 0.5, 1.0)):
    """Évalue AUC et DP gap sous bruit gaussien pour plusieurs modèles.

    Args:
        models_dict: dict {name: model}
        test_data: données de test PyG
        sens: attribut sensible
        device: device torch
        sigmas: niveaux de bruit à tester

    Returns:
        DataFrame avec colonnes Modele, sigma, AUC, DP
    """
    import pandas as pd

    noise_results = []
    for name, mdl in models_dict.items():
        mdl.eval()
        # Propre (sigma=0)
        tp, tl, tpt, auc, _, f, _ = evaluate_link_pred(mdl, test_data, sens, device)
        noise_results.append({'Modele': name, 'sigma': 0.0, 'AUC': auc, 'DP': f['DP']})
        for sigma in sigmas:
            set_seed(42)
            x_noisy = test_data.x + torch.randn_like(test_data.x) * sigma
            with torch.no_grad():
                tp_n = torch.sigmoid(mdl(x_noisy.to(device), test_data.edge_index.to(device),
                                         test_data.edge_label_index.to(device))).cpu().numpy()
            tl_n = test_data.edge_label.numpy()
            tpt_n = classify_pairs(test_data.edge_label_index, sens).numpy()
            f_n = fairness(tp_n, tl_n, tpt_n)
            noise_results.append({'Modele': name, 'sigma': sigma, 'AUC': roc_auc_score(tl_n, tp_n), 'DP': f_n['DP']})

    return pd.DataFrame(noise_results)
