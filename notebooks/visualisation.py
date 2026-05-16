"""Fonctions de visualisation pour le pipeline Responsible AI.

Ce module contient :
    - Style académique : set_academic_style
    - Helpers génériques : pair_color, annot_bars, bar_per_pair, barh_dashboard, fairness_legend
    - Visualisations spécifiques : plot_representation_bias, plot_pair_distribution
    - Acte 1 : plot_models_comparison
    - Acte 2 : plot_feature_importance, plot_fidelity, plot_attention_bias, plot_ablation
    - Acte 3 : plot_pareto
    - Acte 4 : plot_mc_entropy, plot_reliability_diagrams, plot_noise_robustness, plot_radar
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from utils import PAIR_ORDER, PAIRS, C_INTRA, C_INTER, FEAT_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# Style académique
# ═══════════════════════════════════════════════════════════════════════════════

def set_academic_style():
    """Configure Matplotlib pour générer des graphiques vectoriels de qualité papier IEEE."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.5
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers génériques
# ═══════════════════════════════════════════════════════════════════════════════

def pair_color(n):
    """Retourne la couleur associée à un type de paire (intra=bleu, inter=orange)."""
    return C_INTRA if n in ('F->F', 'H->H') else C_INTER


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
    """Bar horizontal trié avec coloration par seuil.

    Args:
        metrics: liste de tuples (name, value, threshold)
        title: titre du graphique
        xlabel: label axe X
        threshold_lines: liste de tuples (value, label, color) pour les lignes verticales
        figsize: taille de la figure
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisations spécifiques
# ═══════════════════════════════════════════════════════════════════════════════

def plot_representation_bias(cos_fh, gen_acc_tr, gen_acc_te, prop_majo, figsize=(15, 5)):
    """Affiche les 2 subplots de Representation Bias : cosine F/H + probing genre.

    Args:
        cos_fh: cosine similarity entre centroïdes F et H
        gen_acc_tr: accuracy train du probing
        gen_acc_te: accuracy test du probing
        prop_majo: proportion classe majoritaire (baseline)

    Returns:
        fig, ax: figure et axes matplotlib
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=.35)

    # Cosine F vs H
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
    """Affiche la distribution data vs pred par paire avec deltas annotés.

    Args:
        data_v: % par paire dans les vraies amitiés
        pred_v: % par paire dans les prédictions positives
        diffs: différences pred - data en points de %

    Returns:
        fig, ax: figure et axe matplotlib
    """
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
           xlabel='Type de paire   ---   delta (modele - donnees) en points de % sous chaque type',
           title='Distribution des paires : données réelles vs prédictions du modèle',
           ylim=(-7, max(data_v + pred_v) * 1.18))
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Acte 1 — Comparaison des modèles
# ═══════════════════════════════════════════════════════════════════════════════

def plot_models_comparison(results, models_list=None):
    """3 panneaux : Performance / Fairness / Leakage pour comparer les modèles.

    Args:
        results: dict {model_name: {auc, ap, f1, f, cos_fh, leakage, ...}}
        models_list: liste ordonnée des noms de modèles (défaut : clés de results)
    """
    if models_list is None:
        models_list = list(results.keys())
    x = np.arange(len(models_list))
    w = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 : Performance
    ax = axes[0]
    for j, (metric, color) in enumerate([('auc', '#1f77b4'), ('ap', '#ff7f0e'), ('f1', '#2ca02c')]):
        vals = [results[m][metric] for m in models_list]
        bars = ax.bar(x + j * w, vals, w, label=metric.upper(), color=color, edgecolor='black')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.008, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set(xticks=x + w, xticklabels=models_list, ylabel='Score', title='Performance', ylim=(0, 1.12))
    ax.legend(fontsize=9)

    # Panel 2 : Fairness
    ax = axes[1]
    for j, (metric, label, color) in enumerate([('DP', 'DP gap', '#d62728'), ('EOpp', 'EOpp', '#9467bd'), ('EO', 'EO', '#8c564b')]):
        vals = [results[m]['f'][metric] for m in models_list]
        bars = ax.bar(x + j * w, vals, w, label=label, color=color, edgecolor='black')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.003, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(0.05, color='orange', ls='--', lw=1.5, label='Seuil fair (0.05)')
    ax.set(xticks=x + w, xticklabels=models_list, ylabel='Ecart', title='Fairness (plus bas = mieux)',
           ylim=(0, max(results[m]['f']['EO'] for m in models_list) * 1.8))
    ax.legend(fontsize=8)

    # Panel 3 : Leakage
    ax = axes[2]
    cos_vals = [results[m]['cos_fh'] for m in models_list]
    leak_vals = [100 * results[m]['leakage'] for m in models_list]
    bars1 = ax.bar(x - w/2, cos_vals, w, label='Cosine F/H', color='#9467bd', edgecolor='black')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, leak_vals, w, label='Leakage (pp)', color='#e377c2', edgecolor='black')
    ax.axhline(1.0, color='green', ls='--', lw=1, alpha=0.5, label='Cosine ideal (1.0)')
    ax.set(xticks=x, xticklabels=models_list, ylabel='Cosine similarity', title='Representation Bias', ylim=(0, 1.15))
    ax2.set_ylabel('Leakage (points de %)')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')

    fig.suptitle('Comparaison des modèles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Acte 2 — GNNExplainer
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(feat_imp, feat_names=None, top_k=15):
    """2 panneaux : top features intra/inter + importance genre par paire.

    Args:
        feat_imp: dict {pair_name: list of masks} (sortie de run_gnnexplainer)
        feat_names: liste des noms de features (défaut : FEAT_NAMES)
        top_k: nombre de features à afficher dans le panel de gauche
    """
    if feat_names is None:
        feat_names = FEAT_NAMES

    imp_by_type = {name: np.mean(feat_imp[name], axis=0) for name in PAIR_ORDER}
    imp_intra = np.mean([imp_by_type['F->F'], imp_by_type['H->H']], axis=0)
    imp_inter = np.mean([imp_by_type['F->H'], imp_by_type['H->F']], axis=0)
    diff = imp_intra - imp_inter
    top_idx = np.argsort(np.abs(diff))[-top_k:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top features : différence intra vs inter
    ax = axes[0]
    y = np.arange(len(top_idx))
    ax.barh(y - 0.18, imp_intra[top_idx], 0.35, label='Intra-genre', color=C_INTRA, edgecolor='black')
    ax.barh(y + 0.18, imp_inter[top_idx], 0.35, label='Inter-genre', color=C_INTER, edgecolor='black')
    ax.set_yticks(y, [feat_names[i] for i in top_idx])
    ax.set_xlabel('Importance moyenne (GNNExplainer)')
    ax.set_title(f'Top-{top_k} features : difference intra vs inter')
    ax.legend(fontsize=9); ax.invert_yaxis()

    # Importance du genre par type de paire
    ax = axes[1]
    gender_idx = feat_names.index('gender')
    gender_vals = [np.mean([m[gender_idx] for m in feat_imp[n]]) for n in PAIR_ORDER]
    bars = ax.bar(PAIR_ORDER, gender_vals, color=[pair_color(n) for n in PAIR_ORDER], edgecolor='black')
    annot_bars(ax, bars, gender_vals, fmt='{:.4f}')
    ax.set_ylabel('Importance (GNNExplainer)')
    ax.set_title('Importance du feature "gender" par type de paire')

    fig.suptitle('GNNExplainer — Quelles features expliquent les predictions de liens ?', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

    global_imp = np.mean([imp_intra, imp_inter], axis=0)
    gender_rank = int(np.sum(global_imp > global_imp[gender_idx]) + 1)
    print(f"Feature 'gender' : rang {gender_rank}/{len(feat_names)} en importance globale")


def plot_fidelity(df_fid, K_values=None):
    """Bar chart de Fidelity+/- par K.

    Args:
        df_fid: DataFrame indexé par K avec colonnes Fidelity+, Fidelity-
        K_values: liste de K (défaut : extrait de l'index)
    """
    if K_values is None:
        K_values = df_fid.index.tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    df_fid.plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c'], edgecolor='black')
    ax.set_xlabel('K (features masquees)'); ax.set_ylabel('Changement moyen de prediction')
    ax.set_title('Fidelity+/- : validation des explications GNNExplainer')
    ax.set_xticklabels([str(k) for k in K_values], rotation=0); ax.legend(fontsize=10)
    plt.tight_layout(); plt.show()
    print(df_fid.to_string(float_format='{:.4f}'.format))


def plot_attention_bias(att_same, att_cross):
    """Histogramme + boxplot des poids d'attention same-genre vs cross-genre.

    Args:
        att_same: array des poids d'attention pour arêtes same-genre
        att_cross: array des poids d'attention pour arêtes cross-genre
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(att_same, bins=50, alpha=0.7, density=True, label=f'Same-genre (n={len(att_same):,})', color=C_INTRA)
    ax.hist(att_cross, bins=50, alpha=0.7, density=True, label=f'Cross-genre (n={len(att_cross):,})', color=C_INTER)
    ax.set_xlabel("Poids d'attention (moyenne 4 tetes)"); ax.set_ylabel('Densite')
    ax.set_title("Distribution des poids d'attention GAT"); ax.legend(fontsize=9)

    ax = axes[1]
    bp = ax.boxplot([att_same, att_cross], labels=['Same-genre', 'Cross-genre'], patch_artist=True)
    bp['boxes'][0].set_facecolor(C_INTRA); bp['boxes'][1].set_facecolor(C_INTER)
    ax.set_ylabel("Poids d'attention"); ax.set_title("Attention GAT par type de voisin")

    fig.suptitle("GAT — Le modele prete-t-il plus attention aux voisins du meme genre ?", fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
    print(f"Attention same-genre  : {att_same.mean():.6f} | cross-genre : {att_cross.mean():.6f} | ratio : {att_same.mean()/att_cross.mean():.4f}")


def plot_ablation(auc_orig, auc_ablated, f_orig, f_ablated):
    """2 panneaux : impact performance + impact fairness de l'ablation d'un feature.

    Args:
        auc_orig: AUC avec le feature
        auc_ablated: AUC sans le feature
        f_orig: dict fairness avec le feature
        f_ablated: dict fairness sans le feature
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(['Avec genre', 'Sans genre'], [auc_orig, auc_ablated],
                  color=['#1f77b4', '#aec7e8'], edgecolor='black', width=0.5)
    annot_bars(ax, bars, [auc_orig, auc_ablated])
    ax.set_ylabel('AUC'); ax.set_title('Impact sur la performance'); ax.set_ylim(0, 1)

    ax = axes[1]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, [f_orig['DP'], f_orig['EOpp'], f_orig['EO']], w, label='Avec genre', color='#d62728', edgecolor='black')
    ax.bar(x + w/2, [f_ablated['DP'], f_ablated['EOpp'], f_ablated['EO']], w, label='Sans genre', color='#ff9896', edgecolor='black')
    ax.axhline(0.05, color='orange', ls='--', lw=1.5, label='Seuil (0.05)')
    ax.set_xticks(x, ['DP gap', 'EOpp', 'EO']); ax.set_ylabel('Ecart'); ax.set_title('Impact sur la fairness'); ax.legend(fontsize=9)

    fig.suptitle('Ablation contrefactuelle : retrait du feature "gender"', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

    print(f"AUC : {auc_orig:.4f} -> {auc_ablated:.4f} (delta = {auc_ablated - auc_orig:+.4f})")
    print(f"DP  : {f_orig['DP']:.4f} -> {f_ablated['DP']:.4f} (delta = {f_ablated['DP'] - f_orig['DP']:+.4f})")
    print(f"\n=> Le DP gap {'AUGMENTE' if f_ablated['DP'] > f_orig['DP'] else 'DIMINUE'} sans le feature genre : "
          f"le biais est {'TOPOLOGIQUE' if f_ablated['DP'] > f_orig['DP'] else 'lie au feature'}.")


# ═══════════════════════════════════════════════════════════════════════════════
# Acte 3 — Pareto
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pareto(act3):
    """Courbe de Pareto AUC vs DP gap + barres AUC/DP par configuration.

    Args:
        act3: dict {config_name: {auc, f: {DP, ...}, ...}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # Panel 1 : Pareto scatter
    ax = axes[0]
    for name, r in act3.items():
        marker = 'o' if name == 'Baseline' else 's' if name == 'Reweighting' else 'D'
        color = '#1f77b4' if name == 'Baseline' else '#2ca02c' if name == 'Reweighting' else '#d62728'
        ax.scatter(r['f']['DP'], r['auc'], marker=marker, color=color, s=120, zorder=5, edgecolor='black', label=name)
        short = name.replace('DP (lam=', 'l=').replace(')', '').replace('FairGNN (a=', 'FG=').replace(')', '')
        ax.annotate(short, (r['f']['DP'], r['auc']), textcoords='offset points', xytext=(8, 5), fontsize=7)
    ax.axvline(0.05, color='orange', ls='--', lw=1.5, alpha=0.7, label='Seuil DP fair (0.05)')
    ax.set_xlabel('DP gap (plus bas = plus equitable)'); ax.set_ylabel('AUC (plus haut = meilleur)')
    ax.set_title('Courbe de Pareto : Performance vs Fairness')
    ax.legend(fontsize=7, loc='lower left'); ax.grid(True, alpha=0.3)

    # Panel 2 : Barres AUC + DP gap
    ax = axes[1]
    names = list(act3.keys())
    aucs = [act3[n]['auc'] for n in names]
    dps = [act3[n]['f']['DP'] for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, aucs, w, label='AUC', color='#1f77b4', edgecolor='black')
    ax_r = ax.twinx()
    ax_r.bar(x + w/2, dps, w, label='DP gap', color='#d62728', edgecolor='black')
    ax_r.axhline(0.05, color='orange', ls='--', lw=1.5, alpha=0.7)
    short_names = [n.replace('DP (lam=', 'l=').replace(')', '').replace('FairGNN (a=', 'FG=').replace(')', '') for n in names]
    ax.set_xticks(x, short_names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('AUC', color='#1f77b4'); ax_r.set_ylabel('DP gap', color='#d62728')
    ax.set_title('AUC et DP gap par configuration'); ax.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper right')

    fig.suptitle('Acte 3 — Compromis Performance vs Fairness', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Acte 4 — MC Dropout, ECE, Robustesse, Radar
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mc_entropy(mc_results, tpt_test, T=30):
    """Boxplots de l'entropie prédictive intra vs inter par modèle.

    Args:
        mc_results: dict {model_name: {entropy, ...}}
        tpt_test: array des codes de paires pour les arêtes test
        T: nombre de passes MC (pour le titre)
    """
    intra_mask = (tpt_test == 0) | (tpt_test == 3)
    inter_mask = (tpt_test == 1) | (tpt_test == 2)

    fig, axes = plt.subplots(1, len(mc_results), figsize=(6 * len(mc_results), 5))
    if len(mc_results) == 1:
        axes = [axes]
    for i, (name, mc) in enumerate(mc_results.items()):
        ax = axes[i]
        ent = mc['entropy']
        bp = ax.boxplot([ent[intra_mask], ent[inter_mask]], labels=['Intra-genre', 'Inter-genre'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(C_INTRA); bp['boxes'][1].set_facecolor(C_INTER)
        ax.set_ylabel('Entropie predictive')
        ax.set_title(f'{name}\nH_intra={ent[intra_mask].mean():.4f}, H_inter={ent[inter_mask].mean():.4f}')
        ax.axhline(np.log(2), color='red', ls='--', lw=1, alpha=0.5, label=f'H max ({np.log(2):.3f})')
        ax.legend(fontsize=8)
    fig.suptitle(f'MC Dropout (T={T}) — Entropie predictive par type de paire', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_reliability_diagrams(mc_results, labels_test, n_bins=15):
    """Reliability diagrams + ECE pour chaque modèle.

    Args:
        mc_results: dict {model_name: {p_bar, ...}}
        labels_test: array des labels binaires
        n_bins: nombre de bins

    Returns:
        ece_values: dict {model_name: ece}
    """
    from metrics import compute_ece

    fig, axes = plt.subplots(1, len(mc_results), figsize=(6 * len(mc_results), 5))
    if len(mc_results) == 1:
        axes = [axes]
    ece_values = {}
    for i, (name, mc) in enumerate(mc_results.items()):
        ax = axes[i]
        preds = mc['p_bar']
        ece, bin_accs, bin_confs, bin_counts = compute_ece(preds, labels_test, n_bins)
        ece_values[name] = ece
        valid = bin_counts > 0
        ax.bar(bin_confs[valid], bin_accs[valid], width=1/n_bins, alpha=0.7, color='#1f77b4', edgecolor='black', label='Accuracy observee')
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Calibration parfaite')
        ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
        ax.set_title(f'{name}\nECE = {ece:.4f}'); ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle('Reliability Diagrams — Calibration (MC Dropout)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()

    for name, ece in ece_values.items():
        v = "bien calibre" if ece < 0.05 else "acceptable" if ece < 0.10 else "mal calibre"
        print(f"  {name:18s} | ECE = {ece:.4f} ({v})")

    return ece_values


def plot_noise_robustness(df_noise, colors):
    """2 panneaux : AUC et DP gap en fonction du bruit sigma.

    Args:
        df_noise: DataFrame avec colonnes Modele, sigma, AUC, DP
        colors: dict {model_name: color}
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sigmas_full = sorted(df_noise['sigma'].unique())

    for name, color in colors.items():
        df_m = df_noise[df_noise['Modele'] == name]
        axes[0].plot(df_m['sigma'], df_m['AUC'], 'o-', color=color, label=name, lw=2, markersize=8)
        axes[1].plot(df_m['sigma'], df_m['DP'], 'o-', color=color, label=name, lw=2, markersize=8)

    axes[0].set_xlabel('Bruit (sigma)'); axes[0].set_ylabel('AUC'); axes[0].set_title('Robustesse performance')
    axes[0].legend(fontsize=9); axes[0].set_xticks(sigmas_full); axes[0].grid(True, alpha=0.3)
    axes[1].axhline(0.05, color='orange', ls='--', lw=1.5, label='Seuil fair')
    axes[1].set_xlabel('Bruit (sigma)'); axes[1].set_ylabel('DP gap'); axes[1].set_title('Robustesse fairness')
    axes[1].legend(fontsize=9); axes[1].set_xticks(sigmas_full); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Robustesse aux perturbations de features', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_radar(summary, colors):
    """Radar 5 axes : Performance / Fairness / Robustesse / Calibration / Certitude.

    Args:
        summary: dict {model_name: {AUC, DP gap, ECE, Entropie moy, Robustesse AUC}}
        colors: dict {model_name: color}
    """
    radar_metrics = {
        'Performance\n(AUC)': {n: s['AUC'] for n, s in summary.items()},
        'Fairness\n(1-DP)': {n: 1 - s['DP gap'] for n, s in summary.items()},
        'Robustesse\n(AUC ratio)': {n: s['Robustesse AUC'] for n, s in summary.items()},
        'Calibration\n(1-ECE)': {n: 1 - s['ECE'] for n, s in summary.items()},
        'Certitude\n(1-H/Hmax)': {n: 1 - s['Entropie moy'] / np.log(2) for n, s in summary.items()},
    }

    categories = list(radar_metrics.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, color in colors.items():
        values = [radar_metrics[cat][name] for cat in categories] + [radar_metrics[categories[0]][name]]
        ax.plot(angles, values, 'o-', color=color, label=name, lw=2, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title('Compromis Performance / Fairness / Robustesse', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10, bbox_to_anchor=(1.35, 0))
    plt.tight_layout(); plt.show()

    # Verdict final
    print("\n=== Verdict final ===")
    for name, s in summary.items():
        fair = "FAIR" if s['DP gap'] < 0.05 else "NON FAIR"
        robust = "ROBUSTE" if s['Robustesse AUC'] > 0.95 else "SENSIBLE"
        print(f"  {name:18s} | AUC={s['AUC']:.4f} | DP={s['DP gap']:.4f} [{fair}] | Rob={s['Robustesse AUC']:.4f} [{robust}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Acte 6 — Réplication multi-régions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_multi_region_dp(df_regions, rho=None, figsize=(14, 5)):
    """Visualisation 2 panneaux pour la réplication multi-régions.

    Panel A : DP gap par région et par méthode (barres groupées).
    Panel B : Homophilie vs DP gap baseline (scatter) avec corrélation Spearman.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(len(df_regions))
    w = 0.27
    axes[0].bar(x - w, df_regions['DP_base'], w, label='Baseline',
                color='#1f77b4', edgecolor='black')
    axes[0].bar(x,     df_regions['DP_dp'],   w, label='DP (lambda=0.1)',
                color='#2ca02c', edgecolor='black')
    axes[0].bar(x + w, df_regions['DP_fg'],   w, label='FairGNN (alpha=0.5)',
                color='#d62728', edgecolor='black')
    axes[0].axhline(0.05, color='orange', ls='--', lw=1.5, label='Seuil DP fair (0.05)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r.split()[0][:10] for r in df_regions.index], rotation=15)
    axes[0].set(ylabel='DP gap', title='DP gap par region - 3 methodes')
    axes[0].legend(fontsize=9, loc='upper right')

    axes[1].scatter(df_regions['homophily'], df_regions['DP_base'], s=140,
                    color='#1f77b4', edgecolor='black', zorder=3, label='Baseline')
    axes[1].scatter(df_regions['homophily'], df_regions['DP_fg'], s=140,
                    color='#d62728', edgecolor='black', zorder=3, label='FairGNN (alpha=0.5)')
    for r_, row in df_regions.iterrows():
        axes[1].annotate(r_.split()[0][:8], (row['homophily'], row['DP_base']),
                         xytext=(6, 6), textcoords='offset points', fontsize=9)
    axes[1].axhline(0.05, color='orange', ls='--', lw=1.5, alpha=0.7)
    title = "Homophilie <-> DP gap"
    if rho is not None and not np.isnan(rho):
        title += f"  (Spearman rho={rho:.2f})"
    axes[1].set(xlabel="Ratio d'homophilie", ylabel='DP gap', title=title)
    axes[1].legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.show()
