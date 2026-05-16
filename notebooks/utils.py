"""Fonctions utilitaires et constantes pour le pipeline Responsible AI.

Ce module contient :
    - Constantes globales (PAIR_ORDER, PAIRS, couleurs)
    - Configuration de la reproductibilité (set_seed)
    - Chargement des données Pokec (download_pokec, load_pokec)
    - Fonctions de verdict (verdict_str, verdict_mark)
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
# Constantes
# ═══════════════════════════════════════════════════════════════════════════════

PAIR_ORDER = ['F->F', 'F->H', 'H->F', 'H->H']
PAIRS = [(name, idx) for idx, name in enumerate(PAIR_ORDER)]
C_INTRA = '#2196F3'
C_INTER = '#FF9800'

# Noms des features (même ordre que load_pokec)
FEAT_NAMES = ['comp', 'age', 'public', 'gender'] + [f'h{i}' for i in range(30)] + [f'l{i}' for i in range(30)]


# ═══════════════════════════════════════════════════════════════════════════════
# Setup & Reproductibilité
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    """Verrouille l'entropie pour garantir une reproductibilité mathématique stricte."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def download_pokec(data_dir):
    """Télécharge les fichiers de données Pokec depuis SNAP si pas déjà présents."""
    base_url = 'https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz'
    profile_url = 'https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz'
    for url in [base_url, profile_url]:
        filename = os.path.basename(url)
        filepath = os.path.join(data_dir, filename[:-3])
        if not os.path.exists(filepath):
            print(f"Téléchargement de {filename}...")
            os.system(f"wget -q {url} -O {filepath}.gz && gunzip -f {filepath}.gz")
        else:
            print(f"{filename} déjà présent, skipping download.")


def load_pokec(data_dir, region='ceska republika'):
    """Charge le sous-ensemble Pokec filtré par région.

    Args:
        data_dir: chemin vers le dossier contenant les fichiers Pokec
        region: préfixe de région pour filtrer les profils

    Returns:
        data: torch_geometric.data.Data avec x, edge_index, sens
        sens: tenseur 1D des genres (0=F, 1=H)
        p: DataFrame pandas des profils (pour analyses descriptives)
    """
    TC_IDX = list(range(9, 29)) + list(range(31, 37)) + [38, 39, 40, 42]
    USECOLS = [0, 1, 2, 3, 4, 7] + TC_IDX

    p = pd.read_csv(os.path.join(data_dir, 'soc-pokec-profiles.txt'), sep='\t', header=None,
                    na_values='null', usecols=USECOLS, low_memory=False)
    p.columns = ['user_id', 'public', 'comp', 'gender', 'region', 'age'] + [f't{i}' for i in range(30)]
    if region:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Verdicts
# ═══════════════════════════════════════════════════════════════════════════════

verdict_str = lambda v, t, op='lt': 'fair' if (op == 'lt' and v <= t) or (op == 'ge' and v >= t) or (op == 'gt' and v > t) else 'biaisé'
verdict_mark = lambda v, t, op='lt': '\u2713' if (op == 'lt' and v <= t) or (op == 'ge' and v >= t) or (op == 'gt' and v > t) else '\u2717'
