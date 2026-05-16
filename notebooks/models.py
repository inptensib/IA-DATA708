"""Modèles GNN et fonctions d'entraînement pour la link prediction.

Ce module contient :
    - Architectures : GCN, SGC, GAT
    - Wrappers : GCNLinkWrapper (pour GNNExplainer)
    - Discriminateur adversarial : GenderDiscriminator (FairGNN)
    - Entraînement : train_with_best, train_fair, train_fairgnn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATv2Conv, SGConv
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm

from metrics import classify_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Architectures
# ═══════════════════════════════════════════════════════════════════════════════

class GCN(nn.Module):
    """GCN encoder (2 couches) + dot product decoder pour la link prediction."""

    def __init__(self, inc, hid=128, out=64, drop=0.5):
        super().__init__()
        self.c1 = GCNConv(inc, hid)
        self.c2 = GCNConv(hid, out)
        self.drop = drop

    def encode(self, x, ei, ew=None):
        h = F.relu(self.c1(x, ei, edge_weight=ew))
        h = F.dropout(h, p=self.drop, training=self.training)
        return self.c2(h, ei, edge_weight=ew)

    def decode(self, z, ei):
        return (z[ei[0]] * z[ei[1]]).sum(dim=-1)

    def forward(self, x, ei, eli, ew=None):
        return self.decode(self.encode(x, ei, ew), eli)


class SGC(nn.Module):
    """Simple Graph Convolution — propagation linéaire (pas de non-linéarité entre couches)."""

    def __init__(self, inc, out=64, K=2):
        super().__init__()
        self.conv = SGConv(inc, out, K=K, cached=True)

    def encode(self, x, ei):
        return self.conv(x, ei)

    def decode(self, z, ei):
        return (z[ei[0]] * z[ei[1]]).sum(dim=-1)

    def forward(self, x, ei, eli):
        return self.decode(self.encode(x, ei), eli)


class GAT(nn.Module):
    """GAT encoder (2 couches GATv2) + dot product decoder avec extraction d'attention."""

    def __init__(self, inc, hid=128, out=64, heads=4, drop=0.5):
        super().__init__()
        self.drop = drop
        self.c1 = GATv2Conv(inc, hid, heads=heads, concat=True, dropout=drop)
        self.c2 = GATv2Conv(hid * heads, out, heads=heads, concat=False, dropout=drop)

    def encode(self, x, ei, ew=None, return_attn=False):
        if return_attn:
            h, attn_1 = self.c1(x, ei, edge_attr=ew, return_attention_weights=True)
            h = F.relu(h)
            h = F.dropout(h, p=self.drop, training=self.training)
            z, attn_2 = self.c2(h, ei, edge_attr=ew, return_attention_weights=True)
            return z, (attn_1, attn_2)
        else:
            h = F.relu(self.c1(x, ei, edge_attr=ew))
            h = F.dropout(h, p=self.drop, training=self.training)
            return self.c2(h, ei, edge_attr=ew)

    def decode(self, z, ei):
        return (z[ei[0]] * z[ei[1]]).sum(dim=-1)

    def forward(self, x, ei, eli, ew=None):
        z = self.encode(x, ei, ew)
        return self.decode(z, eli)


# ═══════════════════════════════════════════════════════════════════════════════
# Wrappers & Modules auxiliaires
# ═══════════════════════════════════════════════════════════════════════════════

class GCNLinkWrapper(nn.Module):
    """Wrapper pour l'Explainer PyG : passe edge_label_index en argument nommé."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_label_index):
        z = self.model.encode(x, edge_index)
        return self.model.decode(z, edge_label_index)


class GenderDiscriminator(nn.Module):
    """MLP discriminateur : prédit le genre depuis les embeddings du GCN (FairGNN)."""

    def __init__(self, in_dim=64, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions d'entraînement
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_best(model, train_data, val_data, device, epochs=100, lr=0.01,
                    weight_decay=5e-4, eval_every=10, criterion=None, desc='Entraînement'):
    """Entraîne un modèle de link prediction avec best-model checkpointing sur val AUC.

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


def train_fair(model, train_data, val_data, sens, device,
               epochs=100, lr=0.01, weight_decay=5e-4, eval_every=10,
               reweight=False, lam_dp=0.0, desc='Entraînement'):
    """Entraînement GCN avec reweighting et/ou régularisation DP.

    Args:
        reweight: si True, pondère la loss par type de paire (w = N_total / 2*N_type)
        lam_dp: coefficient lambda de régularisation DP (0 = désactivé)

    Returns:
        best_auc: meilleur AUC de validation observé
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
        print(f"  Reweighting : w_intra={weights[intra_mask][0]:.4f}, w_inter={weights[inter_mask][0]:.4f}")

    vx, vei = val_data.x.to(device), val_data.edge_index.to(device)
    veli, vl = val_data.edge_label_index.to(device), val_data.edge_label.numpy()

    bce = nn.BCEWithLogitsLoss(reduction='none')
    best_auc, best_state = 0, None

    for ep in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        model.train()
        opt.zero_grad()
        logits = model(xd, eid, elid)

        task_loss = (bce(logits, eld) * weights).mean()

        if lam_dp > 0:
            scores = torch.sigmoid(logits)
            dp_penalty = torch.abs(scores[intra_mask].mean() - scores[inter_mask].mean())
            loss = task_loss + lam_dp * dp_penalty
        else:
            loss = task_loss

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


def train_fairgnn(model, train_data, val_data, sens, device,
                  alpha=1.0, epochs=100, lr=0.01, lr_disc=0.001,
                  weight_decay=5e-4, eval_every=10, desc='FairGNN'):
    """Entraînement adversarial : encodeur vs discriminateur de genre.

    L'encodeur minimise L_task - alpha * L_disc (fool le discriminateur).
    Le discriminateur minimise L_disc (prédire le genre depuis z).
    Alternance : 1 step disc, 1 step encodeur par epoch.

    Returns:
        best_auc: meilleur AUC de validation observé
    """
    disc = GenderDiscriminator(in_dim=64).to(device)
    opt_enc = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_disc = Adam(disc.parameters(), lr=lr_disc)
    bce_task = nn.BCEWithLogitsLoss()
    bce_disc = nn.BCEWithLogitsLoss()

    xd = train_data.x.to(device)
    eid = train_data.edge_index.to(device)
    elid = train_data.edge_label_index.to(device)
    eld = train_data.edge_label.float().to(device)
    sens_d = sens.float().to(device)

    vx, vei = val_data.x.to(device), val_data.edge_index.to(device)
    veli, vl = val_data.edge_label_index.to(device), val_data.edge_label.numpy()

    best_auc, best_state = 0, None

    for ep in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        model.train()
        disc.train()

        # Step 1 : entraîner le discriminateur (z détaché)
        with torch.no_grad():
            z_det = model.encode(xd, eid)
        disc_logits = disc(z_det)
        loss_disc = bce_disc(disc_logits, sens_d)
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Step 2 : entraîner l'encodeur (task - alpha * disc)
        z = model.encode(xd, eid)
        logits = model.decode(z, elid)
        loss_task = bce_task(logits, eld)

        disc_logits_enc = disc(z)
        loss_adv = bce_disc(disc_logits_enc, sens_d)

        loss_enc = loss_task - alpha * loss_adv
        opt_enc.zero_grad()
        loss_enc.backward()
        opt_enc.step()

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
