# PyTorch Geometric GAT baseline for session_value prediction on a heterogeneous graph
# ---------------------------------------------------------------
# Assumptions
# - CSV schema: event_time, event_type, product_id, category_id, user_id, user_session, session_value
# - Goal: predict session_value at the *session* node level, avoiding leakage
# - Split: time-based (train/val/test by session start_time)
# - Features: learned embeddings for node IDs, event_type embedding on edges; temporal position encodings
# - Model: Hetero GAT (HeteroConv with GATConv per relation), readout on session nodes
# - Scaling: use NeighborLoader for minibatching
# ---------------------------------------------------------------

import os
import math
import json
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# If PyG is not installed, you need:
# pip install torch-geometric torch-scatter torch-sparse torch-cluster
# (with the matching torch CUDA versions). See https://pytorch-geometric.readthedocs.io/
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected

# ------------------------
# 1) Load & preprocess CSV
# ------------------------

def load_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse times, sort to preserve order within sessions
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df = df.sort_values(['user_session', 'event_time']).reset_index(drop=True)
    return df

# ------------------------
# 2) Build heterogeneous graph
#    Node types: 'user', 'session', 'product', 'category'
#    Edge types:
#      ('user',   'has',      'session')
#      ('session','contains', 'product')  -- carries edge features (event_type, pos, delta_t)
#      ('product','in',       'category')
# ------------------------

def factorize_series(s: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    cats, idx = np.unique(s.astype(str).values, return_inverse=True)
    mapping = {str(c): int(i) for i, c in enumerate(cats)}
    return idx.astype(np.int64), mapping


def build_hetero_graph(df: pd.DataFrame,
                       time_cutoff: pd.Timestamp = None,
                       min_session_events: int = 1,
                       undirected: bool = True) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
    # Filter (optional) & keep only valid sessions
    if min_session_events > 1:
        vc = df['user_session'].value_counts()
        df = df[df['user_session'].isin(vc[vc >= min_session_events].index)]

    # Session-level times & labels
    sess_grp = df.groupby('user_session')
    session_start = sess_grp['event_time'].min()
    session_end = sess_grp['event_time'].max()
    session_y = sess_grp['session_value'].first()

    # Optional time cutoff (for leakage-safe splits)
    if time_cutoff is not None:
        keep_sessions = session_start.index[session_start <= time_cutoff]
        df = df[df['user_session'].isin(keep_sessions)]
        session_start = session_start.loc[keep_sessions]
        session_end = session_end.loc[keep_sessions]
        session_y = session_y.loc[keep_sessions]

    # Factorize IDs for node indices
    u_idx, user_map = factorize_series(df['user_id'])
    s_idx, sess_map = factorize_series(df['user_session'])
    p_idx, prod_map = factorize_series(df['product_id'])
    c_idx, cat_map  = factorize_series(df['category_id'])

    # Reindex to these integer IDs in the event table
    df2 = df.copy()
    df2['u_idx'] = u_idx
    df2['s_idx'] = s_idx
    df2['p_idx'] = p_idx
    df2['c_idx'] = c_idx

    # Event type as categorical edge attribute
    et_idx, et_map = factorize_series(df2['event_type'])
    df2['et_idx'] = et_idx

    # Build position-in-session and delta_t features
    df2['pos_in_sess'] = df2.groupby('s_idx').cumcount()
    # Normalized position [0,1]
    maxpos = df2.groupby('s_idx')['pos_in_sess'].transform('max').replace(0, 1)
    df2['pos_norm'] = df2['pos_in_sess'] / maxpos

    # Delta t wrt session start (seconds, log1p scaled)
    s_start_time = df2.groupby('s_idx')['event_time'].transform('min')
    df2['dt_s'] = (df2['event_time'] - s_start_time).dt.total_seconds().clip(lower=0).fillna(0)
    df2['dt_log'] = np.log1p(df2['dt_s'])

    # Session labels in index order of 'session' node type
    # Map session ids back to integer order 0..(n_sessions-1)
    sess_order = pd.Series(np.arange(len(sess_map)), index=pd.Index([k for k,_ in sorted(sess_map.items(), key=lambda x: x[1])], name='user_session'))
    # Build y aligned with sess_order
    y_series = session_y.reindex(sess_order.index).fillna(method='ffill').fillna(method='bfill')
    y = torch.tensor(y_series.values, dtype=torch.float)

    data = HeteroData()
    data['user'].num_nodes = len(user_map)
    data['session'].num_nodes = len(sess_map)
    data['product'].num_nodes = len(prod_map)
    data['category'].num_nodes = len(cat_map)

    # user -> session (one edge per session: user owns session)
    # We can take the first user for each session (dataset shows single user per session)
    sess_user = df2.groupby('s_idx')['u_idx'].first()
    edge_u2s = torch.stack([
        torch.tensor(sess_user.values, dtype=torch.long),
        torch.tensor(sess_user.index.values, dtype=torch.long)
    ], dim=0)
    data[('user','has','session')].edge_index = edge_u2s

    # session -> product edges: one per event
    e_src = torch.tensor(df2['s_idx'].values, dtype=torch.long)
    e_dst = torch.tensor(df2['p_idx'].values, dtype=torch.long)
    data[('session','contains','product')].edge_index = torch.stack([e_src, e_dst], dim=0)

    # Edge attributes on session->product
    data[('session','contains','product')].edge_attr = torch.stack([
        torch.tensor(df2['et_idx'].values, dtype=torch.long),                # categorical (will embed)
        torch.tensor(df2['pos_norm'].values, dtype=torch.float),             # [0,1]
        torch.tensor(df2['dt_log'].values, dtype=torch.float),               # log seconds
    ], dim=1)

    # product -> category (static)
    # Use unique pairs
    pc = df2[['p_idx','c_idx']].drop_duplicates()
    data[('product','in','category')].edge_index = torch.tensor(pc.values.T, dtype=torch.long)

    # Label on session nodes
    data['session'].y = y

    # Optional: make graph undirected (adds reverse rels automatically with ToUndirected)
    if undirected:
        data = ToUndirected()(data)

    meta = {
        'user_id_map': user_map,
        'session_id_map': sess_map,
        'product_id_map': prod_map,
        'category_id_map': cat_map,
        'event_type_map': et_map,
    }
    return data, meta

# ------------------------
# 3) Model: Hetero GAT with learnable node embeddings + edge attr fusion
# ------------------------

class EdgeMLP(nn.Module):
    def __init__(self, et_vocab: int, et_emb_dim: int = 8, out_dim: int = 32):
        super().__init__()
        self.et_emb = nn.Embedding(et_vocab, et_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(et_emb_dim + 2, out_dim),  # + pos_norm + dt_log
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, edge_attr):
        et = edge_attr[:, 0].long()
        pos = edge_attr[:, 1:2].float()
        dt  = edge_attr[:, 2:3].float()
        x = torch.cat([self.et_emb(et), pos, dt], dim=1)
        return self.mlp(x)


class HeteroGAT(nn.Module):
    def __init__(self, metadata, hidden_dim=64, heads=4, node_emb_dims=None, et_vocab=10):
        super().__init__()
        node_emb_dims = node_emb_dims or {
            'user': 32,
            'session': 32,
            'product': 64,
            'category': 16,
        }
        # Learnable node ID embeddings per type
        self.node_emb = nn.ModuleDict({})
        for ntype in ['user','session','product','category']:
            # We'll initialize embeddings at runtime when we see num_nodes
            self.node_emb[ntype] = None
        self.node_emb_dims = node_emb_dims

        # Edge encoder for ('session','contains','product')
        self.edge_enc = EdgeMLP(et_vocab=et_vocab, et_emb_dim=8, out_dim=hidden_dim)

        # Two-layer HeteroConv with GAT on each relation
        self.conv1 = HeteroConv({
            ('user','has','session'): GATConv((-1, -1), hidden_dim, heads=heads, add_self_loops=False),
            ('session','contains','product'): GATConv((-1, -1), hidden_dim, heads=heads, add_self_loops=False),
            ('product','in','category'): GATConv((-1, -1), hidden_dim, heads=heads, add_self_loops=False),
            # reverse relations will be added by ToUndirected; HeteroConv can infer if present
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('user','has','session'): GATConv((-1, -1), hidden_dim, heads=1, add_self_loops=False),
            ('session','contains','product'): GATConv((-1, -1), hidden_dim, heads=1, add_self_loops=False),
            ('product','in','category'): GATConv((-1, -1), hidden_dim, heads=1, add_self_loops=False),
        }, aggr='sum')

        # Session readout head (binary)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def init_node_embeddings(self, data: HeteroData):
        # Lazily create nn.Embedding tables matching num_nodes per type
        for ntype, dim in self.node_emb_dims.items():
            num = int(data[ntype].num_nodes)
            emb = nn.Embedding(num, dim)
            nn.init.xavier_uniform_(emb.weight)
            self.node_emb[ntype] = emb

    def forward(self, data: HeteroData) -> torch.Tensor:
        if self.node_emb['user'] is None:
            self.init_node_embeddings(data)
        x_dict = {
            'user': self.node_emb['user'](torch.arange(data['user'].num_nodes, device=data['session'].y.device)),
            'session': self.node_emb['session'](torch.arange(data['session'].num_nodes, device=data['session'].y.device)),
            'product': self.node_emb['product'](torch.arange(data['product'].num_nodes, device=data['session'].y.device)),
            'category': self.node_emb['category'](torch.arange(data['category'].num_nodes, device=data['session'].y.device)),
        }

        # Edge attributes encoding for session->product relation
        e_attr = data[('session','contains','product')].edge_attr.to(data['session'].y.device)
        e_feat = self.edge_enc(e_attr)

        # HeteroConv requires passing x_dict and edge_index dict; we can pass edge_attr via kwargs per relation
        x_dict = self.conv1(x_dict, {
            ('user','has','session'): data[('user','has','session')].edge_index,
            ('session','contains','product'): data[('session','contains','product')].edge_index,
            ('product','in','category'): data[('product','in','category')].edge_index,
        }, edge_attr={('session','contains','product'): e_feat})
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, {
            ('user','has','session'): data[('user','has','session')].edge_index,
            ('session','contains','product'): data[('session','contains','product')].edge_index,
            ('product','in','category'): data[('product','in','category')].edge_index,
        }, edge_attr={('session','contains','product'): e_feat})
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        # Session logits
        logits = self.head(x_dict['session']).squeeze(-1)
        return logits

# ------------------------
# 4) Train / Eval with time-based split and NeighborLoader
# ------------------------

def build_time_splits(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    # Work at session level to avoid leakage
    sess_grp = df.groupby('user_session')['event_time'].min().sort_values()
    n = len(sess_grp)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))
    train_sessions = set(sess_grp.index[:i_train])
    val_sessions   = set(sess_grp.index[i_train:i_val])
    test_sessions  = set(sess_grp.index[i_val:])
    return train_sessions, val_sessions, test_sessions


def mask_from_sessions(data: HeteroData, sess_map: Dict[str,int], split_sets: Tuple[set,set,set]):
    inv_map = {v:k for k,v in sess_map.items()}
    # session nodes are 0..N-1 in the same order as sess_map values
    idx2id = [inv_map[i] for i in range(len(inv_map))]
    train_s, val_s, test_s = split_sets
    train_mask = torch.tensor([sid in train_s for sid in idx2id], dtype=torch.bool)
    val_mask   = torch.tensor([sid in val_s for sid in idx2id], dtype=torch.bool)
    test_mask  = torch.tensor([sid in test_s for sid in idx2id], dtype=torch.bool)
    data['session'].train_mask = train_mask
    data['session'].val_mask   = val_mask
    data['session'].test_mask  = test_mask
    return data


def train_one_epoch(model, data, optimizer, loader=None, device='cpu'):
    model.train()
    if loader is None:
        optimizer.zero_grad()
        logits = model(data)
        y = data['session'].y.to(device)
        loss = F.binary_cross_entropy_with_logits(logits[data['session'].train_mask], y[data['session'].train_mask])
        loss.backward()
        optimizer.step()
        return float(loss.item())
    else:
        # Minibatch with NeighborLoader (node classification on 'session')
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            y = batch['session'].y
            loss = F.binary_cross_entropy_with_logits(logits[batch['session'].train_mask], y[batch['session'].train_mask])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return total_loss / max(1, len(loader))


def evaluate(model, data, mask_name='val_mask'):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        y = data['session'].y
        mask = data['session'][mask_name]
        prob = torch.sigmoid(logits)
        # Simple metrics: LogLoss, AUC (fallback if sklearn missing)
        # LogLoss
        eps = 1e-7
        p = torch.clamp(prob[mask], eps, 1-eps)
        y_true = y[mask]
        logloss = - (y_true*torch.log(p) + (1-y_true)*torch.log(1-p)).mean().item()
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true.cpu().numpy(), prob[mask].cpu().numpy())
        except Exception:
            auc = float('nan')
    return {'logloss': logloss, 'auc': auc}

# ------------------------
# 5) End-to-end driver (skeleton)
# ------------------------

def run_pipeline(csv_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 hidden_dim: int = 64,
                 heads: int = 4,
                 epochs: int = 10,
                 use_neighborloader: bool = False,
                 batch_size: int = 16384,
                 num_neighbors: int = 10):
    df = load_events(csv_path)

    # Time split at session level
    train_s, val_s, test_s = build_time_splits(df, train_ratio=0.7, val_ratio=0.15)

    # Build full graph (we keep all nodes; masks handle split)
    data, meta = build_hetero_graph(df, undirected=True)

    # Masks
    data = mask_from_sessions(data, meta['session_id_map'], (train_s, val_s, test_s))

    # Model
    et_vocab = len(meta['event_type_map'])
    model = HeteroGAT(metadata=data.metadata(), hidden_dim=hidden_dim, heads=heads, et_vocab=et_vocab)
    model = model.to(device)
    data = data.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Optional neighbor loader for node classification on 'session'
    if use_neighborloader:
        loader = NeighborLoader(
            data,
            num_neighbors={key: [num_neighbors, num_neighbors] for key in data.edge_types},
            input_nodes=('session', data['session'].train_mask),
            batch_size=batch_size,
            shuffle=True,
        )
    else:
        loader = None

    best = {'val_auc': -1, 'state': None}
    for epoch in range(1, epochs+1):
        loss = train_one_epoch(model, data, optim, loader=loader, device=device)
        val_metrics = evaluate(model, data, 'val_mask')
        test_metrics = evaluate(model, data, 'test_mask')
        if val_metrics['auc'] > best['val_auc']:
            best['val_auc'] = val_metrics['auc']
            best['state'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | loss {loss:.4f} | val AUC {val_metrics['auc']:.4f} | test AUC {test_metrics['auc']:.4f}")

    # Restore best
    if best['state'] is not None:
        model.load_state_dict(best['state'])

    return model, data, meta

# Example usage (uncomment and set your path):
model, data, meta = run_pipeline('./data/train.csv', epochs=20, use_neighborloader=False)
