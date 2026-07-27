"""Graph encoder for the frozen property teacher (see src/models/gnn_teacher.py).

Copied verbatim from rep_gnn/models.py so that checkpoints trained there
(ckpt/gnn_*.pt) load without conversion. Only the message-passing OPERATOR differs
across archs; the atom/bond embeddings, residual+BatchNorm stack and 3-way pooling are
held fixed -- rep_gnn Follow-up L measured the operator to carry no RL signal once the
teacher is above the quality knee, so this is deliberately not a tuning surface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GATv2Conv, GCNConv, SAGEConv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool)


def _make_conv(arch, hidden, heads=4):
    """One message-passing layer. Only the OPERATOR changes across archs -- the atom/bond
    embeddings, residual connections, normalisation, dropout and the 3-way pooling are held
    fixed, so an arch comparison isolates the operator rather than the surrounding recipe.
    Returns (module, uses_edge_attr)."""
    if arch == "gine":
        mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(),
                            nn.Linear(2 * hidden, hidden))
        return GINEConv(mlp, edge_dim=hidden, train_eps=True), True
    if arch == "gat":
        return GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=hidden), True
    if arch == "transformer":
        return TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden), True
    if arch == "gcn":
        return GCNConv(hidden, hidden), False
    if arch == "sage":
        return SAGEConv(hidden, hidden), False
    raise ValueError(f"unknown arch {arch}")


class GNNEncoder(nn.Module):
    """GINE message-passing encoder over (12-d atom, 6-d bond) graphs.

    Returns a graph-level embedding of size 2*hidden (mean ++ max pool).
    Kept separable so the same encoder can later back an RL Q-head.
    """

    def __init__(self, hidden=256, num_layers=6, dropout=0.1, arch="gine"):
        super().__init__()
        self.hidden = hidden
        self.arch = arch
        self.out_dim = 3 * hidden  # mean ++ max ++ add pooling
        self.atom_emb = nn.Linear(12, hidden)
        self.edge_emb = nn.Linear(6, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_edge = []
        for _ in range(num_layers):
            conv, ue = _make_conv(arch, hidden)
            self.convs.append(conv); self.use_edge.append(ue)
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

    def forward(self, data):
        x = self.atom_emb(data.x)
        ea = self.edge_emb(data.edge_attr)
        for conv, norm, ue in zip(self.convs, self.norms, self.use_edge):
            h = conv(x, data.edge_index, ea) if ue else conv(x, data.edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual
        # mean (intensive), max (salient), add (extensive/count -> MW, HBA/HBD)
        g = torch.cat([global_mean_pool(x, data.batch),
                       global_max_pool(x, data.batch),
                       global_add_pool(x, data.batch)], dim=1)
        return g


class GNNQED(nn.Module):
    """GNN encoder + regression head -> scalar QED (or Q-value)."""

    def __init__(self, hidden=256, num_layers=6, dropout=0.1, head_hidden=256,
                 bounded=False, arch="gine"):
        super().__init__()
        self.encoder = GNNEncoder(hidden, num_layers, dropout, arch=arch)
        self.bounded = bounded  # sigmoid output for QED in [0,1] (supervised)
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Linear(self.encoder.out_dim, head_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.ReLU(),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, data):
        out = self.head(self.encoder(data))
        return torch.sigmoid(out) if self.bounded else out

