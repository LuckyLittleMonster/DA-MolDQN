"""
GNN-based DQN for molecular optimization.

Architecture:
    Frozen GNN encoder (pretrained LargeQEDPredictor) → 1392-dim embedding
    + step fraction → 1393-dim
    → Trainable Q-head MLP → Q-value

The GNN encoder is pretrained on ZINC250K for QED prediction (val_loss=0.0008).
During RL, only the Q-head is trained. The encoder provides rich molecular
representations that capture structure-property relationships.

Usage:
    from gnn_dqn import GNNDQN, smiles_to_graph

    model = GNNDQN(pretrained_path="path/to/checkpoint.pt", device="cuda")
    # Encode molecules → observations (same interface as fingerprint approach)
    obs = model.encode_molecules(["CCO", "c1ccccc1"], step_fraction=0.5)
    q_values = model(obs)  # trainable Q-head only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from dqn import BaseDQN
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv, GINConv, GINEConv, PNAConv,
    global_mean_pool, global_max_pool, global_add_pool,
)
from torch_geometric.data import Data, Batch


# ============================================================================
# Graph construction (matching gnn_pretrain_qed.py)
# ============================================================================

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to torch_geometric Data with normalized features.

    Atom features (12-dim): atomic_num, degree, formal_charge, hybridization,
        is_aromatic, total_Hs, valence, mass, radical_electrons, chiral_tag,
        is_in_ring, implicit_valence.
    Edge features (6-dim): bond_type, is_conjugated, is_in_ring, stereo,
        is_aromatic, bond_dir.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum() / 100.0,
                atom.GetDegree() / 6.0,
                (atom.GetFormalCharge() + 2.0) / 4.0,
                atom.GetHybridization().real / 6.0,
                float(atom.GetIsAromatic()),
                atom.GetTotalNumHs(includeNeighbors=False) / 4.0,
                (atom.GetExplicitValence() + atom.GetImplicitValence()) / 6.0,
                atom.GetMass() / 200.0,
                float(atom.GetNumRadicalElectrons()),
                atom.GetChiralTag().real / 3.0,
                float(atom.IsInRing()),
                atom.GetImplicitValence() / 6.0,
            ]
            atom_features.append(features)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
            bond_features = [
                bond.GetBondType().real / 3.0,
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
                bond.GetStereo().real / 6.0,
                float(bond.GetIsAromatic()),
                bond.GetBondDir().real / 6.0,
            ]
            edge_attrs.extend([bond_features, bond_features])

        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.zeros(0, 6, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception:
        return None


# ============================================================================
# GNN Encoder (matching pretrained LargeQEDPredictor architecture)
# ============================================================================

class GNNMolEncoder(nn.Module):
    """
    GNN molecule encoder matching LargeQEDPredictor from gnn_pretrain_qed.py.

    Architecture (from pretrained checkpoint):
        hidden_dim=348, num_layers=12, num_heads=12, ff_dim_multiplier=2
        Layers cycle: PNA → TransformerConv → GAT → GINE → GCN (×2.4)
        4-way pooling (mean/max/add/std) → 1392-dim output

    Pretrained on ZINC250K for QED prediction (val_loss=0.0008, 28.4M params).
    """

    def __init__(self, hidden_dim=348, num_layers=12, num_heads=12,
                 dropout=0.1, ff_dim_multiplier=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ff_dim = hidden_dim * ff_dim_multiplier
        self.output_dim = hidden_dim * 4  # 4 pooling methods

        # Atom embedding
        self.atom_embedding = nn.Sequential(
            nn.Linear(12, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Feature adapter (6-dim → 12-dim)
        self.feature_adapter = nn.Sequential(
            nn.Linear(6, 12),
            nn.LayerNorm(12),
            nn.GELU(),
        )

        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # GNN layers (cycle: PNA, Transformer, GAT, GINE, GCN)
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.dim_fix_layers = nn.ModuleList()

        deg = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        for i in range(num_layers):
            layer_type = i % 5

            if layer_type == 0:
                aggregators = ['mean', 'max', 'sum', 'std', 'var', 'min']
                scalers = ['identity', 'amplification', 'attenuation']
                self.gnn_layers.append(
                    PNAConv(hidden_dim, hidden_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, train_norm=False))
            elif layer_type == 1:
                self.gnn_layers.append(
                    TransformerConv(hidden_dim, hidden_dim // num_heads,
                                   heads=num_heads, dropout=dropout,
                                   edge_dim=hidden_dim, concat=True))
            elif layer_type == 2:
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads,
                            heads=num_heads, dropout=dropout,
                            edge_dim=hidden_dim, concat=True))
            elif layer_type == 3:
                gin_nn = nn.Sequential(
                    nn.Linear(hidden_dim, self.ff_dim),
                    nn.LayerNorm(self.ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.ff_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                self.gnn_layers.append(
                    GINEConv(gin_nn, edge_dim=hidden_dim))
            else:
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))

            self.batch_norms.append(nn.LayerNorm(hidden_dim))
            self.layer_norms.append(nn.Identity())

            if layer_type in [1, 2]:
                self.dim_fix_layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.dim_fix_layers.append(nn.Identity())

            self.ff_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, self.ff_dim),
                nn.LayerNorm(self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ))

        # Residual projections
        self.residual_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Multi-scale pooling
        self.multi_scale_pooling = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ) for key in ['mean', 'max', 'add', 'std']
        })

        # Graph-level attention
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 4,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, batch) -> torch.Tensor:
        """
        Encode a batch of molecular graphs.

        Args:
            batch: torch_geometric Batch object

        Returns:
            Graph-level embeddings (batch_size, output_dim=1392)
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        edge_attr = getattr(batch, 'edge_attr', None)

        # Input feature handling
        if x.shape[1] == 6:
            x = self.feature_adapter(x)
        elif x.shape[1] != 12:
            raise ValueError(f"Unsupported atom feature dim: {x.shape[1]}")

        x = self.atom_embedding(x)

        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)

        # GNN layers
        for i, layer in enumerate(self.gnn_layers):
            residual = x.clone()

            if isinstance(layer, PNAConv):
                x = layer(x, edge_index)
            elif isinstance(layer, (TransformerConv, GATConv)):
                x = layer(x, edge_index, edge_attr=edge_attr)
            elif isinstance(layer, GINEConv):
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                x = layer(x, edge_index)

            # Fix dimension if needed
            if x.shape[1] != self.hidden_dim:
                if x.shape[1] == self.num_heads * self.hidden_dim:
                    x = x.view(-1, self.num_heads, self.hidden_dim).mean(dim=1)
                elif not isinstance(self.dim_fix_layers[i], nn.Identity):
                    x = self.dim_fix_layers[i](x)
                else:
                    x = F.linear(x, torch.eye(self.hidden_dim, x.shape[1],
                                               device=x.device))

            # Norm
            x = self.batch_norms[i](x)

            # Residual
            x = x + self.residual_proj[i](residual)

            # FFN with residual
            x_res = x
            x = self.ff_layers[i](x)
            x = x + x_res

            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Multi-scale pooling
        mean_pool = global_mean_pool(x, batch_idx)
        max_pool = global_max_pool(x, batch_idx)
        add_pool = global_add_pool(x, batch_idx)
        std_pool = global_mean_pool(
            (x - mean_pool[batch_idx]) ** 2, batch_idx
        ).sqrt()

        mean_feat = self.multi_scale_pooling['mean'](mean_pool)
        max_feat = self.multi_scale_pooling['max'](max_pool)
        add_feat = self.multi_scale_pooling['add'](add_pool)
        std_feat = self.multi_scale_pooling['std'](std_pool)

        graph_features = torch.cat(
            [mean_feat, max_feat, add_feat, std_feat], dim=1
        )

        # Graph-level self-attention
        gf_seq = graph_features.unsqueeze(1)
        attn_out, _ = self.graph_attention(gf_seq, gf_seq, gf_seq)
        graph_features = graph_features + attn_out.squeeze(1)

        return graph_features


# ============================================================================
# GNN-based DQN
# ============================================================================

class GNNDQN(BaseDQN):
    """
    GNN-based DQN: frozen pretrained encoder + trainable Q-head.

    The forward() method takes pre-computed observations (encoder embeddings
    + step fraction), matching the interface of the fingerprint-based DQN.
    This allows the same training loop with minimal changes.

    Encoding is done separately via encode_molecules(), called only during
    action selection (not during training from replay buffer).
    """

    def __init__(self, encoder: GNNMolEncoder, q_head_hidden: int = 512,
                 q_head_dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder.output_dim  # 1392

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # Trainable Q-head: embedding + step_fraction → Q-value
        input_dim = self.encoder_dim + 1  # 1393
        self.q_head = nn.Sequential(
            nn.Linear(input_dim, q_head_hidden),
            nn.LayerNorm(q_head_hidden),
            nn.ReLU(),
            nn.Dropout(q_head_dropout),
            nn.Linear(q_head_hidden, q_head_hidden // 2),
            nn.LayerNorm(q_head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(q_head_dropout),
            nn.Linear(q_head_hidden // 2, 1),
        )

        self._device = None
        self._graph_cache = {}  # SMILES → Data cache

    @property
    def device(self):
        return next(self.q_head.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Q-head forward pass. Takes pre-computed observations.

        Args:
            x: (batch, encoder_dim + 1) — concatenation of GNN embedding
               and step fraction. Same interface as fingerprint DQN.

        Returns:
            Q-values (batch, 1)
        """
        return self.q_head(x)

    def train(self, mode=True):
        """Override to keep encoder always in eval mode."""
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode_molecules(self, smiles_list: List[str],
                         step_fraction: float) -> np.ndarray:
        """
        Encode molecules to observation vectors (for action selection).

        Args:
            smiles_list: List of SMILES strings
            step_fraction: Current step / max_steps (scalar)

        Returns:
            observations: np.ndarray of shape (N, encoder_dim + 1)
        """
        device = self.device
        graphs = []
        valid_mask = []

        for smi in smiles_list:
            if smi in self._graph_cache:
                g = self._graph_cache[smi]
            else:
                g = smiles_to_graph(smi)
                if len(self._graph_cache) < 50000:
                    self._graph_cache[smi] = g

            if g is not None:
                graphs.append(g)
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not graphs:
            return np.zeros((len(smiles_list), self.encoder_dim + 1),
                            dtype=np.float32)

        batch = Batch.from_data_list(graphs).to(device)
        embeddings = self.encoder(batch).cpu().numpy()  # (N_valid, 1392)

        # Build full output including invalid molecules (zero embedding)
        result = np.zeros((len(smiles_list), self.encoder_dim + 1),
                          dtype=np.float32)
        valid_idx = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                result[i, :self.encoder_dim] = embeddings[valid_idx]
                valid_idx += 1
            result[i, -1] = step_fraction

        return result


def load_pretrained_encoder(
    checkpoint_path: str = "/shared/data1/Users/l1062811/git/marl/checkpoints/large_pretrained_qed_predictor.pt",
    device: str = 'cpu',
    **kwargs,
) -> GNNMolEncoder:
    """
    Load pretrained GNNMolEncoder from LargeQEDPredictor checkpoint.

    Args:
        checkpoint_path: Path to large_pretrained_qed_predictor.pt
        device: Device to load on

    Returns:
        GNNMolEncoder with pretrained weights (eval mode)
    """
    encoder = GNNMolEncoder(**kwargs)

    state_dict = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)

    # Filter out predictor keys — we only need encoder layers
    encoder_keys = {k: v for k, v in state_dict.items()
                    if not k.startswith('predictor')}

    # Load with strict=False to handle minor mismatches
    missing, unexpected = encoder.load_state_dict(encoder_keys, strict=False)

    if missing:
        print(f"  Warning: {len(missing)} missing keys in encoder "
              f"(first 5: {missing[:5]})")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys "
              f"(first 5: {unexpected[:5]})")

    encoder.eval()
    total_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"  GNNMolEncoder loaded: {total_params:.1f}M params")

    return encoder


def create_gnn_dqn(
    checkpoint_path: str = "/shared/data1/Users/l1062811/git/marl/checkpoints/large_pretrained_qed_predictor.pt",
    device: str = 'cpu',
    q_head_hidden: int = 512,
) -> GNNDQN:
    """
    Create GNNDQN with pretrained encoder.

    Args:
        checkpoint_path: Path to pretrained encoder checkpoint
        device: Device
        q_head_hidden: Hidden dim for Q-head

    Returns:
        GNNDQN model on the specified device
    """
    print("Loading pretrained GNN encoder...")
    encoder = load_pretrained_encoder(checkpoint_path, device=device)
    model = GNNDQN(encoder, q_head_hidden=q_head_hidden)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  GNNDQN: {trainable/1e6:.2f}M trainable, "
          f"{frozen/1e6:.1f}M frozen")

    return model


if __name__ == '__main__':
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = create_gnn_dqn(device=device)

    test_smiles = ["CCO", "c1ccccc1Br", "CC(=O)O", "c1ccccc1N",
                    "CC(C)(C)OC(=O)NC(=O)OC(C)(C)C"]

    # Test encoding
    start = time.time()
    obs = model.encode_molecules(test_smiles, step_fraction=0.4)
    enc_time = time.time() - start
    print(f"\nEncoded {len(test_smiles)} molecules: {obs.shape}")
    print(f"  Encoding time: {enc_time*1000:.1f}ms")

    # Test Q-head
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    q_values = model(obs_tensor)
    print(f"  Q-values: {q_values.squeeze().tolist()}")

    # Test cache speed
    start = time.time()
    for _ in range(10):
        obs2 = model.encode_molecules(test_smiles, step_fraction=0.6)
    cache_time = (time.time() - start) / 10
    print(f"  Cached encoding time: {cache_time*1000:.1f}ms")

    print("\n=== DONE ===")
