"""
Hypergraph Link Predictor v3: Comprehensive Optimization.

Key improvements over v1/v2:
  Phase 1: Enhanced molecular encoding + reaction center
    - Rich 2D atom features (12+ dims) and bond features (6+ dims)
    - Reaction center atom-level tags from atom mapping
    - Multi-edge reaction graph with soft labels and edge weights
    - Improved RCNS negative sampling
  Phase 2: 3D conformer + frozen MLIP encoder (optional)
    - Gated cross-modal fusion of 2D and 3D embeddings
    - 3D dropout for robustness when conformers unavailable
  Phase 3: Reaction-aware contrastive pre-training support
    - Encoder weights can be loaded from pretrained checkpoint
  Phase 4: Multi-task learning
    - Auxiliary reaction class prediction head
    - Property prediction head (yield, barrier, rate)

Usage:
    # Phase 1 only (no 3D):
    python -m model_reactions.hypergraph_link_predictor_v3 --train --data_dir Data/uspto

    # With pretrained encoder:
    python -m model_reactions.hypergraph_link_predictor_v3 --train --data_dir Data/uspto \\
        --pretrained_encoder model_reactions/checkpoints/pretrained_encoder.pt

    # With 3D features:
    python -m model_reactions.hypergraph_link_predictor_v3 --train --data_dir Data/uspto \\
        --use_3d --conformer_cache Data/precomputed/conformers.pt
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdChemReactions
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model_reactions.reactant_predictor import ReactionDatabase
from model_reactions.link_prediction.reaction_center import (
    extract_reaction_centers,
    heuristic_reaction_center_scores,
)


# =============================================================================
# Configuration
# =============================================================================

# Atom feature dimensions (when one-hot encoded)
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),       # 1-118
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'chirality': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}


def _one_hot(value, choices):
    """One-hot encode a value given a list of choices."""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # unknown
    return encoding


def compute_atom_feature_dim():
    """Compute the total atom feature dimension."""
    dim = 0
    for key, choices in ATOM_FEATURES.items():
        dim += len(choices) + 1  # +1 for unknown
    dim += 3  # is_aromatic, is_in_ring, num_radical_electrons
    dim += 1  # reaction center score
    return dim


def compute_bond_feature_dim():
    """Compute the total bond feature dimension."""
    dim = 0
    for key, choices in BOND_FEATURES.items():
        dim += len(choices) + 1
    dim += 2  # is_conjugated, is_in_ring
    return dim


ATOM_FEATURE_DIM = compute_atom_feature_dim()
BOND_FEATURE_DIM = compute_bond_feature_dim()


@dataclass
class HypergraphLinkConfigV3:
    """Configuration for Hypergraph Link Predictor v3."""
    # Atom/Bond feature dims (auto-computed)
    atom_feature_dim: int = ATOM_FEATURE_DIM
    bond_feature_dim: int = BOND_FEATURE_DIM

    # GNN Encoder
    atom_dim: int = 64
    edge_dim: int = 32
    encoder_hidden_dim: int = 256
    encoder_layers: int = 4
    mol_embedding_dim: int = 256

    # Link predictor
    hidden_dim: int = 256
    num_link_layers: int = 2
    dropout: float = 0.1

    # Training
    temperature: float = 0.07
    label_smoothing: float = 0.05

    # Multi-task
    num_rxn_classes: int = 10
    rxn_class_weight: float = 0.3
    property_weight: float = 0.2

    # RCNS
    rcns_ratio: float = 0.5

    # 3D (optional)
    use_3d: bool = False
    frozen_3d_dim: int = 512  # output dim of frozen 3D encoder
    dropout_3d: float = 0.3   # probability of zeroing 3D embedding during training

    # NCN (Neural Common Neighbor)
    use_ncn: bool = False
    ncn_max_neighbors: int = 32  # max common neighbors to aggregate per pair

    # Topological heuristic features (Scheme 1)
    use_topo: bool = False
    topo_feature_dim: int = 7  # CN, AA, RA, JC, PA, SP_inv, Katz

    # Subgraph sketching (Scheme 2 — BUDDY-style)
    use_sketch: bool = False
    sketch_dim: int = 128       # MinHash signature dimension
    sketch_hops: int = 2        # number of hop levels (1-hop, 2-hop)
    sketch_mlp_hidden: int = 64 # hidden dim for sketch projection MLP
    sketch_out_dim: int = 64    # output dim of sketch MLP

    # CF-NCN (Collaborative Filtering NCN — Scheme 4)
    use_cf_ncn: bool = False
    cf_ncn_k: int = 200           # kNN neighbors per molecule
    cf_ncn_max_bridge: int = 16   # max bridge molecules per direction (32 total)
    cf_ncn_fp_bits: int = 2048    # Morgan FP bits for kNN
    cf_ncn_fp_radius: int = 2     # Morgan FP radius
    cf_ncn_asymmetric: bool = True  # separate aggregators for AB and BA directions
    cf_ncn_query_interact: bool = True  # add query-bridge interaction features
    cf_ncn_use_count: bool = True  # inject bridge count as scalar feature

    @classmethod
    def small(cls):
        return cls(atom_dim=32, edge_dim=16, encoder_hidden_dim=128,
                   encoder_layers=2, mol_embedding_dim=128, hidden_dim=128,
                   num_link_layers=1)

    @classmethod
    def medium(cls):
        return cls()

    @classmethod
    def large(cls):
        return cls(atom_dim=128, edge_dim=64, encoder_hidden_dim=512,
                   encoder_layers=5, mol_embedding_dim=512, hidden_dim=512,
                   num_link_layers=3)


# =============================================================================
# Rich Molecular Graph Features
# =============================================================================

def smiles_to_rich_graph(smiles: str, rxn_center_scores: Optional[np.ndarray] = None):
    """
    Convert SMILES to graph with rich atom and bond features.

    Args:
        smiles: SMILES string
        rxn_center_scores: Optional per-atom reaction center scores (n_atoms,)

    Returns:
        dict with 'atom_features', 'edge_index', 'bond_features', 'n_atoms'
        or None if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    # Atom features
    atom_features = []
    for i, atom in enumerate(mol.GetAtoms()):
        feats = []
        feats.extend(_one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
        feats.extend(_one_hot(atom.GetDegree(), ATOM_FEATURES['degree']))
        feats.extend(_one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
        feats.extend(_one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
        feats.extend(_one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
        feats.extend(_one_hot(atom.GetChiralTag(), ATOM_FEATURES['chirality']))
        feats.append(float(atom.GetIsAromatic()))
        feats.append(float(atom.IsInRing()))
        feats.append(float(atom.GetNumRadicalElectrons()))
        # Reaction center score
        if rxn_center_scores is not None and i < len(rxn_center_scores):
            feats.append(float(rxn_center_scores[i]))
        else:
            feats.append(0.0)
        atom_features.append(feats)

    # Bond features
    edge_index = []
    bond_features = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = []
        feats.extend(_one_hot(bond.GetBondType(), BOND_FEATURES['bond_type']))
        feats.extend(_one_hot(bond.GetStereo(), BOND_FEATURES['stereo']))
        feats.append(float(bond.GetIsConjugated()))
        feats.append(float(bond.IsInRing()))

        # Add both directions
        edge_index.extend([[i, j], [j, i]])
        bond_features.extend([feats, feats])

    if not edge_index:
        # Self-loop for isolated atoms
        edge_index = [[0, 0]]
        bond_features = [[0.0] * BOND_FEATURE_DIM]

    return {
        'atom_features': atom_features,
        'edge_index': edge_index,
        'bond_features': bond_features,
        'n_atoms': n_atoms,
    }


# =============================================================================
# GNN Molecule Encoder (v3 - accepts feature vectors)
# =============================================================================

class MPNNLayerV3(nn.Module):
    """Message Passing Neural Network layer with rich features."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_i, x_j = x[row], x[col]

        messages = self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

        num_nodes = x.size(0)
        aggr = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        count = torch.zeros(num_nodes, 1, device=x.device)

        aggr.scatter_add_(0, col.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
        count.scatter_add_(0, col.unsqueeze(-1), torch.ones_like(col, dtype=torch.float).unsqueeze(-1))
        aggr = aggr / count.clamp(min=1)

        out = self.update_mlp(torch.cat([x, aggr], dim=-1))
        return self.norm(x + self.dropout(out))


class MoleculeEncoderV3(nn.Module):
    """GNN encoder for molecules with rich features."""

    def __init__(self, config: HypergraphLinkConfigV3):
        super().__init__()
        # Linear projections for feature vectors (not embeddings for integers)
        self.atom_proj = nn.Linear(config.atom_feature_dim, config.atom_dim)
        self.input_proj = nn.Linear(config.atom_dim, config.encoder_hidden_dim)
        self.edge_proj = nn.Linear(config.bond_feature_dim, config.edge_dim)

        self.layers = nn.ModuleList([
            MPNNLayerV3(config.encoder_hidden_dim, config.edge_dim, config.dropout)
            for _ in range(config.encoder_layers)
        ])

        self.output_proj = nn.Linear(config.encoder_hidden_dim, config.mol_embedding_dim)
        self.hidden_dim = config.encoder_hidden_dim

    def forward(self, atom_features, edge_index, bond_features, batch_idx):
        """
        Args:
            atom_features: (num_atoms, atom_feature_dim) float tensor
            edge_index: (2, num_edges) long tensor
            bond_features: (num_edges, bond_feature_dim) float tensor
            batch_idx: (num_atoms,) long tensor
        """
        x = self.input_proj(self.atom_proj(atom_features))
        edge_attr = self.edge_proj(bond_features)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Global mean pooling
        batch_size = batch_idx.max().item() + 1
        out = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        count = torch.zeros(batch_size, 1, device=x.device)

        out.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, self.hidden_dim), x)
        count.scatter_add_(0, batch_idx.unsqueeze(-1), torch.ones_like(batch_idx, dtype=torch.float).unsqueeze(-1))

        return self.output_proj(out / count.clamp(min=1))


# =============================================================================
# Cross-Modal Fusion (2D + 3D)
# =============================================================================

class CrossModalFusion(nn.Module):
    """Gated fusion of 2D and 3D molecular embeddings."""

    def __init__(self, d_2d: int, d_3d: int, d_out: int):
        super().__init__()
        self.proj_2d = nn.Linear(d_2d, d_out)
        self.proj_3d = nn.Linear(d_3d, d_out)
        self.gate = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.Sigmoid()
        )

    def forward(self, emb_2d: torch.Tensor, emb_3d: Optional[torch.Tensor] = None):
        """
        Args:
            emb_2d: (B, d_2d) 2D molecular embedding
            emb_3d: (B, d_3d) 3D molecular embedding, or None

        Returns:
            fused: (B, d_out) fused embedding
        """
        h2 = self.proj_2d(emb_2d)
        if emb_3d is None:
            return h2
        h3 = self.proj_3d(emb_3d)
        g = self.gate(torch.cat([h2, h3], dim=-1))
        return h2 + g * h3


# =============================================================================
# Topological Heuristic Features (Scheme 1)
# =============================================================================

def compute_topo_features(edges, n_nodes, all_edges_for_adj):
    """
    Precompute topological heuristic features for a set of node pairs.

    Uses only the edges in all_edges_for_adj to build the graph (typically
    train edges only, to avoid data leakage). For each pair (a,b), the
    target edge (a,b) is REMOVED from the adjacency before computing
    features, preventing label leakage.

    Args:
        edges: list of (a, b) pairs to compute features for
        n_nodes: total number of nodes in the graph
        all_edges_for_adj: edges to build adjacency from (train edges)

    Returns:
        topo_dict: Dict[(a,b)] -> np.ndarray of shape (7,)
            Features: [CN, AA, RA, JC, PA, SP_inv, Katz]
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    # Build adjacency list for efficient neighbor lookups
    adj_list = defaultdict(set)
    edge_set = set()
    for a, b in all_edges_for_adj:
        adj_list[a].add(b)
        adj_list[b].add(a)
        edge_set.add((min(a, b), max(a, b)))

    # Build sparse adjacency matrix for shortest path computation
    rows, cols = [], []
    for a, b in all_edges_for_adj:
        rows.extend([a, b])
        cols.extend([b, a])
    data = np.ones(len(rows), dtype=np.float32)
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Precompute degrees (will be adjusted per-pair for positive edges)
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()

    # Precompute A^2 for Katz
    adj_sq = adj_matrix.dot(adj_matrix)

    # Collect unique nodes for shortest path subset
    edge_nodes = set()
    for a, b in edges:
        edge_nodes.add(a)
        edge_nodes.add(b)
    edge_nodes = sorted(edge_nodes)
    node_to_sp_idx = {n: i for i, n in enumerate(edge_nodes)}

    # Compute shortest paths on the FULL graph
    print(f"  Computing shortest paths for {len(edge_nodes)} nodes...")
    sub_adj = adj_matrix[edge_nodes][:, edge_nodes]
    sp_matrix = shortest_path(sub_adj, directed=False, unweighted=True)
    sp_matrix[sp_matrix == np.inf] = n_nodes

    # Compute features for each pair
    topo_dict = {}
    beta_katz = 0.01

    for a, b in edges:
        key = (min(a, b), max(a, b))
        is_positive = key in edge_set  # True if target edge exists in graph

        # Get neighbor sets; remove target endpoints to avoid leakage
        a_neighbors = adj_list.get(a, set()).copy()
        b_neighbors = adj_list.get(b, set()).copy()
        if is_positive:
            a_neighbors.discard(b)  # Remove b from a's neighbors
            b_neighbors.discard(a)  # Remove a from b's neighbors

        # Adjusted degrees (subtract 1 for each endpoint if edge exists)
        deg_a = degrees[a] - (1 if is_positive else 0)
        deg_b = degrees[b] - (1 if is_positive else 0)

        # Common neighbors (on graph without target edge)
        common = a_neighbors & b_neighbors
        cn = len(common)

        # Adamic-Adar
        aa = 0.0
        for z in common:
            dz = degrees[z]
            if dz > 1:
                aa += 1.0 / np.log(dz)

        # Resource Allocation
        ra = 0.0
        for z in common:
            dz = degrees[z]
            if dz > 0:
                ra += 1.0 / dz

        # Jaccard Coefficient
        union = len(a_neighbors | b_neighbors)
        jc = cn / max(union, 1)

        # Preferential Attachment (adjusted degrees)
        pa = deg_a * deg_b

        # Shortest Path (with edge removed for positive pairs)
        sp_a = node_to_sp_idx.get(a, -1)
        sp_b = node_to_sp_idx.get(b, -1)
        if sp_a >= 0 and sp_b >= 0:
            dist = sp_matrix[sp_a, sp_b]
            if is_positive:
                # Edge removed: SP >= 2 if common neighbor exists, else check
                if cn > 0:
                    dist = 2.0  # shortest via common neighbor
                else:
                    # Need to find SP without direct edge; approximate
                    dist = max(dist + 1, 2.0) if dist <= 1.5 else dist
            sp_inv = 1.0 / max(dist, 1.0)
        else:
            sp_inv = 0.0

        # Katz index (with edge removed)
        paths_1 = adj_matrix[a, b] if not is_positive else 0
        paths_2 = adj_sq[a, b]
        if is_positive:
            # A^2[a,b] counts 2-hop paths, but includes paths through direct edge
            # Correction: subtract contributions through the removed edge
            # Path a->b->b (self-loop, doesn't exist) and a->a->b (self-loop)
            # The 2-hop path a->x->b is fine; remove a->b->x->... no, A^2 only counts
            # 2-hop paths. The direct edge contributes to A^2 via paths a->b->z->...
            # but A^2[a,b] = sum_z A[a,z]*A[z,b], so removing edge (a,b):
            # subtract A[a,b]*A[b,b] + A[a,a]*A[a,b] = 0 (no self-loops)
            # Actually paths_2 through z: if z=a or z=b, A[a,a]=0, A[b,b]=0
            # So A^2[a,b] doesn't include the direct edge. paths_2 is fine.
            pass
        katz = beta_katz * paths_1 + beta_katz ** 2 * paths_2

        topo_dict[key] = np.array(
            [cn, aa, ra, jc, pa, sp_inv, katz], dtype=np.float32
        )

    return topo_dict


def normalize_topo_features(topo_dict):
    """Normalize topological features to zero-mean, unit-variance."""
    if not topo_dict:
        return topo_dict, None, None

    all_feats = np.stack(list(topo_dict.values()))
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std[std < 1e-8] = 1.0

    normalized = {}
    for k, v in topo_dict.items():
        normalized[k] = (v - mean) / std

    return normalized, mean, std


# =============================================================================
# Subgraph Sketching (Scheme 2 — BUDDY-style)
# =============================================================================

def _hash_node(node_id, seed, large_prime=2147483647):
    """Simple hash function for MinHash: (a*x + b) mod p."""
    a = seed * 6364136223846793005 + 1
    b = seed * 1442695040888963407 + 1
    return ((a * node_id + b) % large_prime)


def compute_minhash_sketches(edges, n_nodes, sketch_dim=128, n_hops=2):
    """
    Compute MinHash sketches for k-hop neighborhoods of all nodes.

    For each node, the sketch is a vector of length sketch_dim where each
    element is the minimum hash value over the neighborhood for that hash
    function. This provides a compact, fixed-size representation of the
    neighborhood that supports efficient Jaccard similarity estimation.

    Args:
        edges: list of (a, b) edges to build adjacency
        n_nodes: total number of nodes
        sketch_dim: number of hash functions / sketch dimension
        n_hops: number of hop levels to compute (1=1-hop only, 2=1-hop+2-hop)

    Returns:
        sketches: dict mapping hop_level -> np.ndarray of shape (n_nodes, sketch_dim)
                  hop_level is 1-indexed (1 = 1-hop neighbors, 2 = 2-hop neighbors)
    """
    # Build adjacency list
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    # Generate random seeds for hash functions
    rng = np.random.RandomState(42)
    seeds = rng.randint(1, 2**31, size=sketch_dim)

    INF_HASH = 2**62

    sketches = {}

    # Compute 1-hop sketches
    print(f"  Computing {sketch_dim}-dim MinHash sketches for {n_nodes} nodes...")
    sketch_1hop = np.full((n_nodes, sketch_dim), INF_HASH, dtype=np.int64)

    for node in range(n_nodes):
        neighbors = adj.get(node, set())
        if not neighbors:
            continue
        for s_idx, seed in enumerate(seeds):
            min_hash = INF_HASH
            for nb in neighbors:
                h = _hash_node(nb, seed)
                if h < min_hash:
                    min_hash = h
            sketch_1hop[node, s_idx] = min_hash

    sketches[1] = sketch_1hop
    print(f"    1-hop: {np.sum(sketch_1hop < INF_HASH)} non-inf entries "
          f"({np.mean(np.any(sketch_1hop < INF_HASH, axis=1))*100:.1f}% nodes have neighbors)")

    if n_hops >= 2:
        # 2-hop: union of 1-hop neighbors of all 1-hop neighbors
        sketch_2hop = np.full((n_nodes, sketch_dim), INF_HASH, dtype=np.int64)

        for node in range(n_nodes):
            neighbors_1 = adj.get(node, set())
            if not neighbors_1:
                continue
            # 2-hop neighborhood = union of neighbors of neighbors (excluding self)
            neighbors_2 = set()
            for nb in neighbors_1:
                neighbors_2.update(adj.get(nb, set()))
            neighbors_2.discard(node)  # exclude self
            neighbors_2 -= neighbors_1  # exclude 1-hop (pure 2-hop only)

            if not neighbors_2:
                # Fall back to 1-hop sketch if no 2-hop neighbors
                sketch_2hop[node] = sketch_1hop[node]
                continue

            for s_idx, seed in enumerate(seeds):
                min_hash = INF_HASH
                for nb in neighbors_2:
                    h = _hash_node(nb, seed)
                    if h < min_hash:
                        min_hash = h
                sketch_2hop[node, s_idx] = min_hash

        sketches[2] = sketch_2hop
        print(f"    2-hop: {np.sum(sketch_2hop < INF_HASH)} non-inf entries "
              f"({np.mean(np.any(sketch_2hop < INF_HASH, axis=1))*100:.1f}% nodes have 2-hop)")

    return sketches


def sketches_to_pair_features(sketches, idx_a, idx_b, device):
    """
    Compute pair-level structural features from node sketches.

    For each pair (a, b), computes element-wise min and element-wise equality
    of their sketches at each hop level. The min approximates set intersection
    (like common neighbors), and equality indicates shared neighborhood elements.

    Args:
        sketches: dict mapping hop_level -> np.ndarray (n_nodes, sketch_dim)
        idx_a: tensor of node indices for side A
        idx_b: tensor of node indices for side B
        device: torch device

    Returns:
        pair_feats: tensor of shape (B, n_hops * sketch_dim * 2)
    """
    idx_a_np = idx_a.numpy() if isinstance(idx_a, torch.Tensor) else np.array(idx_a)
    idx_b_np = idx_b.numpy() if isinstance(idx_b, torch.Tensor) else np.array(idx_b)

    INF_HASH = 2**62
    parts = []

    for hop in sorted(sketches.keys()):
        sk = sketches[hop]
        sk_a = sk[idx_a_np]  # (B, sketch_dim)
        sk_b = sk[idx_b_np]  # (B, sketch_dim)

        # Normalize: map hash values to [0, 1] range, inf -> 0
        max_val = np.maximum(sk_a, sk_b)
        valid = (max_val < INF_HASH).astype(np.float32)

        # Feature 1: Element-wise match ratio (both hashed to same min = shared neighbor)
        match = ((sk_a == sk_b) & (sk_a < INF_HASH)).astype(np.float32)
        parts.append(match)

        # Feature 2: Valid indicator (both nodes have non-empty sketch at this position)
        both_valid = ((sk_a < INF_HASH) & (sk_b < INF_HASH)).astype(np.float32)
        parts.append(both_valid)

    pair_feats = np.concatenate(parts, axis=1)  # (B, n_hops * sketch_dim * 2)
    return torch.tensor(pair_feats, dtype=torch.float32, device=device)


class SketchProjector(nn.Module):
    """
    Learned projection of subgraph sketch features to a topological embedding.

    Takes raw pair-level sketch features and projects them through a small MLP
    with a sigmoid gate to allow the model to learn how much to rely on
    topological information.
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        # Learnable gate initialized near 0 (start by ignoring topo)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, sketch_feats):
        """
        Args:
            sketch_feats: (B, input_dim) raw sketch pair features
        Returns:
            topo_emb: (B, output_dim) gated topological embedding
        """
        projected = self.mlp(sketch_feats)
        gate_value = torch.sigmoid(self.gate)
        return gate_value * projected


# =============================================================================
# Neural Common Neighbor (NCN) Aggregator
# =============================================================================

# =============================================================================
# CF-NCN: Collaborative Filtering NCN (Scheme 4)
# =============================================================================

def build_morgan_knn(smiles_list, k=200, fp_bits=2048, fp_radius=2):
    """
    Build kNN graph based on Morgan fingerprint cosine similarity.

    For each molecule, finds the top-k most structurally similar molecules.
    Used by CF-NCN to identify "bridge molecules" — molecules similar to A
    that are known reaction partners of B.

    Args:
        smiles_list: list of SMILES strings
        k: number of nearest neighbors per molecule
        fp_bits: Morgan fingerprint length
        fp_radius: Morgan fingerprint radius

    Returns:
        knn_graph: dict[int, list[(int, float)]] — mol_idx -> [(neighbor_idx, similarity), ...]
                   sorted by similarity descending
    """
    from rdkit.Chem import rdFingerprintGenerator

    n = len(smiles_list)
    print(f"Building Morgan FP kNN graph (n={n}, k={k}, bits={fp_bits}, radius={fp_radius})...")
    t0 = time.time()

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=fp_bits, radius=fp_radius)
    features = np.zeros((n, fp_bits), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros(fp_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features[i] = arr

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features_norm = (features / norms).astype(np.float32)

    # Build kNN in batches
    knn_graph = {}
    batch_size = 500
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = features_norm[start:end] @ features_norm.T  # (batch, n)
        for i in range(start, end):
            local_i = i - start
            sims[local_i, i] = -2.0  # exclude self
            top_k_idx = np.argpartition(sims[local_i], -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(-sims[local_i, top_k_idx])]
            knn_graph[i] = [(int(j), float(sims[local_i, j])) for j in top_k_idx]

    elapsed = time.time() - t0
    print(f"  kNN graph built in {elapsed:.1f}s")
    return knn_graph


def _prepare_cf_cn_batch(idx_a, idx_b, knn_graph, adj, emb_table, max_bridge, device):
    """
    Prepare CF-NCN bridge molecule embeddings for a batch.

    For each pair (A, B):
      bridge_ab = kNN(A) ∩ Partners(B)  — similar to A AND can react with B
      bridge_ba = kNN(B) ∩ Partners(A)  — similar to B AND can react with A

    Args:
        idx_a: (B,) tensor of molecule A indices
        idx_b: (B,) tensor of molecule B indices
        knn_graph: dict[int, list[(int, float)]] — precomputed kNN
        adj: dict[int, set[int]] — reaction adjacency (train edges only)
        emb_table: (n_mols, dim) precomputed embedding table
        max_bridge: max bridge molecules per direction
        device: torch device

    Returns:
        bridge_embs: (B, max_bridge*2, dim)
        bridge_mask: (B, max_bridge*2) bool
        bridge_weights: (B, max_bridge*2) float — similarity weights
    """
    B = len(idx_a)
    dim = emb_table.shape[1]
    total_slots = max_bridge * 2  # max_bridge per direction
    bridge_embs = torch.zeros(B, total_slots, dim, device=device)
    bridge_mask = torch.zeros(B, total_slots, dtype=torch.bool, device=device)
    bridge_weights = torch.zeros(B, total_slots, device=device)

    for i in range(B):
        a = idx_a[i].item()
        b = idx_b[i].item()

        slot = 0

        # Direction 1: kNN(A) ∩ Partners(B)
        partners_b = adj.get(b, set())
        if partners_b:
            knn_a = knn_graph.get(a, [])
            for nb_idx, nb_sim in knn_a:
                if slot >= max_bridge:
                    break
                if nb_idx in partners_b and 0 <= nb_idx < emb_table.shape[0]:
                    bridge_embs[i, slot] = emb_table[nb_idx].to(device)
                    bridge_mask[i, slot] = True
                    bridge_weights[i, slot] = max(nb_sim, 1e-6)
                    slot += 1

        # Direction 2: kNN(B) ∩ Partners(A)
        slot = max_bridge  # start from second half
        partners_a = adj.get(a, set())
        if partners_a:
            knn_b = knn_graph.get(b, [])
            for nb_idx, nb_sim in knn_b:
                if slot >= total_slots:
                    break
                if nb_idx in partners_a and 0 <= nb_idx < emb_table.shape[0]:
                    bridge_embs[i, slot] = emb_table[nb_idx].to(device)
                    bridge_mask[i, slot] = True
                    bridge_weights[i, slot] = max(nb_sim, 1e-6)
                    slot += 1

    return bridge_embs, bridge_mask, bridge_weights


def precompute_cf_bridge_table(edges_list, knn_graph, adj, max_bridge):
    """
    Pre-compute bridge molecule indices/weights for ALL edges in the dataset.
    Done once (or when kNN refreshes), eliminating per-batch Python loops.

    Args:
        edges_list: list of (idx_a, idx_b) tuples — all edges in the dataset
        knn_graph: dict[int, list[(int, float)]]
        adj: dict[int, set[int]]
        max_bridge: max bridge molecules per direction

    Returns:
        bridge_idx_table: (N, max_bridge*2) LongTensor, -1 for empty slots
        bridge_wt_table: (N, max_bridge*2) FloatTensor
    """
    N = len(edges_list)
    total_slots = max_bridge * 2
    idx_table = torch.full((N, total_slots), -1, dtype=torch.long)
    wt_table = torch.zeros(N, total_slots)

    for ei in range(N):
        a, b = edges_list[ei]

        # Direction AB: kNN(A) ∩ Partners(B)
        slot = 0
        partners_b = adj.get(b, set())
        if partners_b:
            for nb_idx, nb_sim in knn_graph.get(a, []):
                if slot >= max_bridge:
                    break
                if nb_idx in partners_b:
                    idx_table[ei, slot] = nb_idx
                    wt_table[ei, slot] = max(nb_sim, 1e-6)
                    slot += 1

        # Direction BA: kNN(B) ∩ Partners(A)
        slot = max_bridge
        partners_a = adj.get(a, set())
        if partners_a:
            for nb_idx, nb_sim in knn_graph.get(b, []):
                if slot >= total_slots:
                    break
                if nb_idx in partners_a:
                    idx_table[ei, slot] = nb_idx
                    wt_table[ei, slot] = max(nb_sim, 1e-6)
                    slot += 1

    n_with_bridges = (idx_table >= 0).any(dim=1).sum().item()
    print(f"  Precomputed CF bridges for {N} edges: {n_with_bridges} have bridges")
    return idx_table, wt_table


def precompute_cn_table(edges_list, adj, max_cn):
    """
    Pre-compute common neighbor indices for ALL edges in the dataset.

    Returns:
        cn_idx_table: (N, max_cn) LongTensor, -1 for empty slots
    """
    N = len(edges_list)
    idx_table = torch.full((N, max_cn), -1, dtype=torch.long)

    for ei in range(N):
        a, b = edges_list[ei]
        cn = adj.get(a, set()) & adj.get(b, set())
        cn = list(cn)[:max_cn]
        for j, c in enumerate(cn):
            idx_table[ei, j] = c

    n_with_cn = (idx_table >= 0).any(dim=1).sum().item()
    print(f"  Precomputed CN for {N} edges: {n_with_cn} have common neighbors")
    return idx_table


def _prepare_cf_cn_batch_fast(edge_idx, cf_bridge_idx_table, cf_bridge_wt_table,
                               emb_table, device):
    """
    Vectorized CF-NCN batch preparation using precomputed tables.
    Replaces the slow per-item Python loop with tensor indexing.

    Args:
        edge_idx: (B,) tensor of dataset edge indices
        cf_bridge_idx_table: (N, total_slots) precomputed bridge mol indices
        cf_bridge_wt_table: (N, total_slots) precomputed similarity weights
        emb_table: (n_mols, dim) current embedding table
        device: torch device
    """
    batch_indices = cf_bridge_idx_table[edge_idx]   # (B, total_slots)
    batch_weights = cf_bridge_wt_table[edge_idx]    # (B, total_slots)
    mask = batch_indices >= 0                        # (B, total_slots)

    # Vectorized embedding gather
    flat_idx = batch_indices.clamp(min=0).reshape(-1)  # (B*total_slots,)
    flat_embs = emb_table[flat_idx]                    # (B*total_slots, dim)
    B = len(edge_idx)
    bridge_embs = flat_embs.reshape(B, -1, emb_table.shape[1])
    bridge_embs[~mask] = 0

    return bridge_embs.to(device), mask.to(device), batch_weights.to(device)


def _prepare_cn_batch_fast(edge_idx, cn_idx_table, emb_table, device):
    """
    Vectorized NCN batch preparation using precomputed tables.

    Args:
        edge_idx: (B,) tensor of dataset edge indices
        cn_idx_table: (N, max_cn) precomputed common neighbor indices
        emb_table: (n_mols, dim) current embedding table
        device: torch device
    """
    batch_indices = cn_idx_table[edge_idx]  # (B, max_cn)
    mask = batch_indices >= 0               # (B, max_cn)

    flat_idx = batch_indices.clamp(min=0).reshape(-1)
    flat_embs = emb_table[flat_idx]
    B = len(edge_idx)
    cn_embs = flat_embs.reshape(B, -1, emb_table.shape[1])
    cn_embs[~mask] = 0

    return cn_embs.to(device), mask.to(device)


class CFNCNAggregator(nn.Module):
    """
    Enhanced CF-NCN aggregator with:
    - Asymmetric aggregation: separate heads for AB (kNN(A)∩Partners(B))
      and BA (kNN(B)∩Partners(A)) directions
    - Query-bridge interaction: attention uses query-bridge difference & product
    - Bridge count: outputs normalized count as scalar feature
    """

    def __init__(self, emb_dim: int, asymmetric: bool = True,
                 query_interact: bool = True, use_count: bool = True):
        super().__init__()
        self.asymmetric = asymmetric
        self.query_interact = query_interact
        self.use_count = use_count
        self.emb_dim = emb_dim

        # Attention input dim depends on query_interact
        attn_input_dim = emb_dim * 3 if query_interact else emb_dim

        n_heads = 2 if asymmetric else 1
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(attn_input_dim, emb_dim // 4),
                nn.GELU(),
                nn.Linear(emb_dim // 4, 1),
            ) for _ in range(n_heads)
        ])
        self.no_bridge_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(emb_dim) * 0.02) for _ in range(n_heads)
        ])

        # Output projection if asymmetric (2*dim → dim)
        if asymmetric:
            self.out_proj = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.GELU(),
            )

    def _aggregate_one_direction(self, bridge_embs, bridge_mask, bridge_weights,
                                  query_emb, head_idx):
        """Aggregate one direction (AB or BA)."""
        has_bridge = bridge_mask.any(dim=1)

        if self.query_interact and query_emb is not None:
            q_exp = query_emb.unsqueeze(1).expand_as(bridge_embs)
            attn_input = torch.cat([bridge_embs, bridge_embs - q_exp,
                                     bridge_embs * q_exp], dim=-1)
        else:
            attn_input = bridge_embs

        attn_scores = self.attn_heads[head_idx](attn_input).squeeze(-1)
        sim_bias = bridge_weights.clamp(min=1e-6).log()
        attn_scores = attn_scores + sim_bias
        attn_scores = attn_scores.masked_fill(~bridge_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        agg = (attn_weights.unsqueeze(-1) * bridge_embs).sum(dim=1)
        agg[~has_bridge] = self.no_bridge_tokens[head_idx]
        return agg

    def forward(self, bridge_embs, bridge_mask, bridge_weights,
                max_bridge, query_emb_a=None, query_emb_b=None):
        """
        Args:
            bridge_embs: (B, max_bridge*2, dim) — first half AB, second half BA
            bridge_mask, bridge_weights: same shape
            max_bridge: split point between AB / BA
            query_emb_a, query_emb_b: (B, dim) embeddings of A and B

        Returns:
            agg: (B, dim), count: (B, 1) or None
        """
        embs_ab, mask_ab, w_ab = bridge_embs[:, :max_bridge], bridge_mask[:, :max_bridge], bridge_weights[:, :max_bridge]
        embs_ba, mask_ba, w_ba = bridge_embs[:, max_bridge:], bridge_mask[:, max_bridge:], bridge_weights[:, max_bridge:]

        if self.asymmetric:
            agg_ab = self._aggregate_one_direction(embs_ab, mask_ab, w_ab, query_emb_a, 0)
            agg_ba = self._aggregate_one_direction(embs_ba, mask_ba, w_ba, query_emb_b, 1)
            agg = self.out_proj(torch.cat([agg_ab, agg_ba], dim=-1))
        else:
            agg = self._aggregate_one_direction(
                bridge_embs, bridge_mask, bridge_weights, query_emb_a, 0)

        count = None
        if self.use_count:
            count = bridge_mask.sum(dim=1, keepdim=True).float() / (max_bridge * 2)
        return agg, count


class NCNAggregator(nn.Module):
    """
    Attention-weighted aggregation of common neighbor embeddings.

    For a pair (A, B), gathers embeddings of their common neighbors
    in the reaction graph and produces a single summary vector.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 4),
            nn.GELU(),
            nn.Linear(emb_dim // 4, 1),
        )
        # Learned token for pairs with no common neighbors
        self.no_cn_token = nn.Parameter(torch.randn(emb_dim) * 0.02)

    def forward(self, cn_embs: torch.Tensor, cn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cn_embs: (B, max_cn, dim) embeddings of common neighbors
            cn_mask: (B, max_cn) bool, True where valid CN exists

        Returns:
            cn_agg: (B, dim) aggregated common neighbor embedding
        """
        B = cn_embs.size(0)
        has_cn = cn_mask.any(dim=1)  # (B,)

        # Attention-weighted mean of common neighbor embeddings
        attn_scores = self.attention(cn_embs).squeeze(-1)  # (B, max_cn)
        attn_scores = attn_scores.masked_fill(~cn_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, max_cn)
        cn_agg = (attn_weights.unsqueeze(-1) * cn_embs).sum(dim=1)  # (B, dim)

        # For pairs with no common neighbors, use learned token
        cn_agg[~has_cn] = self.no_cn_token

        return cn_agg


def build_adjacency(edges):
    """Build adjacency list from edge list (undirected)."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def get_common_neighbors(adj, a, b, max_cn=32):
    """Get common neighbors of a and b from adjacency list."""
    cn = adj.get(a, set()) & adj.get(b, set())
    cn = list(cn)
    if len(cn) > max_cn:
        cn = cn[:max_cn]
    return cn


@torch.no_grad()
def precompute_embedding_table(model, dataset, device, batch_size=512):
    """
    Precompute molecular embeddings for all molecules in the dataset.

    Returns:
        emb_table: (n_mols, dim) tensor on device
    """
    model.eval()

    # Collect all unique molecule indices and their graphs
    unique_mols = sorted(dataset.graph_cache.keys())
    n_unique = len(unique_mols)
    if n_unique == 0:
        return None

    # Determine embedding dim from model
    emb_dim = model.config.mol_embedding_dim

    # We need max mol_idx + 1 rows
    max_idx = max(unique_mols) + 1
    emb_table = torch.zeros(max_idx, emb_dim, device=device)

    # Process in batches (using pre-tensorized graphs for speed)
    for start in range(0, n_unique, batch_size):
        end = min(start + batch_size, n_unique)
        batch_indices = unique_mols[start:end]

        # Prepare batched graphs via torch.cat (fast path for tensorized cache)
        atom_parts, edge_parts, bond_parts, batch_parts = [], [], [], []
        offset = 0

        for i, idx in enumerate(batch_indices):
            graph = dataset.graph_cache[idx]
            n = graph['n_atoms']
            af = graph['atom_features']
            ei = graph['edge_index']
            bf = graph['bond_features']
            # Handle both tensor and list formats
            if not isinstance(af, torch.Tensor):
                af = torch.tensor(af, dtype=torch.float)
                ei = torch.tensor(ei, dtype=torch.long)
                bf = torch.tensor(bf, dtype=torch.float)
            atom_parts.append(af)
            if ei.numel() > 0:
                edge_parts.append(ei + offset)
            bond_parts.append(bf)
            batch_parts.append(torch.full((n,), i, dtype=torch.long))
            offset += n

        if not atom_parts:
            continue

        atom_feats = torch.cat(atom_parts, dim=0).to(device)
        edges_cat = torch.cat(edge_parts, dim=0) if edge_parts else torch.zeros(0, 2, dtype=torch.long)
        edges = edges_cat.t().contiguous().to(device) if edges_cat.numel() > 0 else torch.zeros(2, 0, dtype=torch.long, device=device)
        bond_feats = torch.cat(bond_parts, dim=0).to(device)
        batch_idx = torch.cat(batch_parts, dim=0).to(device)

        # Encode (get link_emb, which is the projected embedding used for scoring)
        _, link_emb = model.encode_molecule(atom_feats, edges, bond_feats, batch_idx)

        for i, idx in enumerate(batch_indices):
            emb_table[idx] = link_emb[i]

    return emb_table


# =============================================================================
# Main Model
# =============================================================================

class HypergraphLinkPredictorV3(nn.Module):
    """
    Hypergraph-aware link predictor v3 with multi-task learning.

    Architecture:
      - 2D MPNN encoder with rich features
      - Optional frozen 3D encoder with gated fusion
      - Link projector
      - Pair scorer (link prediction head)
      - Reaction class predictor (auxiliary)
      - Property predictor (auxiliary: yield, barrier, rate)
    """

    def __init__(self, config: HypergraphLinkConfigV3):
        super().__init__()
        self.config = config

        # 2D Molecule encoder
        self.encoder = MoleculeEncoderV3(config)

        # Optional 3D fusion
        if config.use_3d:
            self.fusion = CrossModalFusion(
                config.mol_embedding_dim, config.frozen_3d_dim, config.mol_embedding_dim
            )
        else:
            self.fusion = None

        # Link prediction projector
        layers = []
        in_dim = config.mol_embedding_dim
        for i in range(config.num_link_layers):
            out_dim = config.hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, config.mol_embedding_dim))
        self.link_projector = nn.Sequential(*layers)

        # Optional NCN aggregator
        if config.use_ncn:
            self.ncn_aggregator = NCNAggregator(config.mol_embedding_dim)
            pair_input_dim = config.mol_embedding_dim * 4  # +cn_emb
        else:
            self.ncn_aggregator = None
            pair_input_dim = config.mol_embedding_dim * 3

        # Optional topological features (Scheme 1)
        if config.use_topo:
            pair_input_dim += config.topo_feature_dim

        # Optional subgraph sketch projector (Scheme 2)
        if config.use_sketch:
            sketch_input_dim = config.sketch_hops * config.sketch_dim * 2
            self.sketch_projector = SketchProjector(
                sketch_input_dim, config.sketch_mlp_hidden, config.sketch_out_dim
            )
            pair_input_dim += config.sketch_out_dim
        else:
            self.sketch_projector = None

        # Optional CF-NCN aggregator (Scheme 4)
        if config.use_cf_ncn:
            self.cf_ncn_aggregator = CFNCNAggregator(
                config.mol_embedding_dim,
                asymmetric=config.cf_ncn_asymmetric,
                query_interact=config.cf_ncn_query_interact,
                use_count=config.cf_ncn_use_count,
            )
            pair_input_dim += config.mol_embedding_dim
            if config.cf_ncn_use_count:
                pair_input_dim += 1  # scalar count feature
        else:
            self.cf_ncn_aggregator = None

        # Pairwise scoring head
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        # Auxiliary: Reaction class predictor
        self.rxn_classifier = nn.Sequential(
            nn.Linear(config.mol_embedding_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_rxn_classes)
        )

        # Auxiliary: Property predictor (yield, barrier, rate)
        self.property_predictor = nn.Sequential(
            nn.Linear(config.mol_embedding_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 3)
        )

    def encode_molecule(self, atom_features, edge_index, bond_features, batch_idx,
                        emb_3d=None):
        """Encode molecule to embedding."""
        mol_emb = self.encoder(atom_features, edge_index, bond_features, batch_idx)

        if self.fusion is not None and emb_3d is not None:
            mol_emb = self.fusion(mol_emb, emb_3d)

        link_emb = self.link_projector(mol_emb)
        return mol_emb, link_emb

    def compute_pair_score(self, emb_a, emb_b, cn_emb=None, topo_feats=None,
                           sketch_emb=None, cf_bridge_emb=None, cf_bridge_count=None):
        """Compute pairwise reaction score."""
        parts = [emb_a, emb_b, emb_a * emb_b]
        if self.ncn_aggregator is not None and cn_emb is not None:
            parts.append(cn_emb)
        if self.config.use_topo and topo_feats is not None:
            parts.append(topo_feats)
        if self.sketch_projector is not None and sketch_emb is not None:
            parts.append(sketch_emb)
        if self.cf_ncn_aggregator is not None and cf_bridge_emb is not None:
            parts.append(cf_bridge_emb)
            if cf_bridge_count is not None:
                parts.append(cf_bridge_count)
        combined = torch.cat(parts, dim=-1)
        return self.pair_scorer(combined).squeeze(-1)

    def predict_rxn_class(self, emb_a, emb_b):
        """Predict reaction class for a pair."""
        combined = torch.cat([emb_a, emb_b], dim=-1)
        return self.rxn_classifier(combined)

    def predict_properties(self, emb_a, emb_b):
        """Predict reaction properties (yield, barrier, rate)."""
        combined = torch.cat([emb_a, emb_b], dim=-1)
        return self.property_predictor(combined)

    def forward(self, mol_a_feats, mol_a_edges, mol_a_bond_feats, mol_a_batch,
                mol_b_feats, mol_b_edges, mol_b_bond_feats, mol_b_batch,
                emb_3d_a=None, emb_3d_b=None,
                cn_embs=None, cn_mask=None,
                topo_feats=None, sketch_feats=None,
                cf_bridge_embs=None, cf_bridge_mask=None, cf_bridge_weights=None):
        """
        Forward pass for training.

        Args:
            cn_embs: (B, max_cn, dim) common neighbor embeddings (NCN)
            cn_mask: (B, max_cn) bool mask for valid common neighbors
            topo_feats: (B, topo_dim) topological heuristic features (Scheme 1)
            sketch_feats: (B, sketch_input_dim) raw sketch pair features (Scheme 2)
            cf_bridge_embs: (B, max_bridge*2, dim) CF-NCN bridge embeddings (Scheme 4)
            cf_bridge_mask: (B, max_bridge*2) bool mask for valid bridges
            cf_bridge_weights: (B, max_bridge*2) similarity weights for bridges

        Returns dict with:
          - link_emb_a, link_emb_b: link embeddings
          - score: link prediction logit
          - rxn_logits: reaction class logits (if applicable)
          - property_pred: predicted (yield, barrier, rate)
        """
        # 3D dropout during training
        if self.training and self.config.use_3d and emb_3d_a is not None:
            if torch.rand(1).item() < self.config.dropout_3d:
                emb_3d_a = None
                emb_3d_b = None

        _, link_emb_a = self.encode_molecule(
            mol_a_feats, mol_a_edges, mol_a_bond_feats, mol_a_batch, emb_3d_a
        )
        _, link_emb_b = self.encode_molecule(
            mol_b_feats, mol_b_edges, mol_b_bond_feats, mol_b_batch, emb_3d_b
        )

        # NCN: aggregate common neighbor embeddings
        cn_agg = None
        if self.ncn_aggregator is not None and cn_embs is not None and cn_mask is not None:
            cn_agg = self.ncn_aggregator(cn_embs, cn_mask)

        # Sketch: project raw sketch features through learned MLP
        sketch_emb = None
        if self.sketch_projector is not None and sketch_feats is not None:
            sketch_emb = self.sketch_projector(sketch_feats)

        # CF-NCN: aggregate bridge molecule embeddings
        cf_bridge_emb = None
        cf_bridge_count = None
        if self.cf_ncn_aggregator is not None and cf_bridge_embs is not None:
            cf_bridge_emb, cf_bridge_count = self.cf_ncn_aggregator(
                cf_bridge_embs, cf_bridge_mask, cf_bridge_weights,
                max_bridge=self.config.cf_ncn_max_bridge,
                query_emb_a=link_emb_a, query_emb_b=link_emb_b)

        score = self.compute_pair_score(link_emb_a, link_emb_b, cn_agg, topo_feats,
                                        sketch_emb, cf_bridge_emb, cf_bridge_count)
        rxn_logits = self.predict_rxn_class(link_emb_a, link_emb_b)
        property_pred = self.predict_properties(link_emb_a, link_emb_b)

        return {
            'link_emb_a': link_emb_a,
            'link_emb_b': link_emb_b,
            'score': score,
            'rxn_logits': rxn_logits,
            'property_pred': property_pred,
        }


# =============================================================================
# Multi-Edge Database Extension
# =============================================================================

def build_multi_edge_info(db: ReactionDatabase):
    """
    Build multi-edge reaction graph information from ReactionDatabase.

    Returns:
        edge_rxn_map: Dict[(mol_a, mol_b)] -> list of rxn_ids
        edge_metadata: Dict[(mol_a, mol_b)] -> {
            'n_reactions': int,
            'rxn_classes': set,
            'avg_yield': float,
            'avg_barrier': float,
            'avg_rate': float,
        }
    """
    edge_rxn_map = defaultdict(list)
    edge_metadata = {}

    for rxn_id, rxn in enumerate(db.reactions):
        indices = rxn['reactant_indices']
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = min(indices[i], indices[j]), max(indices[i], indices[j])
                edge_rxn_map[(a, b)].append(rxn_id)

    # Build metadata
    for edge, rxn_ids in edge_rxn_map.items():
        classes = set()
        yields, barriers, rates = [], [], []
        for rxn_id in rxn_ids:
            rxn = db.reactions[rxn_id]
            classes.add(rxn.get('rxn_class', 0))

        edge_metadata[edge] = {
            'n_reactions': len(rxn_ids),
            'rxn_classes': classes,
            'rxn_ids': rxn_ids,
        }

    return edge_rxn_map, edge_metadata


def compute_soft_labels(edge_metadata: dict, normalization_k: int = 3):
    """
    Compute soft labels for positive edges based on number of reactions.

    Args:
        edge_metadata: from build_multi_edge_info
        normalization_k: number of reactions that maps to label=1.0

    Returns:
        Dict[(a,b)] -> float in [0.5, 1.0]
    """
    soft_labels = {}
    for edge, meta in edge_metadata.items():
        n = meta['n_reactions']
        soft_labels[edge] = min(1.0, 0.5 + 0.5 * n / normalization_k)
    return soft_labels


# =============================================================================
# RCNS (Reaction Center-Aware Negative Sampling)
# =============================================================================

def get_functional_groups(smiles: str):
    """Get functional group fingerprint bits for RCNS."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=256)
    return set(fp.GetOnBits())


_rcns_shared = {}  # Module-level shared state for worker processes (avoids pickling large data)


def _rcns_init(fg_bits_arr, fg_index_lists, all_pos_set):
    """Initializer for RCNS worker pool — stores shared data in global dict."""
    _rcns_shared['fg_bits'] = fg_bits_arr
    _rcns_shared['fg_index'] = fg_index_lists
    _rcns_shared['all_pos'] = all_pos_set


def _rcns_worker(args):
    """Worker function for parallel RCNS hard negative sampling."""
    chunk, seed = args
    fg_bits_arr = _rcns_shared['fg_bits']
    fg_index_lists = _rcns_shared['fg_index']
    all_pos_set = _rcns_shared['all_pos']
    rng = np.random.RandomState(seed)
    results = []
    for a, b in chunk:
        b_bits = fg_bits_arr[b]
        if len(b_bits) == 0:
            continue
        bit = b_bits[rng.randint(len(b_bits))]
        candidates = fg_index_lists.get(bit, [])
        if len(candidates) < 2:
            continue
        n_sample = min(20, len(candidates))
        sampled_idx = rng.choice(len(candidates), n_sample, replace=False)
        for idx in sampled_idx:
            c = candidates[idx]
            if c == b or c == a:
                continue
            edge = (min(a, c), max(a, c))
            if edge not in all_pos_set:
                results.append(edge)
                break
    return results


def rcns_negative_sampling(db: ReactionDatabase, pos_edges, n_negatives, all_pos=None, seed=42, n_workers=None):
    """
    Reaction Center-Aware Negative Sampling (parallelized).

    For each positive edge (A, B), generate a hard negative by replacing B
    with a molecule B' that shares functional groups with B but is NOT a known
    reaction partner of A.

    Mix: ~50% hard RCNS negatives + ~50% random negatives.
    Uses multiprocessing for the hard negative sampling loop.
    Shared data is passed via fork COW (Pool initializer) to avoid pickling.
    """
    import multiprocessing as mp

    rng = np.random.RandomState(seed)
    n_mols = len(db.reactant_smiles)

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 64)

    # Use forkserver if CUDA is already initialized (avoids corrupting child CUDA context).
    # Otherwise use fork (default) for fast COW memory sharing of large data structures.
    if torch.cuda.is_initialized():
        ctx = mp.get_context('forkserver')
    else:
        ctx = mp.get_context('fork')

    # Phase 1: Compute functional group FPs (parallel with Pool)
    t0 = time.time()
    with ctx.Pool(n_workers) as pool:
        fg_results = pool.map(get_functional_groups, db.reactant_smiles, chunksize=2048)
    # Convert to list-of-lists (list indexing faster than dict, and avoids pickling sets)
    fg_bits_arr = [list(bits) for bits in fg_results]
    print(f"  Computed functional group FPs for {n_mols} molecules in {time.time()-t0:.1f}s ({n_workers} workers)")

    # Build inverted index: bit -> list of mol indices
    fg_index = defaultdict(list)
    for i, bits in enumerate(fg_bits_arr):
        for b in bits:
            fg_index[b].append(i)
    fg_index_lists = dict(fg_index)

    if all_pos is None:
        all_pos = set()
        for a, b in pos_edges:
            all_pos.add((min(a, b), max(a, b)))

    # Phase 2: Hard negative sampling (parallel via Pool + initializer)
    n_hard = int(n_negatives * 0.5)
    pos_list = list(pos_edges)
    rng.shuffle(pos_list)

    # Oversample: each worker processes a chunk, we deduplicate after
    oversample_n = min(len(pos_list), n_hard * 3)
    pos_subset = pos_list[:oversample_n]

    t1 = time.time()
    # Split into exactly n_workers chunks
    chunk_size = max(1, (len(pos_subset) + n_workers - 1) // n_workers)
    chunks = []
    for chunk_idx in range(n_workers):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(pos_subset))
        if start >= len(pos_subset):
            break
        chunks.append((pos_subset[start:end], seed + chunk_idx))

    # Use initializer to share large data — avoids pickling
    with ctx.Pool(n_workers, initializer=_rcns_init,
                  initargs=(fg_bits_arr, fg_index_lists, all_pos)) as pool:
        chunk_results = pool.map(_rcns_worker, chunks)

    # Merge and deduplicate
    seen = set()
    hard_negatives = []
    for chunk_res in chunk_results:
        for edge in chunk_res:
            if edge not in seen and len(hard_negatives) < n_hard:
                hard_negatives.append(edge)
                seen.add(edge)
    print(f"  RCNS hard negatives: {len(hard_negatives)}/{n_hard} in {time.time()-t1:.1f}s ({n_workers} workers)")

    # Phase 3: Random negatives for the rest
    random_negatives = []
    n_random = n_negatives - len(hard_negatives)
    attempts = 0
    while len(random_negatives) < n_random and attempts < n_random * 20:
        a, b = rng.randint(0, n_mols), rng.randint(0, n_mols)
        if a == b:
            attempts += 1
            continue
        edge = (min(a, b), max(a, b))
        if edge not in all_pos and edge not in seen:
            random_negatives.append(edge)
            seen.add(edge)
        attempts += 1

    all_neg = hard_negatives + random_negatives
    rng.shuffle(all_neg)
    print(f"  RCNS: {len(hard_negatives)} hard + {len(random_negatives)} random = {len(all_neg)} negatives")
    return all_neg


# =============================================================================
# Dataset
# =============================================================================

class ReactionLinkDatasetV3(Dataset):
    """Dataset with rich graph features, molecule indices, and metadata."""

    def __init__(self, pos_edges, neg_edges, smiles_list,
                 rxn_center_scores=None,
                 soft_labels=None,
                 edge_rxn_classes=None,
                 precompute=True,
                 shared_graph_cache=None):
        """
        Args:
            pos_edges: list of (idx_a, idx_b)
            neg_edges: list of (idx_a, idx_b)
            smiles_list: list of SMILES strings
            rxn_center_scores: Dict[mol_idx] -> np.ndarray per-atom scores
            soft_labels: Dict[(a,b)] -> float soft label for positive edges
            edge_rxn_classes: Dict[(a,b)] -> set of rxn_class ints
            precompute: whether to precompute graphs
            shared_graph_cache: optional pre-built graph cache dict to share across datasets
        """
        self.edges = []
        for a, b in pos_edges:
            key = (min(a, b), max(a, b))
            label = soft_labels.get(key, 1.0) if soft_labels else 1.0
            rxn_cls = edge_rxn_classes.get(key, {0}) if edge_rxn_classes else {0}
            # Use the most common rxn_class (first one)
            rxn_cls_label = min(rxn_cls) if rxn_cls else 0
            self.edges.append((a, b, label, rxn_cls_label))

        for a, b in neg_edges:
            self.edges.append((a, b, 0.0, -1))  # -1 = no rxn class for negatives

        self.smiles_list = smiles_list
        self.rxn_center_scores = rxn_center_scores or {}

        # Use shared graph cache if provided, otherwise build our own
        if shared_graph_cache is not None:
            self.graph_cache = shared_graph_cache
        else:
            self.graph_cache = {}
            if precompute:
                self._build_graph_cache()

    def _build_graph_cache(self):
        """Build graph cache for all unique molecules in this dataset."""
        unique_mols = set()
        for a, b, _, _ in self.edges:
            unique_mols.add(a)
            unique_mols.add(b)

        print(f"Precomputing rich graphs for {len(unique_mols)} molecules...")
        for idx in tqdm(unique_mols, desc="Graphs"):
            if idx < len(self.smiles_list):
                rc = self.rxn_center_scores.get(idx, None)
                graph = smiles_to_rich_graph(self.smiles_list[idx], rc)
                if graph is not None:
                    self.graph_cache[idx] = {
                        'atom_features': torch.tensor(graph['atom_features'], dtype=torch.float),
                        'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long),
                        'bond_features': torch.tensor(graph['bond_features'], dtype=torch.float),
                        'n_atoms': graph['n_atoms'],
                    }
        print(f"Cached {len(self.graph_cache)} valid graphs (tensorized)")

    @staticmethod
    def build_shared_graph_cache(smiles_list, rxn_center_scores=None):
        """Build a graph cache for ALL molecules, to be shared across datasets."""
        rxn_center_scores = rxn_center_scores or {}
        graph_cache = {}
        print(f"Precomputing shared graph cache for {len(smiles_list)} molecules...")
        for idx in tqdm(range(len(smiles_list)), desc="Graphs"):
            rc = rxn_center_scores.get(idx, None)
            graph = smiles_to_rich_graph(smiles_list[idx], rc)
            if graph is not None:
                graph_cache[idx] = {
                    'atom_features': torch.tensor(graph['atom_features'], dtype=torch.float),
                    'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long),
                    'bond_features': torch.tensor(graph['bond_features'], dtype=torch.float),
                    'n_atoms': graph['n_atoms'],
                }
        print(f"Shared cache: {len(graph_cache)} valid graphs (tensorized)")
        return graph_cache

    def __len__(self):
        return len(self.edges)

    _fallback_graph = None  # lazily created

    def _get_graph(self, idx):
        if idx in self.graph_cache:
            return self.graph_cache[idx]
        if idx < len(self.smiles_list):
            rc = self.rxn_center_scores.get(idx, None)
            graph = smiles_to_rich_graph(self.smiles_list[idx], rc)
            if graph is not None:
                return {
                    'atom_features': torch.tensor(graph['atom_features'], dtype=torch.float),
                    'edge_index': torch.tensor(graph['edge_index'], dtype=torch.long),
                    'bond_features': torch.tensor(graph['bond_features'], dtype=torch.float),
                    'n_atoms': graph['n_atoms'],
                }
        # Fallback: single carbon atom (tensor)
        if ReactionLinkDatasetV3._fallback_graph is None:
            ReactionLinkDatasetV3._fallback_graph = {
                'atom_features': torch.zeros(1, ATOM_FEATURE_DIM),
                'edge_index': torch.zeros(1, 2, dtype=torch.long),
                'bond_features': torch.zeros(1, BOND_FEATURE_DIM),
                'n_atoms': 1,
            }
        return ReactionLinkDatasetV3._fallback_graph

    def __getitem__(self, i):
        a, b, label, rxn_cls = self.edges[i]
        return {
            'graph_a': self._get_graph(a),
            'graph_b': self._get_graph(b),
            'label': label,
            'rxn_class': rxn_cls,
            'idx_a': a,
            'idx_b': b,
            'edge_idx': i,
        }


def collate_fn_v3(batch):
    """Collate batch of molecule pairs with rich features.

    Optimized: expects pre-tensorized graphs from ReactionLinkDatasetV3.
    Uses torch.cat instead of torch.tensor(nested_list) for ~5x faster collation.
    """
    def prepare_graphs(graph_list):
        atom_parts = []
        edge_parts = []
        bond_parts = []
        batch_parts = []
        offset = 0

        for i, graph in enumerate(graph_list):
            n = graph['n_atoms']
            atom_parts.append(graph['atom_features'])       # already a tensor
            edges = graph['edge_index']                      # (n_edges, 2) tensor
            if edges.numel() > 0:
                edge_parts.append(edges + offset)
            bond_parts.append(graph['bond_features'])        # already a tensor
            batch_parts.append(torch.full((n,), i, dtype=torch.long))
            offset += n

        all_edges = torch.cat(edge_parts, dim=0) if edge_parts else torch.zeros(0, 2, dtype=torch.long)

        return {
            'atom_features': torch.cat(atom_parts, dim=0),
            'edges': all_edges.t().contiguous() if all_edges.numel() > 0 else torch.zeros(2, 0, dtype=torch.long),
            'bond_features': torch.cat(bond_parts, dim=0),
            'batch': torch.cat(batch_parts, dim=0),
        }

    graphs_a = [item['graph_a'] for item in batch]
    graphs_b = [item['graph_b'] for item in batch]
    labels = [item['label'] for item in batch]
    rxn_classes = [item['rxn_class'] for item in batch]
    idx_a = [item['idx_a'] for item in batch]
    idx_b = [item['idx_b'] for item in batch]
    edge_idx = [item['edge_idx'] for item in batch]

    return {
        'mol_a': prepare_graphs(graphs_a),
        'mol_b': prepare_graphs(graphs_b),
        'labels': torch.tensor(labels, dtype=torch.float),
        'rxn_classes': torch.tensor(rxn_classes, dtype=torch.long),
        'idx_a': torch.tensor(idx_a, dtype=torch.long),
        'idx_b': torch.tensor(idx_b, dtype=torch.long),
        'edge_idx': torch.tensor(edge_idx, dtype=torch.long),
    }


# =============================================================================
# Training
# =============================================================================

def compute_loss(outputs, labels, rxn_classes, config):
    """
    Compute multi-task loss.

    Returns:
        total_loss, loss_dict
    """
    # Primary: BCE loss on link prediction (with label smoothing)
    if config.label_smoothing > 0:
        smooth_labels = labels * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
    else:
        smooth_labels = labels

    bce_loss = F.binary_cross_entropy_with_logits(outputs['score'], smooth_labels)

    # Contrastive loss
    emb_a = F.normalize(outputs['link_emb_a'], dim=-1)
    emb_b = F.normalize(outputs['link_emb_b'], dim=-1)
    sim = torch.sum(emb_a * emb_b, dim=-1) / config.temperature
    contrastive_loss = F.binary_cross_entropy_with_logits(sim, (labels > 0.5).float())

    # Auxiliary: Reaction class prediction (only for positive edges)
    rxn_class_loss = torch.tensor(0.0, device=labels.device)
    pos_mask = rxn_classes >= 0
    if pos_mask.any():
        valid_logits = outputs['rxn_logits'][pos_mask]
        valid_targets = rxn_classes[pos_mask].clamp(0, config.num_rxn_classes - 1)
        rxn_class_loss = F.cross_entropy(valid_logits, valid_targets)

    # Total loss
    total_loss = (
        bce_loss
        + 0.5 * contrastive_loss
        + config.rxn_class_weight * rxn_class_loss
    )

    return total_loss, {
        'bce': bce_loss.item(),
        'contrastive': contrastive_loss.item(),
        'rxn_class': rxn_class_loss.item(),
        'total': total_loss.item(),
    }


def _lookup_topo_feats(topo_dict, idx_a, idx_b, topo_dim, device):
    """Look up precomputed topological features for a batch of pairs."""
    if topo_dict is None:
        return None
    idx_a_np = idx_a.numpy() if isinstance(idx_a, torch.Tensor) else np.array(idx_a)
    idx_b_np = idx_b.numpy() if isinstance(idx_b, torch.Tensor) else np.array(idx_b)
    B = len(idx_a_np)
    feats = np.zeros((B, topo_dim), dtype=np.float32)
    for i in range(B):
        a, b = int(idx_a_np[i]), int(idx_b_np[i])
        key = (min(a, b), max(a, b))
        if key in topo_dict:
            feats[i] = topo_dict[key]
    return torch.tensor(feats, dtype=torch.float, device=device)


def _prepare_cn_batch(idx_a, idx_b, adj, emb_table, max_cn, device):
    """
    Prepare common neighbor embeddings for a batch.

    Args:
        idx_a: (B,) tensor of molecule A indices
        idx_b: (B,) tensor of molecule B indices
        adj: dict adjacency list
        emb_table: (n_mols, dim) precomputed embedding table
        max_cn: max common neighbors to use
        device: torch device

    Returns:
        cn_embs: (B, max_cn, dim) padded common neighbor embeddings
        cn_mask: (B, max_cn) bool mask
    """
    B = len(idx_a)
    dim = emb_table.shape[1]
    cn_embs = torch.zeros(B, max_cn, dim, device=device)
    cn_mask = torch.zeros(B, max_cn, dtype=torch.bool, device=device)

    for i in range(B):
        a = idx_a[i].item()
        b = idx_b[i].item()
        cn_list = get_common_neighbors(adj, a, b, max_cn)
        n_cn = len(cn_list)
        if n_cn > 0:
            cn_indices = torch.tensor(cn_list, dtype=torch.long, device=emb_table.device)
            valid = (cn_indices >= 0) & (cn_indices < emb_table.shape[0])
            cn_indices = cn_indices[valid]
            n_valid = len(cn_indices)
            if n_valid > 0:
                cn_embs[i, :n_valid] = emb_table[cn_indices].to(device)
                cn_mask[i, :n_valid] = True

    return cn_embs, cn_mask


def _lookup_3d_emb(emb_3d_table, indices, device):
    """Look up precomputed 3D embeddings by molecule index."""
    if emb_3d_table is None:
        return None
    indices = indices.to(emb_3d_table.device)
    valid_mask = (indices >= 0) & (indices < emb_3d_table.shape[0])
    emb = torch.zeros(len(indices), emb_3d_table.shape[1], device=device)
    if valid_mask.any():
        emb[valid_mask] = emb_3d_table[indices[valid_mask]]
    return emb


def train_epoch(model, loader, optimizer, device, config, scheduler=None,
                emb_3d_table=None,
                ncn_emb_table=None, cn_idx_table=None,
                topo_dict=None, sketch_data=None,
                cf_bridge_idx_table=None, cf_bridge_wt_table=None, cf_emb_table=None):
    """Train for one epoch (optimized with precomputed lookup tables)."""
    model.train()
    loss_accum = defaultdict(float)
    n_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        for key in ['mol_a', 'mol_b']:
            for k in batch[key]:
                batch[key][k] = batch[key][k].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        rxn_classes = batch['rxn_classes'].to(device)
        edge_idx = batch['edge_idx']

        # Look up 3D embeddings by molecule index
        emb_3d_a = _lookup_3d_emb(emb_3d_table, batch['idx_a'], device)
        emb_3d_b = _lookup_3d_emb(emb_3d_table, batch['idx_b'], device)

        # NCN: vectorized common neighbor lookup
        cn_embs, cn_mask = None, None
        if cn_idx_table is not None and ncn_emb_table is not None:
            cn_embs, cn_mask = _prepare_cn_batch_fast(
                edge_idx, cn_idx_table, ncn_emb_table, device
            )

        # Topo: look up precomputed topological features
        topo_feats = _lookup_topo_feats(
            topo_dict, batch['idx_a'], batch['idx_b'],
            config.topo_feature_dim, device
        )

        # Sketch: compute pair features from precomputed sketches
        sketch_feats = None
        if sketch_data is not None:
            sketch_feats = sketches_to_pair_features(
                sketch_data, batch['idx_a'], batch['idx_b'], device
            )

        # CF-NCN: vectorized bridge molecule lookup
        cf_bridge_embs, cf_bridge_mask, cf_bridge_weights = None, None, None
        if cf_bridge_idx_table is not None and cf_emb_table is not None:
            cf_bridge_embs, cf_bridge_mask, cf_bridge_weights = _prepare_cf_cn_batch_fast(
                edge_idx, cf_bridge_idx_table, cf_bridge_wt_table,
                cf_emb_table, device
            )

        optimizer.zero_grad()

        outputs = model(
            batch['mol_a']['atom_features'], batch['mol_a']['edges'],
            batch['mol_a']['bond_features'], batch['mol_a']['batch'],
            batch['mol_b']['atom_features'], batch['mol_b']['edges'],
            batch['mol_b']['bond_features'], batch['mol_b']['batch'],
            emb_3d_a=emb_3d_a, emb_3d_b=emb_3d_b,
            cn_embs=cn_embs, cn_mask=cn_mask,
            topo_feats=topo_feats,
            sketch_feats=sketch_feats,
            cf_bridge_embs=cf_bridge_embs, cf_bridge_mask=cf_bridge_mask,
            cf_bridge_weights=cf_bridge_weights,
        )

        total_loss, loss_dict = compute_loss(outputs, labels, rxn_classes, config)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k, v in loss_dict.items():
            loss_accum[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}


@torch.no_grad()
def evaluate(model, loader, device, emb_3d_table=None,
             ncn_emb_table=None, cn_idx_table=None,
             topo_dict=None, sketch_data=None,
             cf_bridge_idx_table=None, cf_bridge_wt_table=None, cf_emb_table=None):
    """Evaluate model on link prediction (optimized with precomputed tables)."""
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0
    n_batches = 0

    for batch in loader:
        for key in ['mol_a', 'mol_b']:
            for k in batch[key]:
                batch[key][k] = batch[key][k].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        edge_idx = batch['edge_idx']

        # Look up 3D embeddings by molecule index
        emb_3d_a = _lookup_3d_emb(emb_3d_table, batch['idx_a'], device)
        emb_3d_b = _lookup_3d_emb(emb_3d_table, batch['idx_b'], device)

        # NCN: vectorized common neighbor lookup
        cn_embs, cn_mask = None, None
        if cn_idx_table is not None and ncn_emb_table is not None:
            cn_embs, cn_mask = _prepare_cn_batch_fast(
                edge_idx, cn_idx_table, ncn_emb_table, device
            )

        # Topo: look up precomputed topological features
        topo_feats = None
        if topo_dict is not None:
            raw_model = model.module if hasattr(model, 'module') else model
            topo_feats = _lookup_topo_feats(
                topo_dict, batch['idx_a'], batch['idx_b'],
                raw_model.config.topo_feature_dim, device
            )

        # Sketch: compute pair features from precomputed sketches
        sketch_feats = None
        if sketch_data is not None:
            sketch_feats = sketches_to_pair_features(
                sketch_data, batch['idx_a'], batch['idx_b'], device
            )

        # CF-NCN: vectorized bridge molecule lookup
        cf_bridge_embs, cf_bridge_mask, cf_bridge_weights = None, None, None
        if cf_bridge_idx_table is not None and cf_emb_table is not None:
            cf_bridge_embs, cf_bridge_mask, cf_bridge_weights = _prepare_cf_cn_batch_fast(
                edge_idx, cf_bridge_idx_table, cf_bridge_wt_table,
                cf_emb_table, device
            )

        outputs = model(
            batch['mol_a']['atom_features'], batch['mol_a']['edges'],
            batch['mol_a']['bond_features'], batch['mol_a']['batch'],
            batch['mol_b']['atom_features'], batch['mol_b']['edges'],
            batch['mol_b']['bond_features'], batch['mol_b']['batch'],
            emb_3d_a=emb_3d_a, emb_3d_b=emb_3d_b,
            cn_embs=cn_embs, cn_mask=cn_mask,
            topo_feats=topo_feats,
            sketch_feats=sketch_feats,
            cf_bridge_embs=cf_bridge_embs, cf_bridge_mask=cf_bridge_mask,
            cf_bridge_weights=cf_bridge_weights,
        )

        loss = F.binary_cross_entropy_with_logits(outputs['score'], (labels > 0.5).float())
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(outputs['score']).cpu().numpy()
        y_true.extend((labels > 0.5).float().cpu().numpy().tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    return {
        'loss': total_loss / max(n_batches, 1),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'ap': average_precision_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, (y_prob >= 0.5).astype(int)),
        'f1': f1_score(y_true, (y_prob >= 0.5).astype(int)),
    }


# =============================================================================
# Main training pipeline
# =============================================================================

def train_v3(
    data_dir="Data/uspto",
    n_epochs=40,
    batch_size=128,
    lr=3e-4,
    model_size='medium',
    device='auto',
    num_workers=0,
    save_dir='model_reactions/checkpoints',
    pretrained_encoder=None,
    use_3d=False,
    conformer_cache=None,
    use_ncn=False,
    ncn_update_freq=5,
    use_topo=False,
    use_sketch=False,
    sketch_dim=128,
    use_cf_ncn=False,
    cf_ncn_k=200,
    cf_ncn_update_freq=5,
    knn_cache=None,
    early_stop_patience=0,
    ddp=False,
    local_rank=0,
):
    """Train the hypergraph link predictor v3."""

    # DDP setup
    rank = 0
    world_size = 1
    if ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = local_rank
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
    elif device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    is_main = (rank == 0)

    def log(msg):
        if is_main:
            print(msg)

    log(f"Device: {device}" + (f" (DDP: rank {rank}/{world_size})" if ddp else ""))

    # Build reaction database
    cache_path = os.path.join(data_dir, 'reaction_db_cache.pkl')
    db = ReactionDatabase(data_dir)
    if os.path.exists(cache_path):
        log("Loading cached reaction database...")
        db.load(cache_path)
    else:
        db.build()
        db.save(cache_path)

    n_mols = len(db.reactant_smiles)

    # Phase 1b: Extract reaction centers
    log("\n--- Phase 1b: Extracting reaction centers ---")
    rxn_center_scores = extract_reaction_centers(db)
    # Fall back to heuristic if atom mapping not available
    if len(rxn_center_scores) < len(db.reactant_smiles) * 0.1:
        log("  Atom mapping sparse, using heuristic reaction center scores...")
        for idx, smi in enumerate(db.reactant_smiles):
            if idx not in rxn_center_scores:
                scores = heuristic_reaction_center_scores(smi)
                if scores is not None:
                    rxn_center_scores[idx] = scores
        log(f"  Total molecules with reaction center scores: {len(rxn_center_scores)}")

    # Phase 1c: Build multi-edge info
    log("\n--- Phase 1c: Building multi-edge reaction graph ---")
    edge_rxn_map, edge_metadata = build_multi_edge_info(db)
    soft_labels = compute_soft_labels(edge_metadata)
    edge_rxn_classes = {k: v['rxn_classes'] for k, v in edge_metadata.items()}
    n_multi = sum(1 for v in edge_metadata.values() if v['n_reactions'] > 1)
    log(f"  {len(edge_metadata)} unique edges, {n_multi} with multiple reactions")

    # Get edges by split
    train_edges, val_edges, test_edges = [], [], []
    for rxn in db.reactions:
        indices = rxn['reactant_indices']
        split = rxn['split']
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                edge = (min(a, b), max(a, b))
                if split == 'test':
                    test_edges.append(edge)
                elif split == 'val':
                    val_edges.append(edge)
                else:
                    train_edges.append(edge)

    train_edges = list(set(train_edges))
    val_edges = list(set(val_edges))
    test_edges = list(set(test_edges))
    log(f"Edges: train={len(train_edges)}, val={len(val_edges)}, test={len(test_edges)}")

    # Phase 1d: RCNS negative sampling
    log("\n--- Phase 1d: RCNS Negative Sampling ---")
    all_pos = set(train_edges + val_edges + test_edges)
    train_neg = rcns_negative_sampling(db, train_edges, len(train_edges), all_pos, seed=42)

    def sample_random_negatives(n, seed):
        rng = np.random.RandomState(seed)
        negs, seen = [], set()
        attempts = 0
        while len(negs) < n and attempts < n * 20:
            a, b = rng.randint(0, n_mols), rng.randint(0, n_mols)
            if a == b:
                attempts += 1
                continue
            edge = (min(a, b), max(a, b))
            if edge not in all_pos and edge not in seen:
                negs.append(edge)
                seen.add(edge)
            attempts += 1
        return negs

    val_neg = sample_random_negatives(len(val_edges), seed=43)
    test_neg = sample_random_negatives(len(test_edges), seed=44)

    # Build shared graph cache once for all datasets (saves ~2x memory on FULL)
    log("\nBuilding shared graph cache...")
    shared_cache = ReactionLinkDatasetV3.build_shared_graph_cache(
        db.reactant_smiles, rxn_center_scores)

    # Create datasets (sharing the same graph cache)
    log("Creating datasets with rich features...")
    train_dataset = ReactionLinkDatasetV3(
        train_edges, train_neg, db.reactant_smiles,
        rxn_center_scores=rxn_center_scores,
        soft_labels=soft_labels,
        edge_rxn_classes=edge_rxn_classes,
        shared_graph_cache=shared_cache,
    )
    val_dataset = ReactionLinkDatasetV3(
        val_edges, val_neg, db.reactant_smiles,
        rxn_center_scores=rxn_center_scores,
        shared_graph_cache=shared_cache,
    )
    test_dataset = ReactionLinkDatasetV3(
        test_edges, test_neg, db.reactant_smiles,
        rxn_center_scores=rxn_center_scores,
        shared_graph_cache=shared_cache,
    )

    # Use multiple workers + prefetch for overlapping data prep with GPU compute
    effective_workers = max(num_workers, 8)
    use_pin = (device != 'cpu')

    # DDP: use DistributedSampler for train data
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=effective_workers, collate_fn=collate_fn_v3,
                              persistent_workers=True, pin_memory=use_pin,
                              prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=effective_workers, collate_fn=collate_fn_v3,
                            persistent_workers=True, pin_memory=use_pin,
                            prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=effective_workers, collate_fn=collate_fn_v3,
                             persistent_workers=True, pin_memory=use_pin,
                             prefetch_factor=4)

    # Load precomputed 3D embeddings if available
    emb_3d_table = None
    if use_3d:
        if conformer_cache and os.path.exists(conformer_cache):
            # Check if this is an embedding file (tensor) or conformer cache (dict)
            loaded = torch.load(conformer_cache, map_location='cpu', weights_only=False)
            if isinstance(loaded, torch.Tensor):
                emb_3d_table = loaded.to(device)
                log(f"Loaded precomputed 3D embeddings: {emb_3d_table.shape}")
            elif isinstance(loaded, dict):
                # It's a conformer cache - compute embeddings on the fly
                log(f"Loaded conformer cache with {len(loaded)} entries")
                log("Computing 3D embeddings from conformers...")
                from model_reactions.link_prediction.frozen_encoder import FrozenMLIPEncoder, precompute_3d_embeddings
                encoder_3d = FrozenMLIPEncoder(output_dim=512).to(device)
                encoder_3d.eval()
                for param in encoder_3d.parameters():
                    param.requires_grad = False
                emb_3d_table = precompute_3d_embeddings(
                    db.reactant_smiles, loaded, encoder_3d, device=device
                ).to(device)
                log(f"Computed 3D embeddings: {emb_3d_table.shape}")
                # Save for future reuse
                emb_path = conformer_cache.replace('.pt', '_embeddings.pt')
                torch.save(emb_3d_table.cpu(), emb_path)
                log(f"Saved embeddings to {emb_path}")
        else:
            log("Warning: use_3d=True but no conformer cache provided. Disabling 3D.")
            use_3d = False

    # Model
    config = getattr(HypergraphLinkConfigV3, model_size)()
    config.use_3d = use_3d
    config.use_ncn = use_ncn
    config.use_topo = use_topo
    config.use_sketch = use_sketch
    config.use_cf_ncn = use_cf_ncn
    config.cf_ncn_k = cf_ncn_k
    if use_sketch:
        config.sketch_dim = sketch_dim
    if emb_3d_table is not None:
        config.frozen_3d_dim = emb_3d_table.shape[1]
    model = HypergraphLinkPredictorV3(config).to(device)

    # Load pretrained encoder if available
    pretrained_loaded = False
    if pretrained_encoder and os.path.exists(pretrained_encoder):
        log(f"Loading pretrained encoder from {pretrained_encoder}")
        ckpt = torch.load(pretrained_encoder, map_location=device, weights_only=False)
        if 'encoder_state_dict' in ckpt:
            try:
                model.encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
                pretrained_loaded = True
                log("  Loaded pretrained encoder weights")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    log(f"  Warning: pretrained encoder has incompatible dimensions, training from scratch")
                else:
                    raise
        else:
            log("  Warning: no encoder_state_dict found in checkpoint")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model: {model_size}, Parameters: {n_params:,} (trainable: {n_trainable:,})")

    # DDP: wrap model
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        log(f"  Wrapped with DistributedDataParallel (device_ids=[{local_rank}])")

    # Optimizer with differential learning rates if pretrained encoder was loaded
    raw_model = model.module if ddp else model
    if pretrained_loaded:
        encoder_params = list(raw_model.encoder.parameters())
        other_params = [p for n, p in raw_model.named_parameters() if not n.startswith('encoder.')]
        optimizer = AdamW([
            {'params': encoder_params, 'lr': lr * 0.1},
            {'params': other_params, 'lr': lr},
        ], weight_decay=1e-5)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.05)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps, T_mult=1)
    # Manual warmup: linearly scale LR from 0 to target over warmup_steps
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps])

    # NCN: Build adjacency from train edges only (avoid leakage)
    ncn_adj = None
    ncn_emb_table = None
    train_cn_idx, val_cn_idx, test_cn_idx = None, None, None
    if use_ncn:
        log("\n--- NCN: Building reaction graph adjacency ---")
        ncn_adj = build_adjacency(train_edges)
        n_nodes_with_neighbors = sum(1 for v in ncn_adj.values() if len(v) > 0)
        avg_degree = np.mean([len(v) for v in ncn_adj.values()]) if ncn_adj else 0
        log(f"  Nodes with neighbors: {n_nodes_with_neighbors}, avg degree: {avg_degree:.1f}")

        # Precompute CN index tables for each split (one-time)
        log("  Precomputing CN index tables...")
        train_edge_pairs = [(a, b) for a, b, _, _ in train_dataset.edges]
        val_edge_pairs = [(a, b) for a, b, _, _ in val_dataset.edges]
        test_edge_pairs = [(a, b) for a, b, _, _ in test_dataset.edges]
        train_cn_idx = precompute_cn_table(train_edge_pairs, ncn_adj, config.ncn_max_neighbors)
        val_cn_idx = precompute_cn_table(val_edge_pairs, ncn_adj, config.ncn_max_neighbors)
        test_cn_idx = precompute_cn_table(test_edge_pairs, ncn_adj, config.ncn_max_neighbors)

        # Initial embedding table (with untrained model)
        log("  Computing initial embedding table...")
        ncn_emb_table = precompute_embedding_table(raw_model, train_dataset, device)
        if ncn_emb_table is not None:
            log(f"  Embedding table shape: {ncn_emb_table.shape}")

    # Topological heuristic features (Scheme 1)
    topo_train, topo_val, topo_test = None, None, None
    if use_topo:
        log("\n--- Scheme 1: Computing topological heuristic features ---")
        all_train_val_edges = train_edges + val_edges  # val edges are also known at training time
        # Compute features for all splits (using train edges only for adjacency)
        all_pairs = train_edges + [(a, b) for a, b in train_neg]
        all_pairs += val_edges + [(a, b) for a, b in val_neg]
        all_pairs += test_edges + [(a, b) for a, b in test_neg]
        all_pairs = list(set(all_pairs))
        log(f"  Computing topo features for {len(all_pairs)} pairs using {len(train_edges)} train edges...")
        topo_raw = compute_topo_features(all_pairs, n_mols, train_edges)
        topo_train, mean, std = normalize_topo_features(topo_raw)
        # Use train normalization stats for val/test (same dict, already normalized)
        topo_val = topo_train
        topo_test = topo_train
        log(f"  Feature stats (before normalization):")
        raw_feats = np.stack(list(topo_raw.values()))
        feat_names = ['CN', 'AA', 'RA', 'JC', 'PA', 'SP_inv', 'Katz']
        for i, name in enumerate(feat_names):
            log(f"    {name}: mean={raw_feats[:, i].mean():.4f}, "
                f"std={raw_feats[:, i].std():.4f}, "
                f"max={raw_feats[:, i].max():.4f}, "
                f"nonzero={np.count_nonzero(raw_feats[:, i])}/{len(raw_feats)}")

    # Subgraph sketches (Scheme 2)
    sketch_data = None
    if use_sketch:
        log(f"\n--- Scheme 2: Computing MinHash subgraph sketches (dim={sketch_dim}) ---")
        sketch_data = compute_minhash_sketches(
            train_edges, n_mols, sketch_dim=sketch_dim, n_hops=2
        )

    # CF-NCN: Build Morgan FP kNN graph + adjacency + precompute tables (Scheme 4)
    cf_knn_graph = None
    cf_adj = None
    cf_emb_table = None
    train_cf_bridge_idx, train_cf_bridge_wt = None, None
    val_cf_bridge_idx, val_cf_bridge_wt = None, None
    test_cf_bridge_idx, test_cf_bridge_wt = None, None
    if use_cf_ncn:
        log(f"\n--- Scheme 4: CF-NCN (Collaborative Filtering NCN) ---")
        if knn_cache and os.path.exists(knn_cache):
            log(f"  Loading precomputed kNN graph from {knn_cache}...")
            import pickle
            with open(knn_cache, 'rb') as f:
                knn_data = pickle.load(f)
            cf_knn_graph = knn_data['knn_graph']
            log(f"  Loaded kNN graph: {knn_data['n_molecules']} mols, k={knn_data['k']}, "
                f"method={knn_data.get('method', 'unknown')}")
        else:
            log(f"  Building Morgan FP kNN graph (k={cf_ncn_k})...")
            cf_knn_graph = build_morgan_knn(
                db.reactant_smiles, k=cf_ncn_k,
                fp_bits=config.cf_ncn_fp_bits, fp_radius=config.cf_ncn_fp_radius
            )
        log(f"  kNN graph built for {len(cf_knn_graph)} molecules")
        cf_adj = build_adjacency(train_edges)
        n_cf_nodes = sum(1 for v in cf_adj.values() if len(v) > 0)
        log(f"  CF adjacency: {n_cf_nodes} nodes with partners")

        # Precompute bridge index tables for each split (one-time, huge speedup)
        log("  Precomputing CF bridge index tables...")
        train_edge_pairs = [(a, b) for a, b, _, _ in train_dataset.edges]
        val_edge_pairs = [(a, b) for a, b, _, _ in val_dataset.edges]
        test_edge_pairs = [(a, b) for a, b, _, _ in test_dataset.edges]
        train_cf_bridge_idx, train_cf_bridge_wt = precompute_cf_bridge_table(
            train_edge_pairs, cf_knn_graph, cf_adj, config.cf_ncn_max_bridge)
        val_cf_bridge_idx, val_cf_bridge_wt = precompute_cf_bridge_table(
            val_edge_pairs, cf_knn_graph, cf_adj, config.cf_ncn_max_bridge)
        test_cf_bridge_idx, test_cf_bridge_wt = precompute_cf_bridge_table(
            test_edge_pairs, cf_knn_graph, cf_adj, config.cf_ncn_max_bridge)

        # Initial embedding table
        log("  Computing initial CF embedding table...")
        cf_emb_table = precompute_embedding_table(raw_model, train_dataset, device)
        if cf_emb_table is not None:
            log(f"  CF embedding table shape: {cf_emb_table.shape}")

    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0

    es_msg = f", early stop patience={early_stop_patience}" if early_stop_patience > 0 else ""
    log(f"\nTraining for {n_epochs} epochs{es_msg}...")
    log("-" * 100)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # DDP: set epoch for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # NCN: refresh embedding table periodically
        if use_ncn and ncn_emb_table is not None and epoch > 1 and (epoch - 1) % ncn_update_freq == 0:
            emb_model = raw_model if ddp else model
            ncn_emb_table = precompute_embedding_table(emb_model, train_dataset, device)

        # CF-NCN: refresh embedding table periodically
        if use_cf_ncn and cf_emb_table is not None and epoch > 1 and (epoch - 1) % cf_ncn_update_freq == 0:
            emb_model = raw_model if ddp else model
            cf_emb_table = precompute_embedding_table(emb_model, train_dataset, device)

        train_metrics = train_epoch(model, train_loader, optimizer, device, config, scheduler,
                                    emb_3d_table,
                                    ncn_emb_table, train_cn_idx,
                                    topo_train, sketch_data,
                                    train_cf_bridge_idx, train_cf_bridge_wt, cf_emb_table)
        val_metrics = evaluate(model, val_loader, device, emb_3d_table,
                               ncn_emb_table, val_cn_idx,
                               topo_val, sketch_data,
                               val_cf_bridge_idx, val_cf_bridge_wt, cf_emb_table)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]['lr']
        log(f"Epoch {epoch:3d}/{n_epochs} | "
             f"Train: loss={train_metrics['total']:.4f} "
             f"(bce={train_metrics['bce']:.3f}, ctr={train_metrics['contrastive']:.3f}, "
             f"rxn={train_metrics['rxn_class']:.3f}) | "
             f"Val: AUC={val_metrics['roc_auc']:.4f}, AP={val_metrics['ap']:.4f}, "
             f"Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f} | "
             f"lr={lr_current:.2e} | {elapsed:.1f}s")

        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            patience_counter = 0
            save_model = raw_model if ddp else model
            best_model_state = {k: v.cpu().clone() for k, v in save_model.state_dict().items()}
            if is_main:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'config': config,
                    'val_auc': best_val_auc,
                    'val_metrics': val_metrics,
                }, os.path.join(save_dir, 'hypergraph_link_v3_best.pt'))
            log(f"  -> Saved best model (val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                log(f"  -> Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    # Load best model and evaluate on test (only on main rank for DDP)
    eval_model = raw_model if ddp else model
    if best_model_state is not None:
        eval_model.load_state_dict(best_model_state)
        eval_model.to(device)

    # Refresh NCN table with best model before final evaluation
    if use_ncn and ncn_emb_table is not None:
        ncn_emb_table = precompute_embedding_table(eval_model, train_dataset, device)

    # Refresh CF-NCN table with best model before final evaluation
    if use_cf_ncn and cf_emb_table is not None:
        cf_emb_table = precompute_embedding_table(eval_model, train_dataset, device)

    log(f"\n{'='*60}")
    log("Evaluating on full test set...")
    test_metrics = evaluate(eval_model, test_loader, device, emb_3d_table,
                            ncn_emb_table, test_cn_idx,
                            topo_test, sketch_data,
                            test_cf_bridge_idx, test_cf_bridge_wt, cf_emb_table)

    log(f"\n{'='*60}")
    log(f"Hypergraph Link Predictor v3 Results")
    log(f"{'='*60}")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            log(f"  {k}: {v:.4f}")
        else:
            log(f"  {k}: {v}")

    # Save final (only on main rank)
    if is_main:
        torch.save({
            'model_state_dict': best_model_state,
            'config': config,
            'test_metrics': test_metrics,
            'val_auc': best_val_auc,
        }, os.path.join(save_dir, 'hypergraph_link_v3_final.pt'))

    return test_metrics


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypergraph Link Predictor v3')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--data_dir', type=str, default='Data/uspto')
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model_size', type=str, default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='model_reactions/checkpoints')
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--use_3d', action='store_true',
                        help='Enable 3D features (requires conformer cache)')
    parser.add_argument('--conformer_cache', type=str, default=None,
                        help='Path to precomputed conformer cache (.pt)')
    parser.add_argument('--use_ncn', action='store_true',
                        help='Enable Neural Common Neighbor features')
    parser.add_argument('--ncn_update_freq', type=int, default=5,
                        help='Refresh NCN embedding table every N epochs')
    parser.add_argument('--use_topo', action='store_true',
                        help='Enable precomputed topological heuristic features (Scheme 1)')
    parser.add_argument('--use_sketch', action='store_true',
                        help='Enable BUDDY-style subgraph sketching (Scheme 2)')
    parser.add_argument('--sketch_dim', type=int, default=128,
                        help='MinHash sketch dimension for Scheme 2')
    parser.add_argument('--use_cf_ncn', action='store_true',
                        help='Enable CF-NCN collaborative filtering bridge molecules (Scheme 4)')
    parser.add_argument('--cf_ncn_k', type=int, default=200,
                        help='Number of kNN neighbors for CF-NCN')
    parser.add_argument('--cf_ncn_update_freq', type=int, default=5,
                        help='Refresh CF-NCN embedding table every N epochs')
    parser.add_argument('--knn_cache', type=str, default=None,
                        help='Path to precomputed kNN graph (from scripts/precompute_knn.py)')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Early stopping patience (0=disabled)')
    args = parser.parse_args()

    if args.train:
        train_v3(
            data_dir=args.data_dir,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_size=args.model_size,
            device=args.device,
            num_workers=args.num_workers,
            save_dir=args.save_dir,
            pretrained_encoder=args.pretrained_encoder,
            use_3d=args.use_3d,
            conformer_cache=args.conformer_cache,
            use_ncn=args.use_ncn,
            ncn_update_freq=args.ncn_update_freq,
            use_topo=args.use_topo,
            use_sketch=args.use_sketch,
            sketch_dim=args.sketch_dim,
            use_cf_ncn=args.use_cf_ncn,
            cf_ncn_k=args.cf_ncn_k,
            cf_ncn_update_freq=args.cf_ncn_update_freq,
            knn_cache=args.knn_cache,
            early_stop_patience=args.early_stop_patience,
        )
    else:
        print("Use --train to train the model")
