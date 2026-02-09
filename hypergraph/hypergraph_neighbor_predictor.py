#!/usr/bin/env python
"""
Hypergraph Neighbor Predictor for Chemical Reactions.

This module implements reaction prediction as hypergraph neighbor prediction:
- 1-hop neighbor: Product of reaction (mol + reactants -> product)
- 2-hop neighbor: Co-reactants (other molecules in same reaction)

The hypergraph structure:
- Nodes: Molecules (reactants and products)
- Hyperedges: Reactions (connecting multiple reactants to products)

Training objective:
- Given a molecule, predict which products are reachable (1-hop)
- Given a molecule, predict which co-reactants are likely (2-hop)

This is analogous to:
- Link prediction in graphs: predict if edge exists between nodes
- Hyperedge prediction: predict if molecules participate in same reaction

Usage:
    predictor = HypergraphNeighborPredictor(checkpoint_path, data_dir)
    products, co_reactants, scores = predictor.predict_neighbors(mol)
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class HypergraphConfig:
    """Configuration for Hypergraph Neighbor Prediction."""
    # Encoder
    atom_dim: int = 128
    edge_dim: int = 64
    encoder_hidden_dim: int = 512
    encoder_layers: int = 4
    mol_embedding_dim: int = 512

    # Hypergraph layers
    hidden_dim: int = 768
    num_hypergraph_layers: int = 6
    num_attention_heads: int = 12
    dropout: float = 0.1

    # Prediction heads
    num_reaction_types: int = 10

    # Rich molecular features (for V2 encoder)
    atom_feat_dim: int = 40   # see ATOM_FEAT_DIM
    edge_feat_dim: int = 12   # see EDGE_FEAT_DIM

    # Directed hypergraph
    directed: bool = True
    set_aggr: str = "attention"  # "mean", "sum", "attention"
    max_reactants: int = 3

    @classmethod
    def medium(cls):
        return cls()


# =============================================================================
# GNN Encoder Components
# =============================================================================

class MPNNLayer(nn.Module):
    """Message Passing Neural Network layer."""
    
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


class MoleculeEncoder(nn.Module):
    """GNN encoder for molecules."""
    
    def __init__(self, config: HypergraphConfig):
        super().__init__()
        self.atom_embedding = nn.Embedding(120, config.atom_dim)
        self.input_proj = nn.Linear(config.atom_dim, config.encoder_hidden_dim)
        self.edge_embedding = nn.Embedding(5, config.edge_dim)
        
        self.layers = nn.ModuleList([
            MPNNLayer(config.encoder_hidden_dim, config.edge_dim, config.dropout)
            for _ in range(config.encoder_layers)
        ])
        
        self.output_proj = nn.Linear(config.encoder_hidden_dim, config.mol_embedding_dim)
        self.hidden_dim = config.encoder_hidden_dim
        
    def forward(self, atom_types, edge_index, edge_types, batch_idx):
        x = self.input_proj(self.atom_embedding(atom_types))
        edge_attr = self.edge_embedding(edge_types)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Global mean pooling
        batch_size = batch_idx.max().item() + 1
        out = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        count = torch.zeros(batch_size, 1, device=x.device)
        
        out.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, self.hidden_dim), x)
        count.scatter_add_(0, batch_idx.unsqueeze(-1), torch.ones_like(batch_idx, dtype=torch.float).unsqueeze(-1))
        
        return self.output_proj(out / count.clamp(min=1))


class MoleculeEncoderV2(nn.Module):
    """GNN encoder with rich atom/bond features (linear projection, no embedding)."""

    def __init__(self, config: HypergraphConfig):
        super().__init__()
        self.atom_proj = nn.Linear(config.atom_feat_dim, config.encoder_hidden_dim)
        self.edge_proj = nn.Linear(config.edge_feat_dim, config.edge_dim)

        self.layers = nn.ModuleList([
            MPNNLayer(config.encoder_hidden_dim, config.edge_dim, config.dropout)
            for _ in range(config.encoder_layers)
        ])

        self.output_proj = nn.Linear(config.encoder_hidden_dim, config.mol_embedding_dim)
        self.hidden_dim = config.encoder_hidden_dim

    def forward(self, atom_features, edge_index, edge_features, batch_idx):
        """
        Args:
            atom_features: (N, atom_feat_dim) float
            edge_index: (2, E) long
            edge_features: (E, edge_feat_dim) float
            batch_idx: (N,) long
        """
        x = self.atom_proj(atom_features)
        edge_attr = self.edge_proj(edge_features) if edge_features.numel() > 0 else \
            torch.zeros(0, self.edge_proj.out_features, device=atom_features.device)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        batch_size = batch_idx.max().item() + 1
        out = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        count = torch.zeros(batch_size, 1, device=x.device)

        out.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, self.hidden_dim), x)
        count.scatter_add_(0, batch_idx.unsqueeze(-1),
                           torch.ones_like(batch_idx, dtype=torch.float).unsqueeze(-1))

        return self.output_proj(out / count.clamp(min=1))


# =============================================================================
# Hypergraph Attention for Neighbor Aggregation
# =============================================================================

class HypergraphNeighborAttention(nn.Module):
    """
    Hypergraph attention for aggregating information from hyperedge neighbors.
    
    In reaction hypergraph:
    - Hyperedge connects: [reactant1, reactant2, ..., product1, product2, ...]
    - 1-hop neighbors of a reactant: products in same hyperedge
    - 2-hop neighbors: other reactants (co-reactants)
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention for 1-hop (reactant -> product)
        self.q_proj_1hop = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_1hop = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_1hop = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention for 2-hop (reactant -> co-reactant)
        self.q_proj_2hop = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_2hop = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_2hop = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, mol_emb, product_embs=None, co_reactant_embs=None):
        """
        Args:
            mol_emb: Query molecule embedding (batch, hidden_dim)
            product_embs: 1-hop neighbor embeddings (batch, num_products, hidden_dim) or None
            co_reactant_embs: 2-hop neighbor embeddings (batch, num_co_reactants, hidden_dim) or None
        """
        batch_size = mol_emb.size(0)
        residual = mol_emb
        
        updates = []
        
        # 1-hop attention (to products)
        if product_embs is not None:
            q = self.q_proj_1hop(mol_emb).view(batch_size, 1, self.num_heads, self.head_dim)
            k = self.k_proj_1hop(product_embs).view(batch_size, -1, self.num_heads, self.head_dim)
            v = self.v_proj_1hop(product_embs).view(batch_size, -1, self.num_heads, self.head_dim)
            
            # Attention: (batch, 1, heads, head_dim) x (batch, n, heads, head_dim) -> (batch, 1, heads, n)
            scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            
            # Aggregate: (batch, heads, 1, n) x (batch, n, heads, head_dim) -> (batch, 1, heads, head_dim)
            hop1_out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
            hop1_out = hop1_out.view(batch_size, self.hidden_dim)
            updates.append(hop1_out)
        
        # 2-hop attention (to co-reactants)
        if co_reactant_embs is not None:
            q = self.q_proj_2hop(mol_emb).view(batch_size, 1, self.num_heads, self.head_dim)
            k = self.k_proj_2hop(co_reactant_embs).view(batch_size, -1, self.num_heads, self.head_dim)
            v = self.v_proj_2hop(co_reactant_embs).view(batch_size, -1, self.num_heads, self.head_dim)
            
            scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            
            hop2_out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
            hop2_out = hop2_out.view(batch_size, self.hidden_dim)
            updates.append(hop2_out)
        
        # Combine updates
        if updates:
            if len(updates) == 2:
                combined = torch.cat(updates, dim=-1)
                update = self.out_proj(combined)
            else:
                update = updates[0]
            mol_emb = self.norm(residual + self.dropout(update))
        
        # FFN
        mol_emb = self.ffn_norm(mol_emb + self.dropout(self.ffn(mol_emb)))
        
        return mol_emb


# =============================================================================
# Hypergraph Neighbor Prediction Model
# =============================================================================

class HypergraphNeighborNet(nn.Module):
    """
    Hypergraph Neural Network for Neighbor Prediction.
    
    Given a molecule, predicts:
    1. Product embeddings (1-hop neighbors)
    2. Co-reactant embeddings (2-hop neighbors)
    3. Reaction type (which hyperedge type)
    
    Training: Contrastive learning
    - Positive: actual products/co-reactants from reactions
    - Negative: random molecules
    """
    
    def __init__(self, config: HypergraphConfig):
        super().__init__()
        self.config = config
        
        # Molecule encoder
        self.encoder = MoleculeEncoder(config)
        
        # Project to hypergraph space
        self.input_proj = nn.Linear(config.mol_embedding_dim, config.hidden_dim)
        
        # Hypergraph attention layers
        self.hypergraph_layers = nn.ModuleList([
            HypergraphNeighborAttention(config.hidden_dim, config.num_attention_heads, config.dropout)
            for _ in range(config.num_hypergraph_layers)
        ])
        
        # Prediction heads
        # 1-hop predictor: molecule -> product embedding
        self.product_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.mol_embedding_dim)
        )
        
        # 2-hop predictor: molecule -> co-reactant embedding
        self.co_reactant_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.mol_embedding_dim)
        )
        
        # Reaction type classifier
        self.reaction_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_reaction_types)
        )
        
        # Score predictor (reaction feasibility)
        self.score_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
    def encode_molecule(self, atom_types, edge_index, edge_types, batch_idx):
        """Encode molecule to embedding."""
        return self.encoder(atom_types, edge_index, edge_types, batch_idx)
    
    def forward(self, mol_atoms, mol_edges, mol_edge_types, mol_batch,
                product_atoms=None, product_edges=None, product_edge_types=None, product_batch=None,
                co_react_atoms=None, co_react_edges=None, co_react_edge_types=None, co_react_batch=None):
        """
        Forward pass for training/inference.
        
        During training: provide products and co-reactants for supervision
        During inference: only provide query molecule
        """
        # Encode query molecule
        mol_emb = self.encoder(mol_atoms, mol_edges, mol_edge_types, mol_batch)
        mol_h = self.input_proj(mol_emb)
        
        # Encode products if provided (for training)
        product_emb = None
        if product_atoms is not None:
            product_emb = self.encoder(product_atoms, product_edges, product_edge_types, product_batch)
            product_h = self.input_proj(product_emb)
        else:
            product_h = None
        
        # Encode co-reactants if provided
        co_react_emb = None
        if co_react_atoms is not None:
            co_react_emb = self.encoder(co_react_atoms, co_react_edges, co_react_edge_types, co_react_batch)
            co_react_h = self.input_proj(co_react_emb)
        else:
            co_react_h = None
        
        # Apply hypergraph attention layers
        for layer in self.hypergraph_layers:
            # Reshape for attention (add sequence dimension)
            p_h = product_h.unsqueeze(1) if product_h is not None else None
            c_h = co_react_h.unsqueeze(1) if co_react_h is not None else None
            mol_h = layer(mol_h, p_h, c_h)
        
        # Predict neighbor embeddings
        pred_product_emb = self.product_predictor(mol_h)
        pred_co_react_emb = self.co_reactant_predictor(mol_h)
        
        # Predict reaction type and score
        rxn_logits = self.reaction_classifier(mol_h)
        rxn_score = self.score_predictor(mol_h).squeeze(-1)
        
        return {
            'mol_embedding': mol_emb,
            'mol_hidden': mol_h,
            'pred_product_emb': pred_product_emb,
            'pred_co_react_emb': pred_co_react_emb,
            'true_product_emb': product_emb,
            'true_co_react_emb': co_react_emb,
            'rxn_logits': rxn_logits,
            'rxn_score': torch.sigmoid(rxn_score)
        }
    
    def predict_neighbors(self, mol_atoms, mol_edges, mol_edge_types, mol_batch):
        """
        Predict 1-hop (products) and 2-hop (co-reactants) neighbor embeddings.

        Returns embeddings that can be used to retrieve similar molecules from database.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            outputs = self.forward(mol_atoms, mol_edges, mol_edge_types, mol_batch)
        if was_training:
            self.train()

        return {
            'product_emb': outputs['pred_product_emb'],
            'co_reactant_emb': outputs['pred_co_react_emb'],
            'rxn_score': outputs['rxn_score']
        }


# =============================================================================
# Directed Hypergraph Model Components
# =============================================================================

class TailSetEncoder(nn.Module):
    """Permutation-invariant encoder for reactant sets.

    Given a set of reactant embeddings {r1, r2, ...}, produces a single
    tail representation that is invariant to the ordering of reactants.

    Modes:
        - "mean": Simple mean pooling (DeepSets baseline)
        - "sum": DeepSets with learnable phi then sum
        - "attention": Self-attention (permutation equivariant) + mean pooling
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, mode: str = "attention",
                 dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        if mode == "attention":
            self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                                   batch_first=True, dropout=dropout)
            self.norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.ffn_norm = nn.LayerNorm(hidden_dim)
        elif mode == "sum":
            self.phi = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(self, reactant_embs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            reactant_embs: (batch, max_reactants, hidden_dim)
            mask: (batch, max_reactants) — True for padding positions
        Returns:
            (batch, hidden_dim) — permutation-invariant tail representation
        """
        if self.mode == "attention":
            h, _ = self.self_attn(reactant_embs, reactant_embs, reactant_embs,
                                  key_padding_mask=mask)
            h = self.norm(reactant_embs + h)
            h = self.ffn_norm(h + self.ffn(h))
            if mask is not None:
                h = h.masked_fill(mask.unsqueeze(-1), 0.0)
                counts = (~mask).sum(-1, keepdim=True).clamp(min=1).float()
                return h.sum(1) / counts
            return h.mean(1)
        elif self.mode == "sum":
            h = self.phi(reactant_embs)
            if mask is not None:
                h = h.masked_fill(mask.unsqueeze(-1), 0.0)
            return h.sum(1)
        else:  # mean
            if mask is not None:
                h = reactant_embs.masked_fill(mask.unsqueeze(-1), 0.0)
                counts = (~mask).sum(-1, keepdim=True).clamp(min=1).float()
                return h.sum(1) / counts
            return reactant_embs.mean(1)


class DirectedHypergraphNet(nn.Module):
    """Two-stage directed hypergraph model.

    Stage 1 (co-reactant prediction): single reactant -> CoReactantHead -> embedding
    Stage 2 (product prediction):     TailSetEncoder({r1, r2, ...}) -> ProductHead -> embedding

    Stage 2 is permutation-invariant over the reactant set.
    """

    def __init__(self, config: HypergraphConfig):
        super().__init__()
        self.config = config

        # Shared molecule encoder (V2: rich features)
        self.encoder = MoleculeEncoderV2(config)

        # Project mol embedding to hidden space
        self.input_proj = nn.Linear(config.mol_embedding_dim, config.hidden_dim)

        # Role embeddings: 0 = reactant (tail), 1 = product (head)
        self.role_embedding = nn.Embedding(2, config.hidden_dim)

        # Permutation-invariant reactant-set encoder
        self.tail_encoder = TailSetEncoder(
            config.hidden_dim,
            num_heads=config.num_attention_heads,
            mode=config.set_aggr,
            dropout=config.dropout,
        )

        # Stage 1: single reactant -> co-reactant embedding
        self.co_reactant_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.mol_embedding_dim),
        )

        # Stage 2: tail representation -> product embedding
        self.product_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.mol_embedding_dim),
        )

        # Reaction type classifier (from tail repr)
        self.reaction_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_reaction_types),
        )

        # Score predictor (reaction feasibility, from tail repr)
        self.score_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    # -- helpers --
    ROLE_TAIL = 0
    ROLE_HEAD = 1

    def encode_molecule(self, atom_features, edge_index, edge_features, batch_idx):
        return self.encoder(atom_features, edge_index, edge_features, batch_idx)

    def _proj_with_role(self, mol_emb: torch.Tensor, role: int) -> torch.Tensor:
        """Project mol embedding to hidden dim and add role embedding."""
        h = self.input_proj(mol_emb)
        h = h + self.role_embedding.weight[role]
        return h

    # -- training forward --
    def forward(self, reactant_graph_list, product_graph):
        """Training forward.

        Args:
            reactant_graph_list: list of dicts, each with keys
                (atom_features, edge_index, edge_features, batch) — one per reactant slot.
                Length = max_reactants; padding slots have None.
            product_graph: dict with (atom_features, edge_index, edge_features, batch)

        Returns dict with losses-relevant tensors.
        """
        device = next(self.parameters()).device
        batch_size = product_graph['batch'].max().item() + 1

        # Encode each reactant
        reactant_embs = []  # raw encoder outputs (batch, mol_emb_dim)
        reactant_h = []     # projected+role (batch, hidden_dim)
        valid_mask_cols = []  # bool: True if this slot is real

        for rg in reactant_graph_list:
            if rg is None:
                reactant_embs.append(None)
                reactant_h.append(None)
                valid_mask_cols.append(False)
            else:
                emb = self.encoder(rg['atom_features'], rg['edge_index'],
                                   rg['edge_features'], rg['batch'])
                reactant_embs.append(emb)
                reactant_h.append(self._proj_with_role(emb, self.ROLE_TAIL))
                valid_mask_cols.append(True)

        num_slots = len(reactant_graph_list)

        # Stack into (batch, num_slots, hidden_dim), build padding mask
        h_stack = torch.zeros(batch_size, num_slots, self.config.hidden_dim, device=device)
        pad_mask = torch.ones(batch_size, num_slots, dtype=torch.bool, device=device)

        for s, (h, valid) in enumerate(zip(reactant_h, valid_mask_cols)):
            if valid:
                h_stack[:, s, :] = h
                pad_mask[:, s] = False

        # Stage 1: each reactant independently predicts co-reactants
        co_react_preds = self.co_reactant_predictor(h_stack)  # (B, S, emb_dim)

        # Stage 2: permutation-invariant aggregation -> product prediction
        tail_repr = self.tail_encoder(h_stack, mask=pad_mask)  # (B, hidden)
        pred_product_emb = self.product_predictor(tail_repr)   # (B, emb_dim)

        # Encode product
        product_emb = self.encoder(product_graph['atom_features'], product_graph['edge_index'],
                                   product_graph['edge_features'], product_graph['batch'])

        # Classifier + score from tail representation
        rxn_logits = self.reaction_classifier(tail_repr)
        rxn_score = torch.sigmoid(self.score_predictor(tail_repr).squeeze(-1))

        return {
            'pred_product_emb': pred_product_emb,
            'co_react_preds': co_react_preds,     # (B, S, emb_dim)
            'true_product_emb': product_emb,
            'true_reactant_embs': reactant_embs,   # list of (B, emb_dim) or None
            'valid_mask': ~pad_mask,               # (B, S) True=real
            'tail_repr': tail_repr,
            'rxn_logits': rxn_logits,
            'rxn_score': rxn_score,
        }

    # -- inference --
    def predict_neighbors(self, atom_features, edge_index, edge_features, batch_idx):
        """Single-molecule inference (for RL).

        Returns embeddings for co-reactant retrieval and product retrieval.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            mol_emb = self.encoder(atom_features, edge_index, edge_features, batch_idx)
            mol_h = self._proj_with_role(mol_emb, self.ROLE_TAIL)

            # Stage 1: co-reactant prediction
            pred_co_react_emb = self.co_reactant_predictor(mol_h)

            # Stage 2: product prediction with single-reactant set
            tail_repr = self.tail_encoder(mol_h.unsqueeze(1))  # (B,1,H)
            pred_product_emb = self.product_predictor(tail_repr)

            rxn_score = torch.sigmoid(self.score_predictor(tail_repr).squeeze(-1))

        if was_training:
            self.train()

        return {
            'product_emb': pred_product_emb,
            'co_reactant_emb': pred_co_react_emb,
            'rxn_score': rxn_score,
        }

    def predict_product_with_coreactants(self, mol_emb, co_react_embs):
        """Refined product prediction given retrieved co-reactants.

        Args:
            mol_emb: (B, mol_emb_dim) — raw encoder output for query mol
            co_react_embs: (B, N, mol_emb_dim) — raw encoder outputs for co-reactants
        Returns:
            (B, mol_emb_dim) — refined product embedding
        """
        with torch.no_grad():
            mol_h = self._proj_with_role(mol_emb, self.ROLE_TAIL)
            co_h = self._proj_with_role(co_react_embs, self.ROLE_TAIL)
            # Concat along set dimension: (B, 1+N, H)
            h_stack = torch.cat([mol_h.unsqueeze(1), co_h], dim=1)
            tail_repr = self.tail_encoder(h_stack)
            return self.product_predictor(tail_repr)


# =============================================================================
# Neighbor Predictor Interface
# =============================================================================

def smiles_to_graph(smiles: str):
    """Convert SMILES to graph representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_index, edge_types = [], []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bond_type = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                     Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}.get(bt, 0)
        edge_index.extend([[i, j], [j, i]])
        edge_types.extend([bond_type, bond_type])
    
    if not edge_index:
        edge_index, edge_types = [[0, 0]], [0]
    
    return {'atom_types': atom_types, 'edge_index': edge_index, 'edge_types': edge_types}


# --- Rich feature constants for V2 graphs ---
_COMMON_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 14, 5]  # C N O F P S Cl Br I Si B
_DEGREES = [0, 1, 2, 3, 4, 5, 6]
_CHARGES = [-2, -1, 0, 1, 2]
_NUM_HS = [0, 1, 2, 3, 4]
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
               Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
_STEREOS = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ]

ATOM_FEAT_DIM = ((len(_COMMON_ATOMS) + 1) + (len(_DEGREES) + 1) +
                 (len(_CHARGES) + 1) + (len(_NUM_HS) + 1) +
                 (len(_HYBRIDIZATIONS) + 1) + 2)  # = 40
EDGE_FEAT_DIM = (len(_BOND_TYPES) + 1) + (len(_STEREOS) + 1) + 2  # = 12


def _one_hot(val, choices):
    enc = [0] * (len(choices) + 1)
    try:
        enc[choices.index(val)] = 1
    except ValueError:
        enc[-1] = 1
    return enc


def smiles_to_graph_v2(smiles: str):
    """Convert SMILES to pre-tensorized graph with rich features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        f = _one_hot(atom.GetAtomicNum(), _COMMON_ATOMS)
        f += _one_hot(atom.GetDegree(), _DEGREES)
        f += _one_hot(atom.GetFormalCharge(), _CHARGES)
        f += _one_hot(atom.GetTotalNumHs(), _NUM_HS)
        f += _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        f.append(int(atom.GetIsAromatic()))
        f.append(int(atom.IsInRing()))
        atom_feats.append(f)

    # Edge features
    edge_index = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _one_hot(bond.GetBondType(), _BOND_TYPES)
        bf += _one_hot(bond.GetStereo(), _STEREOS)
        bf.append(int(bond.GetIsConjugated()))
        bf.append(int(bond.IsInRing()))
        edge_index.extend([[i, j], [j, i]])
        edge_feats.extend([bf, bf])

    af = torch.tensor(atom_feats, dtype=torch.float)
    if edge_index:
        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        ef = torch.tensor(edge_feats, dtype=torch.float)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
        ef = torch.zeros(0, EDGE_FEAT_DIM, dtype=torch.float)

    return {'atom_features': af, 'edge_index': ei, 'edge_features': ef}


# Dummy graph for failed parses (V2 format, pre-tensorized)
DUMMY_GRAPH_V2 = {
    'atom_features': torch.zeros(1, ATOM_FEAT_DIM),
    'edge_index': torch.zeros(2, 0, dtype=torch.long),
    'edge_features': torch.zeros(0, EDGE_FEAT_DIM),
}


class HypergraphNeighborPredictor:
    """
    Interface for predicting reaction neighbors using Hypergraph GNN.
    
    Given a molecule, predicts:
    - 1-hop neighbors (products): what products this molecule can form
    - 2-hop neighbors (co-reactants): what other molecules it can react with
    """
    
    # Default checkpoint path
    DEFAULT_CHECKPOINT = "hypergraph/checkpoints/neighbor_predictor_best.pt"
    
    def __init__(self, checkpoint_path: str = None, data_dir: str = "Data/uspto",
                 device: str = "auto", top_k: int = 10, max_index_mols: int = 10000):
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        self.max_index_mols = max_index_mols
        
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.config = HypergraphConfig.medium()
        self.model = HypergraphNeighborNet(self.config).to(self.device)
        
        # Try to load checkpoint
        if checkpoint_path is None:
            # Try default path
            for base in [Path(__file__).parent.parent, Path.cwd()]:
                default_path = base / self.DEFAULT_CHECKPOINT
                if default_path.exists():
                    checkpoint_path = str(default_path)
                    break
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print("Warning: No checkpoint loaded, using random initialization")
        
        # Build molecule database
        self._build_molecule_database()
        
        print(f"HypergraphNeighborPredictor initialized on {self.device}")
    
    def _load_checkpoint(self, path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            # Try to load, skip mismatched keys
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except Exception as e:
                print(f"Could not load checkpoint ({e}), using random initialization")
        self.model.eval()
    
    def _build_molecule_database(self):
        """Build database of molecules with their embeddings."""
        print("Building molecule database...")
        
        # Load reactions from USPTO
        train_path = self.data_dir / "train.csv"
        if not train_path.exists():
            print(f"Data not found at {train_path}")
            self.product_db = {}
            self.reactant_db = {}
            return
        
        df = pd.read_csv(train_path)
        
        # Index products and reactants
        self.product_db = {}  # smiles -> [(reactants, reaction_info), ...]
        self.reactant_db = {}  # smiles -> [(co_reactants, products, reaction_info), ...]
        
        for _, row in df.iterrows():
            rxn_smiles = row.get('rxn_smiles', row.get('canonical_rxn', ''))
            if '>>' not in str(rxn_smiles):
                continue
            
            parts = rxn_smiles.split('>>')
            reactants = parts[0].split('.')
            products = parts[1].split('.') if len(parts) > 1 else []
            rxn_class = int(row.get('class', row.get('rxn_class', 0)))
            
            info = {'rxn_class': rxn_class, 'yield': row.get('yield', 0.5)}
            
            # Index products
            for product in products:
                if product not in self.product_db:
                    self.product_db[product] = []
                self.product_db[product].append((reactants, info))
            
            # Index reactants
            for i, reactant in enumerate(reactants):
                co_reactants = [r for j, r in enumerate(reactants) if j != i]
                if reactant not in self.reactant_db:
                    self.reactant_db[reactant] = []
                self.reactant_db[reactant].append((co_reactants, products, info))
        
        print(f"Indexed {len(self.product_db)} products, {len(self.reactant_db)} reactants")
        
        # Build embedding index
        self._build_embedding_index(max_mols=self.max_index_mols)
    
    def _build_embedding_index(self, max_mols: int = 10000):
        """Build embedding index for fast neighbor lookup."""
        print(f"Building embedding index (max {max_mols} molecules)...")
        
        self.model.eval()
        
        # Sample molecules
        all_mols = list(set(list(self.product_db.keys()) + list(self.reactant_db.keys())))[:max_mols]
        
        embeddings = []
        valid_smiles = []
        
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(all_mols), batch_size):
                batch = all_mols[i:i+batch_size]
                
                # Prepare batch
                all_atoms, all_edges, all_edge_types, all_batch = [], [], [], []
                offset = 0
                batch_valid = []
                
                for j, smiles in enumerate(batch):
                    graph = smiles_to_graph(smiles)
                    if graph is None:
                        continue
                    
                    n = len(graph['atom_types'])
                    all_atoms.extend(graph['atom_types'])
                    for e in graph['edge_index']:
                        all_edges.append([e[0] + offset, e[1] + offset])
                    all_edge_types.extend(graph['edge_types'])
                    all_batch.extend([len(batch_valid)] * n)
                    offset += n
                    batch_valid.append(smiles)
                
                if not batch_valid:
                    continue
                
                # Encode
                atoms = torch.tensor(all_atoms, dtype=torch.long, device=self.device)
                edges = torch.tensor(all_edges, dtype=torch.long, device=self.device).t().contiguous()
                edge_types = torch.tensor(all_edge_types, dtype=torch.long, device=self.device)
                batch_idx = torch.tensor(all_batch, dtype=torch.long, device=self.device)
                
                emb = self.model.encode_molecule(atoms, edges, edge_types, batch_idx)
                embeddings.append(emb.cpu().numpy())
                valid_smiles.extend(batch_valid)
        
        if embeddings:
            self.mol_embeddings = np.vstack(embeddings)
            self.mol_smiles = valid_smiles
            # Normalize
            norms = np.linalg.norm(self.mol_embeddings, axis=1, keepdims=True)
            self.mol_embeddings = self.mol_embeddings / np.where(norms > 0, norms, 1)
            print(f"Built index with {len(self.mol_smiles)} molecules")
        else:
            self.mol_embeddings = None
            self.mol_smiles = []
    
    def _encode_molecule(self, mol):
        """Encode a single molecule."""
        if isinstance(mol, str):
            smiles = mol
            mol = Chem.MolFromSmiles(smiles)
        else:
            smiles = Chem.MolToSmiles(mol)
        
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None
        
        with torch.no_grad():
            atoms = torch.tensor(graph['atom_types'], dtype=torch.long, device=self.device)
            edges = torch.tensor(graph['edge_index'], dtype=torch.long, device=self.device).t().contiguous()
            edge_types = torch.tensor(graph['edge_types'], dtype=torch.long, device=self.device)
            batch_idx = torch.zeros(len(graph['atom_types']), dtype=torch.long, device=self.device)
            
            return self.model.encode_molecule(atoms, edges, edge_types, batch_idx)
    
    def predict_neighbors(self, mol, return_details: bool = False):
        """
        Predict hypergraph neighbors for a molecule.
        
        Args:
            mol: RDKit Mol or SMILES string
            
        Returns:
            products: List of predicted product SMILES (1-hop neighbors)
            co_reactants: List of predicted co-reactant SMILES (2-hop neighbors)
            scores: Reaction feasibility scores
        """
        if isinstance(mol, str):
            smiles = mol
            mol = Chem.MolFromSmiles(smiles)
        else:
            smiles = Chem.MolToSmiles(mol)
        
        if mol is None:
            return [], [], np.array([])
        
        # Get predicted neighbor embeddings
        graph = smiles_to_graph(smiles)
        if graph is None:
            return [], [], np.array([])
        
        with torch.no_grad():
            atoms = torch.tensor(graph['atom_types'], dtype=torch.long, device=self.device)
            edges = torch.tensor(graph['edge_index'], dtype=torch.long, device=self.device).t().contiguous()
            edge_types = torch.tensor(graph['edge_types'], dtype=torch.long, device=self.device)
            batch_idx = torch.zeros(len(graph['atom_types']), dtype=torch.long, device=self.device)
            
            preds = self.model.predict_neighbors(atoms, edges, edge_types, batch_idx)
        
        # Retrieve neighbors using predicted embeddings
        pred_product_emb = preds['product_emb'].cpu().numpy()[0]
        pred_co_react_emb = preds['co_reactant_emb'].cpu().numpy()[0]
        
        # Normalize
        pred_product_emb = pred_product_emb / (np.linalg.norm(pred_product_emb) + 1e-8)
        pred_co_react_emb = pred_co_react_emb / (np.linalg.norm(pred_co_react_emb) + 1e-8)
        
        # Find nearest neighbors in database
        if self.mol_embeddings is not None:
            # 1-hop: products
            product_sims = np.dot(self.mol_embeddings, pred_product_emb)
            product_indices = np.argsort(product_sims)[::-1][:self.top_k]
            products = [(self.mol_smiles[i], product_sims[i]) for i in product_indices]
            
            # 2-hop: co-reactants  
            co_react_sims = np.dot(self.mol_embeddings, pred_co_react_emb)
            co_react_indices = np.argsort(co_react_sims)[::-1][:self.top_k]
            co_reactants = [(self.mol_smiles[i], co_react_sims[i]) for i in co_react_indices]
        else:
            products, co_reactants = [], []
        
        # Also check database for exact matches
        if smiles in self.reactant_db:
            for co_reacts, prods, info in self.reactant_db[smiles][:self.top_k]:
                for p in prods:
                    if p not in [x[0] for x in products]:
                        products.append((p, 1.0))
                for c in co_reacts:
                    if c not in [x[0] for x in co_reactants]:
                        co_reactants.append((c, 1.0))
        
        # Format output
        product_smiles = [p[0] for p in products[:self.top_k]]
        co_react_smiles = [c[0] for c in co_reactants[:self.top_k]]
        scores = np.array([p[1] for p in products[:self.top_k]])
        
        if return_details:
            return product_smiles, co_react_smiles, scores, {
                'product_sims': products,
                'co_reactant_sims': co_reactants,
                'rxn_score': float(preds['rxn_score'][0])
            }
        
        return product_smiles, co_react_smiles, scores
    
    def get_valid_actions(self, mol):
        """
        Compatibility interface with ReactionPredictor.

        Returns:
            acts: Co-reactants (2-hop neighbors)
            rets: Products (1-hop neighbors)
            scores: Reaction scores
        """
        products, co_reactants, scores = self.predict_neighbors(mol)
        return co_reactants, products, scores


class DirectedHypergraphNeighborPredictor:
    """Directed hypergraph predictor with separate reactant/product indices.

    Two-stage inference:
      Stage 1: query mol -> co-reactant embedding -> search reactant_index
      Stage 2: TailSetEncoder({query, co-reactant}) -> product embedding -> search product_index
    """

    DEFAULT_CHECKPOINT = "hypergraph/checkpoints/directed_predictor_best.pt"

    def __init__(self, checkpoint_path: str = None, data_dir: str = "Data/uspto",
                 device: str = "auto", top_k: int = 10, max_index_mols: int = 10000):
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        self.max_index_mols = max_index_mols

        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.config = HypergraphConfig.medium()
        self.model = DirectedHypergraphNet(self.config).to(self.device)

        if checkpoint_path is None:
            for base in [Path(__file__).parent.parent, Path.cwd()]:
                default_path = base / self.DEFAULT_CHECKPOINT
                if default_path.exists():
                    checkpoint_path = str(default_path)
                    break

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print("Warning: No checkpoint loaded, using random initialization")

        self._build_molecule_database()
        print(f"DirectedHypergraphNeighborPredictor initialized on {self.device}")

    def _load_checkpoint(self, path: str):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except Exception:
                print("Could not load checkpoint, using random initialization")
        self.model.eval()

    def _smiles_to_tensors(self, smiles: str):
        """Convert SMILES to device tensors for encode_molecule (V2 format)."""
        graph = smiles_to_graph_v2(smiles)
        if graph is None:
            return None
        af = graph['atom_features'].to(self.device)
        ei = graph['edge_index'].to(self.device)
        ef = graph['edge_features'].to(self.device)
        batch_idx = torch.zeros(af.size(0), dtype=torch.long, device=self.device)
        return af, ei, ef, batch_idx

    def _build_molecule_database(self):
        """Build separate reactant and product embedding indices."""
        print("Building molecule database...")

        train_path = self.data_dir / "train.csv"
        if not train_path.exists():
            print(f"Data not found at {train_path}")
            self.reactant_set = set()
            self.product_set = set()
            self.reactant_db = {}
            return

        df = pd.read_csv(train_path)

        self.reactant_set = set()
        self.product_set = set()
        self.reactant_db = {}  # smiles -> [(co_reactants, products, info), ...]

        for _, row in df.iterrows():
            rxn_smiles = row.get('rxn_smiles', row.get('canonical_rxn', ''))
            if '>>' not in str(rxn_smiles):
                continue
            parts = rxn_smiles.split('>>')
            reactants = parts[0].split('.')
            products = parts[1].split('.') if len(parts) > 1 else []
            rxn_class = int(row.get('class', row.get('rxn_class', 0)))
            info = {'rxn_class': rxn_class}

            self.reactant_set.update(reactants)
            self.product_set.update(products)

            for i, r in enumerate(reactants):
                co = [reactants[j] for j in range(len(reactants)) if j != i]
                if r not in self.reactant_db:
                    self.reactant_db[r] = []
                self.reactant_db[r].append((co, products, info))

        print(f"Indexed {len(self.reactant_set)} reactants, {len(self.product_set)} products")
        self._build_embedding_indices()

    def _encode_smiles_batch(self, smiles_list):
        """Encode a list of SMILES to numpy embeddings (V2 format, torch.cat collation)."""
        embeddings, valid = [], []
        self.model.eval()
        with torch.no_grad():
            bs = 64
            for i in range(0, len(smiles_list), bs):
                chunk = smiles_list[i:i+bs]
                af_parts, ei_parts, ef_parts, b_parts = [], [], [], []
                offset, bv = 0, []
                for smi in chunk:
                    graph = smiles_to_graph_v2(smi)
                    if graph is None:
                        continue
                    af = graph['atom_features']
                    ei = graph['edge_index']
                    ef = graph['edge_features']
                    n = af.size(0)
                    af_parts.append(af)
                    if ei.numel() > 0:
                        ei_parts.append(ei + offset)
                    ef_parts.append(ef)
                    b_parts.append(torch.full((n,), len(bv), dtype=torch.long))
                    offset += n
                    bv.append(smi)
                if not bv:
                    continue
                af_t = torch.cat(af_parts, dim=0).to(self.device)
                ei_t = (torch.cat(ei_parts, dim=1) if ei_parts
                        else torch.zeros(2, 0, dtype=torch.long)).to(self.device)
                ef_t = (torch.cat(ef_parts, dim=0) if ef_parts
                        else torch.zeros(0, EDGE_FEAT_DIM)).to(self.device)
                b_t = torch.cat(b_parts, dim=0).to(self.device)
                emb = self.model.encode_molecule(af_t, ei_t, ef_t, b_t)
                embeddings.append(emb.cpu().numpy())
                valid.extend(bv)
        if embeddings:
            embs = np.vstack(embeddings)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.where(norms > 0, norms, 1)
            return embs, valid
        return None, []

    def _build_embedding_indices(self):
        """Build separate indices for reactants and products."""
        max_r = self.max_index_mols
        max_p = self.max_index_mols

        r_list = list(self.reactant_set)[:max_r]
        p_list = list(self.product_set)[:max_p]

        print(f"Building reactant index ({len(r_list)} mols)...")
        self.reactant_embeddings, self.reactant_smiles = self._encode_smiles_batch(r_list)

        print(f"Building product index ({len(p_list)} mols)...")
        self.product_embeddings, self.product_smiles = self._encode_smiles_batch(p_list)

        nr = len(self.reactant_smiles) if self.reactant_smiles else 0
        np_ = len(self.product_smiles) if self.product_smiles else 0
        print(f"Built indices: {nr} reactants, {np_} products")

    def predict_neighbors(self, mol, return_details: bool = False):
        """Predict products and co-reactants using two-stage directed prediction."""
        if isinstance(mol, str):
            smiles = mol
            mol_obj = Chem.MolFromSmiles(smiles)
        else:
            smiles = Chem.MolToSmiles(mol)
            mol_obj = mol

        if mol_obj is None:
            return [], [], np.array([])

        tensors = self._smiles_to_tensors(smiles)
        if tensors is None:
            return [], [], np.array([])

        # Stage 1: predict co-reactant embedding
        preds = self.model.predict_neighbors(*tensors)
        pred_co_emb = preds['co_reactant_emb'].cpu().numpy()[0]
        pred_co_emb = pred_co_emb / (np.linalg.norm(pred_co_emb) + 1e-8)

        # Search reactant index for co-reactants
        co_reactants = []
        if self.reactant_embeddings is not None:
            sims = np.dot(self.reactant_embeddings, pred_co_emb)
            indices = np.argsort(sims)[::-1][:self.top_k]
            co_reactants = [(self.reactant_smiles[i], float(sims[i])) for i in indices]

        # Stage 2: refined product prediction using retrieved co-reactants
        pred_prod_emb = preds['product_emb'].cpu().numpy()[0]

        if co_reactants:
            # Encode the top co-reactant and do refined prediction
            top_co_smi = co_reactants[0][0]
            co_tensors = self._smiles_to_tensors(top_co_smi)
            if co_tensors is not None:
                with torch.no_grad():
                    mol_raw = self.model.encode_molecule(*tensors)
                    co_raw = self.model.encode_molecule(*co_tensors)
                    refined = self.model.predict_product_with_coreactants(
                        mol_raw, co_raw.unsqueeze(1))
                    pred_prod_emb = refined.cpu().numpy()[0]

        pred_prod_emb = pred_prod_emb / (np.linalg.norm(pred_prod_emb) + 1e-8)

        # Search product index
        products = []
        if self.product_embeddings is not None:
            sims = np.dot(self.product_embeddings, pred_prod_emb)
            indices = np.argsort(sims)[::-1][:self.top_k]
            products = [(self.product_smiles[i], float(sims[i])) for i in indices]

        # Exact-match augmentation
        if smiles in self.reactant_db:
            for co_reacts, prods, info in self.reactant_db[smiles][:self.top_k]:
                for p in prods:
                    if p not in [x[0] for x in products]:
                        products.append((p, 1.0))
                for c in co_reacts:
                    if c not in [x[0] for x in co_reactants]:
                        co_reactants.append((c, 1.0))

        product_smiles = [p[0] for p in products[:self.top_k]]
        co_react_smiles = [c[0] for c in co_reactants[:self.top_k]]
        scores = np.array([p[1] for p in products[:self.top_k]])

        if return_details:
            return product_smiles, co_react_smiles, scores, {
                'product_sims': products,
                'co_reactant_sims': co_reactants,
                'rxn_score': float(preds['rxn_score'][0]),
            }
        return product_smiles, co_react_smiles, scores

    def get_valid_actions(self, mol):
        """Compatibility interface with ReactionPredictor."""
        products, co_reactants, scores = self.predict_neighbors(mol)
        return co_reactants, products, scores


# =============================================================================
# Demo
# =============================================================================

def main():
    print("=" * 60)
    print("Hypergraph Neighbor Predictor Demo")
    print("=" * 60)
    
    # Use trained checkpoint
    checkpoint = "hypergraph/checkpoints/neighbor_predictor_best.pt"
    
    predictor = HypergraphNeighborPredictor(
        checkpoint_path=checkpoint,
        data_dir="Data/uspto",
        top_k=10,
        max_index_mols=10000
    )
    
    test_mols = [
        'CCO',           # Ethanol
        'c1ccccc1O',     # Phenol
        'CC(=O)O',       # Acetic acid
        'CC(=O)Cl',      # Acetyl chloride
        'c1ccc(N)cc1',   # Aniline
    ]
    
    print("\n" + "=" * 60)
    print("Predicting Reaction Neighbors")
    print("=" * 60)
    
    for smiles in test_mols:
        print(f"\n{'='*40}")
        print(f"Input: {smiles}")
        print(f"{'='*40}")
        
        products, co_reactants, scores, details = predictor.predict_neighbors(smiles, return_details=True)
        
        print(f"\n1-hop Neighbors (Predicted Products):")
        for i, (prod, score) in enumerate(zip(products[:5], scores[:5])):
            print(f"  {i+1}. {prod} (score: {score:.3f})")
        
        print(f"\n2-hop Neighbors (Predicted Co-reactants):")
        for i, co in enumerate(co_reactants[:5]):
            print(f"  {i+1}. {co}")
        
        print(f"\nReaction Score: {details['rxn_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("Testing get_valid_actions() Interface")
    print("=" * 60)
    
    mol = Chem.MolFromSmiles('c1ccccc1Br')  # Bromobenzene
    acts, rets, scores = predictor.get_valid_actions(mol)
    
    print(f"\nInput: Bromobenzene (c1ccccc1Br)")
    print(f"Co-reactants (acts): {acts[:5]}")
    print(f"Products (rets): {rets[:5]}")
    print(f"Scores: {scores[:5]}")


if __name__ == "__main__":
    main()
