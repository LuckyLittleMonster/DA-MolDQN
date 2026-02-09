"""
Hypergraph Link Predictor for Reaction Partner Prediction (Model 1a).

Architecture:
  1. MoleculeEncoder (MPNN-based GNN) encodes molecules to embeddings
  2. Hypergraph-aware projector maps embeddings for link prediction
  3. Scoring: dot product between projected embeddings

Training:
  - Contrastive learning: positive pairs (molecules that react) vs negatives
  - InfoNCE loss for learning discriminative embeddings
  
Inference (for RL):
  - Pre-compute embeddings for all database molecules
  - For query molecule: encode -> find nearest neighbors -> return co-reactants
  - Target: <10ms per query

Usage:
    python -m model_reactions.hypergraph_link_predictor --train --data_dir Data/uspto
    python -m model_reactions.hypergraph_link_predictor --benchmark --data_dir Data/uspto
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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model_reactions.reactant_predictor import ReactionDatabase


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HypergraphLinkConfig:
    """Configuration for Hypergraph Link Predictor."""
    # GNN Encoder
    atom_dim: int = 64
    edge_dim: int = 32
    encoder_hidden_dim: int = 256
    encoder_layers: int = 3
    mol_embedding_dim: int = 256
    
    # Link predictor
    hidden_dim: int = 256
    num_link_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.1
    
    # Training
    temperature: float = 0.07
    margin: float = 0.5
    
    @classmethod
    def small(cls):
        return cls(atom_dim=32, edge_dim=16, encoder_hidden_dim=128, 
                   encoder_layers=2, mol_embedding_dim=128, hidden_dim=128,
                   num_link_layers=1, num_attention_heads=2)
    
    @classmethod
    def medium(cls):
        return cls()
    
    @classmethod
    def large(cls):
        return cls(atom_dim=128, edge_dim=64, encoder_hidden_dim=512,
                   encoder_layers=4, mol_embedding_dim=512, hidden_dim=512,
                   num_link_layers=3, num_attention_heads=8)


# =============================================================================
# GNN Molecule Encoder (reused from hypergraph/hypergraph_neighbor_predictor.py)
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
    
    def __init__(self, config: HypergraphLinkConfig):
        super().__init__()
        self.atom_embedding = nn.Embedding(100, config.atom_dim)
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


# =============================================================================
# Hypergraph Link Prediction Model
# =============================================================================

class HypergraphLinkPredictor(nn.Module):
    """
    Hypergraph-aware link predictor for reaction partner prediction.
    
    Given a molecule, produces an embedding optimized for finding reaction partners
    through nearest-neighbor retrieval.
    
    The hypergraph structure captures that molecules in the same reaction form
    a hyperedge, allowing the model to learn complex reaction patterns beyond
    pairwise similarity.
    """
    
    def __init__(self, config: HypergraphLinkConfig):
        super().__init__()
        self.config = config
        
        # Molecule encoder
        self.encoder = MoleculeEncoder(config)
        
        # Link prediction projector: maps mol embedding -> link-aware embedding
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
        
        # Pairwise scoring head (for explicit link scoring)
        self.pair_scorer = nn.Sequential(
            nn.Linear(config.mol_embedding_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def encode_molecule(self, atom_types, edge_index, edge_types, batch_idx):
        """Encode molecule to embedding."""
        mol_emb = self.encoder(atom_types, edge_index, edge_types, batch_idx)
        link_emb = self.link_projector(mol_emb)
        return mol_emb, link_emb
    
    def compute_pair_score(self, emb_a, emb_b):
        """
        Compute pairwise reaction score.
        
        Args:
            emb_a: (batch, dim) - molecule A embedding
            emb_b: (batch, dim) - molecule B embedding
            
        Returns:
            score: (batch,) - reaction probability logit
        """
        # Element-wise product captures interaction
        combined = torch.cat([emb_a, emb_b, emb_a * emb_b], dim=-1)
        return self.pair_scorer(combined).squeeze(-1)
    
    def forward(self, mol_a_atoms, mol_a_edges, mol_a_edge_types, mol_a_batch,
                mol_b_atoms, mol_b_edges, mol_b_edge_types, mol_b_batch):
        """
        Forward pass for training.
        
        Args:
            mol_a_*: Molecule A graph data
            mol_b_*: Molecule B graph data
            
        Returns:
            dict with embeddings and scores
        """
        _, link_emb_a = self.encode_molecule(mol_a_atoms, mol_a_edges, mol_a_edge_types, mol_a_batch)
        _, link_emb_b = self.encode_molecule(mol_b_atoms, mol_b_edges, mol_b_edge_types, mol_b_batch)
        
        # Pair score
        score = self.compute_pair_score(link_emb_a, link_emb_b)
        
        return {
            'link_emb_a': link_emb_a,
            'link_emb_b': link_emb_b,
            'score': score,
        }


# =============================================================================
# Dataset
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


class ReactionLinkDataset(Dataset):
    """Dataset for link prediction training."""
    
    def __init__(self, pos_edges, neg_edges, smiles_list, precompute=True):
        """
        Args:
            pos_edges: list of (idx_a, idx_b) positive pairs
            neg_edges: list of (idx_a, idx_b) negative pairs
            smiles_list: list of SMILES strings indexed by molecule index
            precompute: whether to precompute graphs
        """
        self.edges = [(a, b, 1) for a, b in pos_edges] + [(a, b, 0) for a, b in neg_edges]
        self.smiles_list = smiles_list
        
        # Precompute graphs
        self.graph_cache = {}
        if precompute:
            unique_mols = set()
            for a, b, _ in self.edges:
                unique_mols.add(a)
                unique_mols.add(b)
            
            print(f"Precomputing graphs for {len(unique_mols)} molecules...")
            for idx in tqdm(unique_mols, desc="Graphs"):
                if idx < len(self.smiles_list):
                    graph = smiles_to_graph(self.smiles_list[idx])
                    if graph is not None:
                        self.graph_cache[idx] = graph
            print(f"Cached {len(self.graph_cache)} valid graphs")
    
    def __len__(self):
        return len(self.edges)
    
    def _get_graph(self, idx):
        if idx in self.graph_cache:
            return self.graph_cache[idx]
        if idx < len(self.smiles_list):
            graph = smiles_to_graph(self.smiles_list[idx])
            if graph is not None:
                return graph
        return {'atom_types': [6], 'edge_index': [[0, 0]], 'edge_types': [0]}
    
    def __getitem__(self, i):
        a, b, label = self.edges[i]
        return {
            'graph_a': self._get_graph(a),
            'graph_b': self._get_graph(b),
            'label': label,
        }


def collate_fn(batch):
    """Collate batch of molecule pairs."""
    def prepare_graphs(graph_list):
        atoms, edges, edge_types, batch_idx = [], [], [], []
        offset = 0
        
        for i, graph in enumerate(graph_list):
            if graph is None:
                graph = {'atom_types': [6], 'edge_index': [[0, 0]], 'edge_types': [0]}
            
            n = len(graph['atom_types'])
            atoms.extend(graph['atom_types'])
            for e in graph['edge_index']:
                edges.append([e[0] + offset, e[1] + offset])
            edge_types.extend(graph['edge_types'])
            batch_idx.extend([i] * n)
            offset += n
        
        return {
            'atoms': torch.tensor(atoms, dtype=torch.long),
            'edges': torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long),
            'edge_types': torch.tensor(edge_types, dtype=torch.long),
            'batch': torch.tensor(batch_idx, dtype=torch.long)
        }
    
    graphs_a = [item['graph_a'] for item in batch]
    graphs_b = [item['graph_b'] for item in batch]
    labels = [item['label'] for item in batch]
    
    return {
        'mol_a': prepare_graphs(graphs_a),
        'mol_b': prepare_graphs(graphs_b),
        'labels': torch.tensor(labels, dtype=torch.float),
    }


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_contrastive = 0
    total_bce = 0
    n_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        # Move to device
        for key in ['mol_a', 'mol_b']:
            for k in batch[key]:
                batch[key][k] = batch[key][k].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            batch['mol_a']['atoms'], batch['mol_a']['edges'],
            batch['mol_a']['edge_types'], batch['mol_a']['batch'],
            batch['mol_b']['atoms'], batch['mol_b']['edges'],
            batch['mol_b']['edge_types'], batch['mol_b']['batch'],
        )
        
        # BCE loss on pair scores
        bce_loss = F.binary_cross_entropy_with_logits(outputs['score'], labels)
        
        # Contrastive loss on embeddings
        emb_a = F.normalize(outputs['link_emb_a'], dim=-1)
        emb_b = F.normalize(outputs['link_emb_b'], dim=-1)
        
        # In-batch negatives: positive pairs should be similar, negative pairs should be dissimilar
        sim = torch.sum(emb_a * emb_b, dim=-1) / config.temperature
        contrastive_loss = F.binary_cross_entropy_with_logits(
            sim, labels, reduction='mean'
        )
        
        loss = bce_loss + 0.5 * contrastive_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_contrastive += contrastive_loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'bce': total_bce / max(n_batches, 1),
        'contrastive': total_contrastive / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on link prediction."""
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        for key in ['mol_a', 'mol_b']:
            for k in batch[key]:
                batch[key][k] = batch[key][k].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            batch['mol_a']['atoms'], batch['mol_a']['edges'],
            batch['mol_a']['edge_types'], batch['mol_a']['batch'],
            batch['mol_b']['atoms'], batch['mol_b']['edges'],
            batch['mol_b']['edge_types'], batch['mol_b']['batch'],
        )
        
        loss = F.binary_cross_entropy_with_logits(outputs['score'], labels)
        total_loss += loss.item()
        n_batches += 1
        
        probs = torch.sigmoid(outputs['score']).cpu().numpy()
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.tolist())
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'roc_auc': roc_auc,
        'ap': ap,
        'accuracy': acc,
        'f1': f1,
    }


# =============================================================================
# Main training pipeline
# =============================================================================

def train_hypergraph_link_predictor(
    data_dir="Data/uspto",
    n_epochs=30,
    batch_size=128,
    lr=3e-4,
    model_size='medium',
    device='auto',
    max_train_edges=None,
    max_test_edges=None,
    num_workers=0,
    save_dir='model_reactions/checkpoints',
):
    """Train the hypergraph link predictor."""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Build reaction database
    cache_path = os.path.join(data_dir, 'reaction_db_cache.pkl')
    db = ReactionDatabase(data_dir)
    if os.path.exists(cache_path):
        print("Loading cached reaction database...")
        db.load(cache_path)
    else:
        db.build()
        db.save(cache_path)
    
    # Get train/val/test edges
    n_mols = len(db.reactant_smiles)
    
    train_edges = []
    val_edges = []
    test_edges = []
    
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
    
    print(f"Edges: train={len(train_edges)}, val={len(val_edges)}, test={len(test_edges)}")
    
    # Subsample if needed
    if max_train_edges and len(train_edges) > max_train_edges:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_edges), max_train_edges, replace=False)
        train_edges = [train_edges[i] for i in idx]
        print(f"  Subsampled train: {len(train_edges)}")
    
    if max_test_edges and len(test_edges) > max_test_edges:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(test_edges), max_test_edges, replace=False)
        test_edges = [test_edges[i] for i in idx]
    
    if max_test_edges and len(val_edges) > max_test_edges:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(val_edges), max_test_edges, replace=False)
        val_edges = [val_edges[i] for i in idx]
    
    # Sample negatives
    all_pos = set(train_edges + val_edges + test_edges)
    
    def sample_negatives(n, seed=42):
        rng = np.random.RandomState(seed)
        negs = []
        seen = set()
        attempts = 0
        while len(negs) < n and attempts < n * 20:
            a = rng.randint(0, n_mols)
            b = rng.randint(0, n_mols)
            if a == b:
                attempts += 1
                continue
            edge = (min(a, b), max(a, b))
            if edge not in all_pos and edge not in seen:
                negs.append(edge)
                seen.add(edge)
            attempts += 1
        return negs
    
    train_neg = sample_negatives(len(train_edges), seed=42)
    val_neg = sample_negatives(len(val_edges), seed=43)
    test_neg = sample_negatives(len(test_edges), seed=44)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ReactionLinkDataset(train_edges, train_neg, db.reactant_smiles)
    val_dataset = ReactionLinkDataset(val_edges, val_neg, db.reactant_smiles)
    test_dataset = ReactionLinkDataset(test_edges, test_neg, db.reactant_smiles)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)
    
    # Model
    config = getattr(HypergraphLinkConfig, model_size)()
    model = HypergraphLinkPredictor(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_size}, Parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    
    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    best_val_auc = 0
    best_model_state = None
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 80)
    
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        elapsed = time.time() - t0
        
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"Train: loss={train_metrics['loss']:.4f} (bce={train_metrics['bce']:.3f}, ctr={train_metrics['contrastive']:.3f}) | "
              f"Val: AUC={val_metrics['roc_auc']:.4f}, AP={val_metrics['ap']:.4f}, Acc={val_metrics['accuracy']:.4f} | "
              f"lr={lr_current:.2e} | {elapsed:.1f}s")
        
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'config': config,
                'val_auc': best_val_auc,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, 'hypergraph_link_best.pt'))
            print(f"  -> Saved best model (val AUC: {best_val_auc:.4f})")
    
    # Load best model and evaluate on test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Hypergraph Link Predictor Results")
    print(f"{'='*60}")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Save final checkpoint with test metrics
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'test_metrics': test_metrics,
        'val_auc': best_val_auc,
    }, os.path.join(save_dir, 'hypergraph_link_final.pt'))
    
    return test_metrics


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypergraph Link Predictor')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark only')
    parser.add_argument('--data_dir', type=str, default='Data/uspto')
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model_size', type=str, default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_train_edges', type=int, default=None)
    parser.add_argument('--max_test_edges', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='model_reactions/checkpoints')
    args = parser.parse_args()
    
    if args.train or args.benchmark:
        train_hypergraph_link_predictor(
            data_dir=args.data_dir,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_size=args.model_size,
            device=args.device,
            max_train_edges=args.max_train_edges,
            max_test_edges=args.max_test_edges,
            num_workers=args.num_workers,
            save_dir=args.save_dir,
        )
    else:
        print("Use --train to train model, or --benchmark to run benchmark")
