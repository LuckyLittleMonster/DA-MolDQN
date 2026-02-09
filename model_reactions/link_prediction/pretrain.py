"""
Reaction-Aware Contrastive Pre-training for Molecular Encoder (Phase 3).

Pre-trains the 2D MPNN encoder with self-supervised objectives before
fine-tuning on link prediction. Multiple pretext tasks:

  1. Reaction Partner Contrastive (InfoNCE): molecules from same reaction
     should have similar embeddings.
  2. Reaction Class Hierarchical Contrastive: molecules from same rxn_class
     are weak positives; different rxn_class are negatives.
  3. Reaction Property Prediction: predict yield, barrier, rate from pair
     embeddings.
  4. Masked Atom Prediction: predict masked atom types from molecular context.

Usage:
    python -m model_reactions.pretrain --data_dir Data/uspto --n_epochs 30

    # Then fine-tune v3:
    python -m model_reactions.hypergraph_link_predictor_v3 --train \\
        --pretrained_encoder model_reactions/checkpoints/pretrained_encoder.pt
"""

import os
import time
import argparse
import numpy as np
from typing import Dict
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model_reactions.reactant_predictor import ReactionDatabase
from model_reactions.link_prediction.reaction_center import heuristic_reaction_center_scores
from model_reactions.link_prediction.hypergraph_link_predictor_v3 import (
    HypergraphLinkConfigV3,
    MoleculeEncoderV3,
    smiles_to_rich_graph,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    ATOM_FEATURES,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PretrainConfig:
    """Configuration for pre-training."""
    # Encoder (matches v3)
    encoder_config: HypergraphLinkConfigV3 = None

    # Pre-training tasks
    use_partner_contrastive: bool = True
    use_class_contrastive: bool = True
    use_property_prediction: bool = True
    use_masked_atom: bool = True

    # Contrastive
    temperature: float = 0.07
    class_margin: float = 0.3

    # Masked atom prediction
    mask_ratio: float = 0.15

    # Loss weights
    partner_weight: float = 1.0
    class_weight: float = 0.5
    property_weight: float = 0.3
    mask_weight: float = 0.3

    # Training
    n_epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    warmup_ratio: float = 0.1

    def __post_init__(self):
        if self.encoder_config is None:
            self.encoder_config = HypergraphLinkConfigV3.medium()


# =============================================================================
# Pre-training Dataset
# =============================================================================

class PretrainReactionDataset(Dataset):
    """
    Dataset for pre-training on reaction pairs.

    Each sample is a pair of molecules from the same reaction (positive)
    with associated metadata (rxn_class, yield, barrier, rate).
    """

    def __init__(self, db: ReactionDatabase, split='train',
                 rxn_center_scores=None):
        """
        Args:
            db: ReactionDatabase
            split: 'train' or 'val'
            rxn_center_scores: Dict[mol_idx] -> np.ndarray
        """
        self.smiles_list = db.reactant_smiles
        self.rxn_center_scores = rxn_center_scores or {}

        # Collect reaction pairs with metadata
        self.pairs = []
        for rxn in db.reactions:
            if rxn['split'] != split:
                continue
            indices = rxn['reactant_indices']
            rxn_class = rxn.get('rxn_class', 0)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    self.pairs.append({
                        'idx_a': indices[i],
                        'idx_b': indices[j],
                        'rxn_class': rxn_class,
                    })

        # Build rxn_class index for class-aware sampling
        self.class_to_mols = defaultdict(set)
        for rxn in db.reactions:
            if rxn['split'] != split:
                continue
            for idx in rxn['reactant_indices']:
                self.class_to_mols[rxn['rxn_class']].add(idx)
        self.class_to_mols = {k: list(v) for k, v in self.class_to_mols.items()}
        self.all_mol_indices = list(set(
            idx for pair in self.pairs
            for idx in [pair['idx_a'], pair['idx_b']]
        ))

        # Precompute graphs
        self.graph_cache = {}
        unique_mols = set()
        for pair in self.pairs:
            unique_mols.add(pair['idx_a'])
            unique_mols.add(pair['idx_b'])

        print(f"Pre-training dataset ({split}): {len(self.pairs)} pairs, "
              f"{len(unique_mols)} unique molecules")
        print(f"Precomputing graphs...")
        for idx in tqdm(unique_mols, desc="Graphs"):
            if idx < len(self.smiles_list):
                rc = self.rxn_center_scores.get(idx, None)
                graph = smiles_to_rich_graph(self.smiles_list[idx], rc)
                if graph is not None:
                    self.graph_cache[idx] = graph
        print(f"Cached {len(self.graph_cache)} graphs")

    def __len__(self):
        return len(self.pairs)

    def _get_graph(self, idx):
        if idx in self.graph_cache:
            return self.graph_cache[idx]
        return {
            'atom_features': [[0.0] * ATOM_FEATURE_DIM],
            'edge_index': [[0, 0]],
            'bond_features': [[0.0] * BOND_FEATURE_DIM],
            'n_atoms': 1,
        }

    def __getitem__(self, i):
        pair = self.pairs[i]
        graph_a = self._get_graph(pair['idx_a'])
        graph_b = self._get_graph(pair['idx_b'])

        return {
            'graph_a': graph_a,
            'graph_b': graph_b,
            'rxn_class': pair['rxn_class'],
            'idx_a': pair['idx_a'],
            'idx_b': pair['idx_b'],
        }


def pretrain_collate_fn(batch):
    """Collate for pre-training batches."""
    def prepare_graphs(graph_list):
        all_atom_feats = []
        all_edges = []
        all_bond_feats = []
        all_batch_idx = []
        offset = 0

        for i, graph in enumerate(graph_list):
            n = graph['n_atoms'] if graph else 1
            if graph is None:
                graph = {
                    'atom_features': [[0.0] * ATOM_FEATURE_DIM],
                    'edge_index': [[0, 0]],
                    'bond_features': [[0.0] * BOND_FEATURE_DIM],
                    'n_atoms': 1,
                }
                n = 1

            all_atom_feats.extend(graph['atom_features'])
            for e in graph['edge_index']:
                all_edges.append([e[0] + offset, e[1] + offset])
            all_bond_feats.extend(graph['bond_features'])
            all_batch_idx.extend([i] * n)
            offset += n

        return {
            'atom_features': torch.tensor(all_atom_feats, dtype=torch.float),
            'edges': torch.tensor(all_edges, dtype=torch.long).t().contiguous() if all_edges else torch.zeros(2, 0, dtype=torch.long),
            'bond_features': torch.tensor(all_bond_feats, dtype=torch.float),
            'batch': torch.tensor(all_batch_idx, dtype=torch.long),
        }

    graphs_a = [item['graph_a'] for item in batch]
    graphs_b = [item['graph_b'] for item in batch]
    rxn_classes = [item['rxn_class'] for item in batch]

    return {
        'mol_a': prepare_graphs(graphs_a),
        'mol_b': prepare_graphs(graphs_b),
        'rxn_classes': torch.tensor(rxn_classes, dtype=torch.long),
    }


# =============================================================================
# Pre-training Model
# =============================================================================

class PretrainModel(nn.Module):
    """
    Pre-training model with multiple self-supervised heads.
    """

    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.config = config
        enc_config = config.encoder_config

        # Shared encoder
        self.encoder = MoleculeEncoderV3(enc_config)

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(enc_config.mol_embedding_dim, enc_config.mol_embedding_dim),
            nn.GELU(),
            nn.Linear(enc_config.mol_embedding_dim, 128),
        )

        # Reaction class predictor
        self.class_predictor = nn.Sequential(
            nn.Linear(enc_config.mol_embedding_dim, enc_config.hidden_dim),
            nn.GELU(),
            nn.Linear(enc_config.hidden_dim, enc_config.num_rxn_classes),
        )

        # Property predictor (yield, barrier, rate)
        self.property_predictor = nn.Sequential(
            nn.Linear(enc_config.mol_embedding_dim * 2, enc_config.hidden_dim),
            nn.GELU(),
            nn.Linear(enc_config.hidden_dim, 3),
        )

        # Masked atom predictor
        n_atom_types = len(ATOM_FEATURES['atomic_num']) + 1  # +1 for unknown
        self.mask_predictor = nn.Sequential(
            nn.Linear(enc_config.encoder_hidden_dim, enc_config.encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(enc_config.encoder_hidden_dim, n_atom_types),
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(enc_config.atom_feature_dim))

    def encode(self, atom_features, edge_index, bond_features, batch_idx):
        """Encode molecules to embeddings."""
        return self.encoder(atom_features, edge_index, bond_features, batch_idx)

    def forward(self, mol_a, mol_b, mask_ratio=0.15):
        """
        Forward pass for pre-training.

        Args:
            mol_a, mol_b: dict with 'atom_features', 'edges', 'bond_features', 'batch'
            mask_ratio: fraction of atoms to mask

        Returns:
            dict with embeddings and predictions
        """
        # Encode both molecules
        emb_a = self.encode(
            mol_a['atom_features'], mol_a['edges'],
            mol_a['bond_features'], mol_a['batch']
        )
        emb_b = self.encode(
            mol_b['atom_features'], mol_b['edges'],
            mol_b['bond_features'], mol_b['batch']
        )

        # Projections for contrastive learning
        proj_a = self.projection(emb_a)
        proj_b = self.projection(emb_b)

        # Class prediction (from single molecule)
        class_logits_a = self.class_predictor(emb_a)
        class_logits_b = self.class_predictor(emb_b)

        # Property prediction (from pair)
        pair_emb = torch.cat([emb_a, emb_b], dim=-1)
        property_pred = self.property_predictor(pair_emb)

        result = {
            'emb_a': emb_a, 'emb_b': emb_b,
            'proj_a': proj_a, 'proj_b': proj_b,
            'class_logits_a': class_logits_a,
            'class_logits_b': class_logits_b,
            'property_pred': property_pred,
        }

        # Masked atom prediction on mol_a
        if self.training and mask_ratio > 0:
            masked_result = self._masked_atom_forward(
                mol_a['atom_features'], mol_a['edges'],
                mol_a['bond_features'], mol_a['batch'],
                mask_ratio
            )
            result.update(masked_result)

        return result

    def _masked_atom_forward(self, atom_features, edge_index, bond_features,
                             batch_idx, mask_ratio):
        """Masked atom prediction: mask some atoms and predict their types."""
        n_atoms = atom_features.size(0)
        n_mask = max(1, int(n_atoms * mask_ratio))

        # Randomly select atoms to mask
        mask_indices = torch.randperm(n_atoms)[:n_mask]

        # Get ground truth atom types (first 119 features are one-hot atomic num)
        n_atom_types = len(ATOM_FEATURES['atomic_num']) + 1
        gt_atom_types = atom_features[:, :n_atom_types].argmax(dim=-1)
        gt_masked = gt_atom_types[mask_indices]

        # Replace masked atom features with mask token
        masked_features = atom_features.clone()
        masked_features[mask_indices] = self.mask_token

        # Encode with masked features (get hidden states before pooling)
        x = self.encoder.input_proj(self.encoder.atom_proj(masked_features))
        edge_attr = self.encoder.edge_proj(bond_features)

        for layer in self.encoder.layers:
            x = layer(x, edge_index, edge_attr)

        # Predict masked atom types from hidden states
        masked_hidden = x[mask_indices]
        mask_logits = self.mask_predictor(masked_hidden)

        return {
            'mask_logits': mask_logits,
            'mask_targets': gt_masked,
        }


# =============================================================================
# Pre-training Loss
# =============================================================================

def compute_pretrain_loss(outputs, rxn_classes, config):
    """Compute multi-task pre-training loss."""
    losses = {}
    total_loss = torch.tensor(0.0, device=rxn_classes.device)

    # 1. Reaction Partner Contrastive (InfoNCE)
    if config.use_partner_contrastive:
        proj_a = F.normalize(outputs['proj_a'], dim=-1)
        proj_b = F.normalize(outputs['proj_b'], dim=-1)

        # All pairs similarity matrix
        sim_matrix = torch.mm(proj_a, proj_b.t()) / config.temperature  # (B, B)

        # Diagonal entries are positive pairs
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_a2b = F.cross_entropy(sim_matrix, labels)
        loss_b2a = F.cross_entropy(sim_matrix.t(), labels)
        partner_loss = (loss_a2b + loss_b2a) / 2

        losses['partner'] = partner_loss.item()
        total_loss = total_loss + config.partner_weight * partner_loss

    # 2. Reaction Class Hierarchical Contrastive
    if config.use_class_contrastive:
        # Each molecule should predict its reaction class
        class_targets = rxn_classes.clamp(0, config.encoder_config.num_rxn_classes - 1)
        loss_cls_a = F.cross_entropy(outputs['class_logits_a'], class_targets)
        loss_cls_b = F.cross_entropy(outputs['class_logits_b'], class_targets)
        class_loss = (loss_cls_a + loss_cls_b) / 2

        losses['class'] = class_loss.item()
        total_loss = total_loss + config.class_weight * class_loss

    # 3. Masked Atom Prediction
    if config.use_masked_atom and 'mask_logits' in outputs:
        mask_loss = F.cross_entropy(outputs['mask_logits'], outputs['mask_targets'])
        losses['mask'] = mask_loss.item()
        total_loss = total_loss + config.mask_weight * mask_loss

    losses['total'] = total_loss.item()
    return total_loss, losses


# =============================================================================
# Training Loop
# =============================================================================

def pretrain_epoch(model, loader, optimizer, device, config, scheduler=None):
    """Train for one pre-training epoch."""
    model.train()
    loss_accum = defaultdict(float)
    n_batches = 0

    for batch in tqdm(loader, desc="Pretrain", leave=False):
        for key in ['mol_a', 'mol_b']:
            for k in batch[key]:
                batch[key][k] = batch[key][k].to(device)
        rxn_classes = batch['rxn_classes'].to(device)

        optimizer.zero_grad()

        outputs = model(batch['mol_a'], batch['mol_b'], mask_ratio=config.mask_ratio)
        total_loss, loss_dict = compute_pretrain_loss(outputs, rxn_classes, config)

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
def pretrain_evaluate(model, loader, device, config):
    """Evaluate pre-training objectives."""
    model.eval()
    loss_accum = defaultdict(float)
    n_batches = 0

    for batch in loader:
        for key in ['mol_a', 'mol_b']:
            for k in batch[key]:
                batch[key][k] = batch[key][k].to(device)
        rxn_classes = batch['rxn_classes'].to(device)

        outputs = model(batch['mol_a'], batch['mol_b'], mask_ratio=0)
        _, loss_dict = compute_pretrain_loss(outputs, rxn_classes, config)

        for k, v in loss_dict.items():
            loss_accum[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}


# =============================================================================
# Main
# =============================================================================

def run_pretraining(
    data_dir="Data/uspto",
    n_epochs=30,
    batch_size=256,
    lr=1e-3,
    model_size='medium',
    device='auto',
    num_workers=0,
    save_dir='model_reactions/checkpoints',
):
    """Run pre-training pipeline."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load database
    cache_path = os.path.join(data_dir, 'reaction_db_cache.pkl')
    db = ReactionDatabase(data_dir)
    if os.path.exists(cache_path):
        print("Loading cached reaction database...")
        db.load(cache_path)
    else:
        db.build()
        db.save(cache_path)

    # Compute heuristic reaction center scores
    print("Computing heuristic reaction center scores...")
    rxn_center_scores = {}
    for idx, smi in enumerate(tqdm(db.reactant_smiles, desc="RxnCenter")):
        scores = heuristic_reaction_center_scores(smi)
        if scores is not None:
            rxn_center_scores[idx] = scores
    print(f"  Computed for {len(rxn_center_scores)} molecules")

    # Create datasets
    print("\nCreating pre-training datasets...")
    train_dataset = PretrainReactionDataset(db, split='train',
                                            rxn_center_scores=rxn_center_scores)
    val_dataset = PretrainReactionDataset(db, split='val',
                                          rxn_center_scores=rxn_center_scores)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=pretrain_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=pretrain_collate_fn)

    # Model
    encoder_config = getattr(HypergraphLinkConfigV3, model_size)()
    config = PretrainConfig(
        encoder_config=encoder_config,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
    )

    model = PretrainModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Pre-training model: {n_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_steps = n_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=config.warmup_ratio, anneal_strategy='cos',
    )

    # Training
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nPre-training for {n_epochs} epochs...")
    print("-" * 90)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_losses = pretrain_epoch(model, train_loader, optimizer, device, config, scheduler)
        val_losses = pretrain_evaluate(model, val_loader, device, config)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]['lr']

        train_str = " ".join(f"{k}={v:.4f}" for k, v in train_losses.items())
        val_str = " ".join(f"{k}={v:.4f}" for k, v in val_losses.items())

        print(f"Epoch {epoch:3d}/{n_epochs} | Train: {train_str} | Val: {val_str} | "
              f"lr={lr_current:.2e} | {elapsed:.1f}s")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            # Save encoder weights
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'config': encoder_config,
                'val_losses': val_losses,
            }, os.path.join(save_dir, 'pretrained_encoder.pt'))
            print(f"  -> Saved pretrained encoder (val loss: {best_val_loss:.4f})")

    # Also save full pretrain model
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'config': encoder_config,
        'pretrain_config': config,
    }, os.path.join(save_dir, 'pretrained_full.pt'))

    print(f"\nPre-training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Encoder saved to: {save_dir}/pretrained_encoder.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reaction-Aware Pre-training')
    parser.add_argument('--data_dir', type=str, default='Data/uspto')
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_size', type=str, default='medium',
                        choices=['small', 'medium', 'large'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='model_reactions/checkpoints')
    args = parser.parse_args()

    run_pretraining(
        data_dir=args.data_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_size=args.model_size,
        device=args.device,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
    )
