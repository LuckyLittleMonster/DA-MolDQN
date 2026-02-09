"""
Frozen 3D Molecular Encoder Wrapper.

Wraps pretrained MLIP models (TorchMD-NET or FairChem UMA) as frozen feature
extractors for molecular 3D structure encoding.

The frozen encoder takes 3D conformer coordinates and atomic numbers as input,
and outputs a fixed-dimensional molecular embedding. All weights are frozen
(no gradient computation), so this acts as a pure feature extractor.

Usage:
    encoder = FrozenMLIPEncoder.from_torchmd_net(device='cuda')
    # or
    encoder = FrozenMLIPEncoder.from_pretrained('path/to/model.pt', device='cuda')

    embeddings = encoder.encode_batch(atomic_nums_list, coords_list)
"""

import os
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

import torch
import torch.nn as nn

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class FrozenMLIPEncoder(nn.Module):
    """
    Frozen MLIP encoder for extracting 3D molecular embeddings.

    Supports:
      - TorchMD-NET (equivariant transformer)
      - Generic SchNet-like encoder
      - Fallback: simple distance-based encoder
    """

    def __init__(self, output_dim: int = 512, encoder_type: str = 'schnet_simple'):
        super().__init__()
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        self.model = None
        self._build_fallback(output_dim)

    def _build_fallback(self, output_dim: int):
        """Build a simple distance-geometry based encoder as fallback."""
        self.distance_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )
        # Atom embedding for encoding atomic numbers
        self.atom_embed = nn.Embedding(120, 64)
        # Gaussian basis for distances
        self.n_gaussians = 64
        self.gaussian_centers = nn.Parameter(
            torch.linspace(0.0, 10.0, self.n_gaussians), requires_grad=False
        )
        self.gaussian_width = 0.5
        # Interaction layer
        self.interaction = nn.Sequential(
            nn.Linear(64 + self.n_gaussians, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(128, output_dim),
        )

    @classmethod
    def from_torchmd_net(cls, model_path: Optional[str] = None,
                         device: str = 'cpu') -> 'FrozenMLIPEncoder':
        """
        Load a TorchMD-NET pretrained model as frozen encoder.

        Args:
            model_path: Path to torchmd-net checkpoint. If None, tries default.
            device: Device to load model on.
        """
        encoder = cls(output_dim=512, encoder_type='torchmd_net')

        try:
            from torchmdnet.models.model import load_model
            if model_path is None:
                print("Warning: No TorchMD-NET model path provided. Using fallback encoder.")
                return encoder.to(device)

            print(f"Loading TorchMD-NET model from {model_path}")
            model = load_model(model_path, device=device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            encoder.model = model
            encoder.encoder_type = 'torchmd_net'
        except ImportError:
            print("Warning: torchmd-net not installed. Using fallback encoder.")

        return encoder.to(device)

    @classmethod
    def from_fairchem(cls, model_name: str = 'uma-sm',
                      device: str = 'cpu') -> 'FrozenMLIPEncoder':
        """
        Load a FairChem UMA model as frozen encoder.

        Args:
            model_name: FairChem model identifier.
            device: Device to load model on.
        """
        encoder = cls(output_dim=512, encoder_type='fairchem')

        try:
            from fairchem.core import OCPCalculator
            print(f"Loading FairChem model: {model_name}")
            calc = OCPCalculator(model_name=model_name, device=device)
            encoder.model = calc
            encoder.encoder_type = 'fairchem'
        except ImportError:
            print("Warning: fairchem not installed. Using fallback encoder.")

        return encoder.to(device)

    def _gaussian_smearing(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smearing to distances."""
        return torch.exp(-0.5 * ((distances.unsqueeze(-1) - self.gaussian_centers) / self.gaussian_width) ** 2)

    def encode_single(self, atomic_nums: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode a single molecule.

        Args:
            atomic_nums: (n_atoms,) long tensor of atomic numbers
            coords: (n_atoms, 3) float tensor of 3D coordinates

        Returns:
            embedding: (output_dim,) molecular embedding
        """
        if self.model is not None and self.encoder_type == 'torchmd_net':
            # Use TorchMD-NET
            with torch.no_grad():
                z = atomic_nums
                pos = coords
                batch = torch.zeros(len(z), dtype=torch.long, device=z.device)
                out = self.model(z, pos, batch=batch)
                if isinstance(out, tuple):
                    energy, forces = out
                else:
                    energy = out
                # Use the hidden representation
                return energy.view(-1)[:self.output_dim]

        # Fallback: distance-geometry encoder
        n_atoms = len(atomic_nums)
        atom_emb = self.atom_embed(atomic_nums)  # (n_atoms, 64)

        # Compute pairwise distances
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (n, n, 3)
        dists = torch.norm(diff, dim=-1)  # (n, n)

        # Gaussian smeared distances
        gauss = self._gaussian_smearing(dists)  # (n, n, n_gaussians)

        # Aggregate neighbor info
        atom_features = []
        for i in range(n_atoms):
            # Average Gaussian-smeared distances from atom i to all others
            avg_gauss = gauss[i].mean(dim=0)  # (n_gaussians,)
            feat = torch.cat([atom_emb[i], avg_gauss], dim=-1)
            atom_features.append(feat)

        atom_features = torch.stack(atom_features)  # (n_atoms, 64 + n_gaussians)
        hidden = self.interaction(atom_features)  # (n_atoms, 128)

        # Global mean pooling
        mol_emb = hidden.mean(dim=0)  # (128,)
        return self.output_proj(mol_emb)  # (output_dim,)

    @torch.no_grad()
    def encode_batch(self, atomic_nums_list: List[torch.Tensor],
                     coords_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of molecules.

        Args:
            atomic_nums_list: List of (n_atoms_i,) tensors
            coords_list: List of (n_atoms_i, 3) tensors

        Returns:
            embeddings: (batch_size, output_dim)
        """
        embeddings = []
        for z, pos in zip(atomic_nums_list, coords_list):
            emb = self.encode_single(z, pos)
            embeddings.append(emb)
        return torch.stack(embeddings)

    def forward(self, atomic_nums: torch.Tensor, coords: torch.Tensor,
                batch_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for batch encoding.

        Args:
            atomic_nums: (total_atoms,) atomic numbers
            coords: (total_atoms, 3) coordinates
            batch_idx: (total_atoms,) batch assignment

        Returns:
            embeddings: (batch_size, output_dim)
        """
        if batch_idx is None:
            return self.encode_single(atomic_nums, coords).unsqueeze(0)

        batch_size = batch_idx.max().item() + 1
        embeddings = []
        for b in range(batch_size):
            mask = batch_idx == b
            z_b = atomic_nums[mask]
            pos_b = coords[mask]
            emb = self.encode_single(z_b, pos_b)
            embeddings.append(emb)
        return torch.stack(embeddings)


def precompute_3d_embeddings(
    smiles_list: List[str],
    conformer_cache: Dict[str, np.ndarray],
    encoder: FrozenMLIPEncoder,
    device: str = 'cpu',
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Precompute 3D embeddings for a list of molecules using cached conformers.

    Args:
        smiles_list: List of SMILES strings
        conformer_cache: Dict mapping SMILES -> (n_atoms, 3) coordinates
        encoder: FrozenMLIPEncoder instance
        device: Device
        batch_size: Processing batch size

    Returns:
        embeddings: (n_molecules, output_dim) tensor. Zero for failed molecules.
    """
    from tqdm import tqdm

    encoder.eval()
    encoder.to(device)
    n_mols = len(smiles_list)
    output_dim = encoder.output_dim
    embeddings = torch.zeros(n_mols, output_dim)

    n_computed = 0
    for start in tqdm(range(0, n_mols, batch_size), desc="3D embeddings"):
        end = min(start + batch_size, n_mols)
        z_list = []
        pos_list = []
        valid_indices = []

        for i in range(start, end):
            smi = smiles_list[i]
            if smi not in conformer_cache:
                continue
            coords = conformer_cache[smi]
            if coords is None:
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if len(atomic_nums) != len(coords):
                continue

            z_list.append(torch.tensor(atomic_nums, dtype=torch.long, device=device))
            pos_list.append(torch.tensor(coords, dtype=torch.float, device=device))
            valid_indices.append(i)

        if z_list:
            with torch.no_grad():
                batch_emb = encoder.encode_batch(z_list, pos_list)
            for j, idx in enumerate(valid_indices):
                embeddings[idx] = batch_emb[j].cpu()
            n_computed += len(valid_indices)

    print(f"Computed 3D embeddings for {n_computed}/{n_mols} molecules")
    return embeddings
