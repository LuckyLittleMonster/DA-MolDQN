"""
Precompute 3D conformers and MLIP embeddings for all database molecules.

Uses nvmolkit for GPU-accelerated ETKDG conformer generation, with
RDKit CPU fallback.

Usage:
    python -m model_reactions.scripts.precompute_3d --data_dir Data/uspto \\
        --output_dir Data/precomputed/model_reactions

    # With MLIP embedding precomputation:
    python -m model_reactions.scripts.precompute_3d --data_dir Data/uspto \\
        --output_dir Data/precomputed/model_reactions --compute_embeddings
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict

import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from model_reactions.reactant_predictor import ReactionDatabase


def compute_conformer_rdkit(smiles: str) -> Optional[np.ndarray]:
    """
    Generate a single 3D conformer using RDKit ETKDG.

    Args:
        smiles: SMILES string

    Returns:
        coords: (n_atoms, 3) numpy array, or None on failure
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    try:
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True  # fallback for difficult molecules
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            return None

        # Optimize with MMFF
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

        # Remove Hs and get coordinates for heavy atoms only
        mol = Chem.RemoveHs(mol)
        conf = mol.GetConformer()
        coords = np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ], dtype=np.float32)

        return coords

    except Exception:
        return None


def compute_conformers_nvmolkit(smiles_list, batch_size=1000):
    """
    GPU-accelerated conformer generation using nvmolkit.

    Args:
        smiles_list: List of SMILES strings
        batch_size: Batch size for GPU processing

    Returns:
        Dict mapping SMILES -> (n_atoms, 3) numpy array
    """
    try:
        import nvmolkit.embedMolecules as nvembed
    except ImportError:
        print("nvmolkit not available. Falling back to RDKit CPU.")
        return None

    results = {}
    n_total = len(smiles_list)
    n_success = 0

    for start in tqdm(range(0, n_total, batch_size), desc="nvmolkit conformers"):
        end = min(start + batch_size, n_total)
        batch_smiles = smiles_list[start:end]

        # Prepare molecules
        mols = []
        valid_indices = []
        for i, smi in enumerate(batch_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mol = Chem.AddHs(mol)
                mols.append(mol)
                valid_indices.append(start + i)

        if not mols:
            continue

        try:
            # Run GPU ETKDG
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            embedded = nvembed(mols, params=params)

            for mol, idx in zip(embedded, valid_indices):
                if mol is None or mol.GetNumConformers() == 0:
                    continue
                mol_noH = Chem.RemoveHs(mol)
                if mol_noH.GetNumConformers() == 0:
                    continue
                conf = mol_noH.GetConformer()
                coords = np.array([
                    [conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(mol_noH.GetNumAtoms())
                ], dtype=np.float32)
                results[smiles_list[idx]] = coords
                n_success += 1
        except Exception as e:
            print(f"  nvmolkit batch failed: {e}, falling back to CPU for this batch")
            for idx in valid_indices:
                smi = smiles_list[idx]
                coords = compute_conformer_rdkit(smi)
                if coords is not None:
                    results[smi] = coords
                    n_success += 1

    print(f"Generated conformers: {n_success}/{n_total}")
    return results


def compute_all_conformers(smiles_list, use_gpu=True):
    """
    Compute conformers for all molecules, trying GPU first then CPU fallback.

    Args:
        smiles_list: List of SMILES strings
        use_gpu: Try GPU (nvmolkit) first

    Returns:
        Dict mapping SMILES -> (n_atoms, 3) numpy array
    """
    results = {}

    if use_gpu:
        gpu_results = compute_conformers_nvmolkit(smiles_list)
        if gpu_results is not None:
            results.update(gpu_results)
            # Find remaining molecules
            remaining = [s for s in smiles_list if s not in results]
            if remaining:
                print(f"CPU fallback for {len(remaining)} remaining molecules...")
                for smi in tqdm(remaining, desc="CPU conformers"):
                    coords = compute_conformer_rdkit(smi)
                    if coords is not None:
                        results[smi] = coords
            return results

    # Pure CPU path
    print("Computing conformers with RDKit CPU...")
    for smi in tqdm(smiles_list, desc="CPU conformers"):
        coords = compute_conformer_rdkit(smi)
        if coords is not None:
            results[smi] = coords

    print(f"Generated conformers: {len(results)}/{len(smiles_list)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Precompute 3D conformers")
    parser.add_argument('--data_dir', type=str, default='Data/uspto')
    parser.add_argument('--output_dir', type=str, default='Data/precomputed/model_reactions')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU conformer generation')
    parser.add_argument('--compute_embeddings', action='store_true',
                        help='Also compute MLIP embeddings')
    parser.add_argument('--mlip_model', type=str, default=None,
                        help='Path to MLIP model checkpoint')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load reaction database
    cache_path = os.path.join(args.data_dir, 'reaction_db_cache.pkl')
    db = ReactionDatabase(args.data_dir)
    if os.path.exists(cache_path):
        print("Loading cached reaction database...")
        db.load(cache_path)
    else:
        db.build()
        db.save(cache_path)

    smiles_list = db.reactant_smiles
    print(f"Total molecules: {len(smiles_list)}")

    # Compute conformers
    print("\n=== Computing 3D Conformers ===")
    t0 = time.time()
    conformers = compute_all_conformers(smiles_list, use_gpu=not args.no_gpu)
    elapsed = time.time() - t0
    print(f"Conformer generation: {elapsed:.1f}s, {len(conformers)}/{len(smiles_list)} success")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'conformers.pt')
    torch.save(conformers, output_path)
    print(f"Saved conformers to {output_path}")

    # Optionally compute MLIP embeddings
    if args.compute_embeddings:
        print("\n=== Computing MLIP Embeddings ===")
        from model_reactions.link_prediction.frozen_encoder import FrozenMLIPEncoder, precompute_3d_embeddings

        if args.mlip_model:
            encoder = FrozenMLIPEncoder.from_torchmd_net(args.mlip_model, device=device)
        else:
            print("No MLIP model specified, using fallback distance-geometry encoder")
            encoder = FrozenMLIPEncoder(output_dim=512).to(device)
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

        embeddings = precompute_3d_embeddings(
            smiles_list, conformers, encoder, device=device
        )

        emb_path = os.path.join(args.output_dir, 'mlip_embeddings.pt')
        torch.save(embeddings, emb_path)
        print(f"Saved MLIP embeddings to {emb_path}")
        print(f"Embedding shape: {embeddings.shape}")
        n_nonzero = (embeddings.abs().sum(dim=1) > 0).sum().item()
        print(f"Non-zero embeddings: {n_nonzero}/{len(smiles_list)}")


if __name__ == '__main__':
    main()
