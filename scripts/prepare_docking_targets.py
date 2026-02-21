#!/usr/bin/env python3
"""Download and prepare docking target proteins from RCSB PDB.

For each target:
1. Download PDB from RCSB
2. Separate protein from co-crystallized ligand
3. Remove water, add hydrogens
4. Convert protein to PDBQT (using unidock_tools pure-RDKit converter)
5. Compute binding box center from ligand centroid
6. Save config.json with docking box parameters

Targets:
- sEH (PDB: 1ZD5) — soluble Epoxide Hydrolase, standard GFlowNet benchmark
- DRD2 (PDB: 3PBL) — Dopamine D2 Receptor, PMO benchmark
- GSK3β (PDB: 1UV5) — Glycogen Synthase Kinase 3 beta, PMO benchmark
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'refs' / 'Uni-Dock' / 'unidock_tools' / 'src'))

from unidock_tools.modules.protein_prep.pdb2pdbqt import pdb2pdbqt

# ─── Target definitions ───────────────────────────────────────────────
TARGETS = {
    'seh': {
        'pdb_id': '1ZD5',
        'name': 'soluble Epoxide Hydrolase',
        'ligand_resname': 'NC7',  # 4-cyano-N-[(1S)-1-(4-fluorophenyl)-2-... in 1ZD5
        'box_size': (22.5, 22.5, 22.5),
        'box_padding': 4.0,  # Å padding around ligand
    },
    'drd2': {
        'pdb_id': '3PBL',
        'name': 'Dopamine D2 Receptor',
        'ligand_resname': 'ETQ',  # Eticlopride in 3PBL
        'box_size': (22.5, 22.5, 22.5),
        'box_padding': 4.0,
    },
    'gsk3b': {
        'pdb_id': '1UV5',
        'name': 'Glycogen Synthase Kinase 3 beta',
        'ligand_resname': 'BRW',  # BRW ligand in 1UV5
        'box_size': (22.5, 22.5, 22.5),
        'box_padding': 4.0,
    },
}

TARGETS_DIR = PROJECT_ROOT / 'Data' / 'docking_targets'


def download_pdb(pdb_id: str, out_path: Path) -> Path:
    """Download PDB file from RCSB."""
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    pdb_file = out_path / f'{pdb_id}.pdb'
    if pdb_file.exists():
        print(f"  PDB {pdb_id} already downloaded")
        return pdb_file
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, str(pdb_file))
    print(f"  Saved to {pdb_file}")
    return pdb_file


def extract_protein_and_ligand(pdb_file: Path, ligand_resname: str, out_dir: Path):
    """Extract protein (no water/ligand) and ligand coordinates from PDB.

    Returns (protein_pdb_path, ligand_center_xyz).
    """
    protein_lines = []
    ligand_coords = []

    with open(pdb_file) as f:
        for line in f:
            record = line[:6].strip()
            if record in ('ATOM', 'HETATM'):
                resname = line[17:20].strip()
                if resname == 'HOH':
                    continue  # skip water
                if resname == ligand_resname:
                    # Extract ligand atom coordinates
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        ligand_coords.append([x, y, z])
                    except ValueError:
                        pass
                    continue  # don't include ligand in protein
                protein_lines.append(line)
            elif record in ('TER', 'END', 'MODEL', 'ENDMDL'):
                protein_lines.append(line)

    # Write protein-only PDB
    protein_pdb = out_dir / 'protein_clean.pdb'
    with open(protein_pdb, 'w') as f:
        f.writelines(protein_lines)

    # Compute ligand centroid
    if ligand_coords:
        center = np.mean(ligand_coords, axis=0)
        print(f"  Ligand {ligand_resname}: {len(ligand_coords)} atoms, "
              f"center = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    else:
        center = np.array([0.0, 0.0, 0.0])
        print(f"  WARNING: Ligand {ligand_resname} not found in {pdb_file.name}!")

    return protein_pdb, center


def convert_pdb_to_pdbqt(pdb_file: Path, pdbqt_file: Path):
    """Convert protein PDB to PDBQT using unidock_tools RDKit-based converter."""
    print(f"  Converting to PDBQT ...")

    # Try direct conversion
    try:
        pdb2pdbqt(str(pdb_file), str(pdbqt_file))
        # Verify output
        n_lines = sum(1 for line in open(pdbqt_file)
                      if line.startswith('ATOM'))
        print(f"  PDBQT: {n_lines} atoms written to {pdbqt_file.name}")
        return True
    except Exception as e:
        print(f"  Direct conversion failed: {e}")

    # Fallback: load with sanitize=False, add Hs manually
    try:
        print(f"  Trying fallback (sanitize=False) ...")
        mol = Chem.MolFromPDBFile(str(pdb_file), removeHs=False, sanitize=False)
        if mol is None:
            print(f"  ERROR: RDKit cannot read {pdb_file}")
            return False

        # Try partial sanitization
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        except Exception:
            pass

        # Compute charges
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass

        # Write PDBQT manually
        _write_pdbqt_fallback(mol, pdbqt_file)
        n_lines = sum(1 for line in open(pdbqt_file)
                      if line.startswith('ATOM'))
        print(f"  PDBQT (fallback): {n_lines} atoms written")
        return True
    except Exception as e2:
        print(f"  Fallback also failed: {e2}")
        return False


def _write_pdbqt_fallback(mol, pdbqt_file: Path):
    """Minimal PDBQT writer when standard converter fails."""
    from unidock_tools.modules.protein_prep.pdb2pdbqt import (
        get_pdbqt_atom_lines, receptor_mol_to_pdbqt_str
    )
    pdbqt_str = receptor_mol_to_pdbqt_str(mol)
    with open(pdbqt_file, 'w') as f:
        f.write(pdbqt_str)


def prepare_target(target_key: str, target_info: dict):
    """Full pipeline for one target."""
    print(f"\n{'='*60}")
    print(f"Preparing {target_key} ({target_info['name']})")
    print(f"PDB: {target_info['pdb_id']}")
    print(f"{'='*60}")

    out_dir = TARGETS_DIR / target_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download PDB
    pdb_file = download_pdb(target_info['pdb_id'], out_dir)

    # 2. Extract protein and ligand center
    protein_pdb, ligand_center = extract_protein_and_ligand(
        pdb_file, target_info['ligand_resname'], out_dir)

    # 3. Convert to PDBQT
    pdbqt_file = out_dir / 'receptor.pdbqt'
    success = convert_pdb_to_pdbqt(protein_pdb, pdbqt_file)

    if not success:
        print(f"  FAILED to create PDBQT for {target_key}")
        return False

    # 4. Save config with docking box
    config = {
        'pdb_id': target_info['pdb_id'],
        'name': target_info['name'],
        'ligand_resname': target_info['ligand_resname'],
        'center_x': round(float(ligand_center[0]), 3),
        'center_y': round(float(ligand_center[1]), 3),
        'center_z': round(float(ligand_center[2]), 3),
        'size_x': target_info['box_size'][0],
        'size_y': target_info['box_size'][1],
        'size_z': target_info['box_size'][2],
    }
    config_file = out_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_file}")
    print(f"  Box center: ({config['center_x']}, {config['center_y']}, {config['center_z']})")
    print(f"  Box size: ({config['size_x']}, {config['size_y']}, {config['size_z']})")

    return True


def main():
    print("Preparing docking targets for DA-MolDQN")
    print(f"Output directory: {TARGETS_DIR}")

    results = {}
    for key, info in TARGETS.items():
        success = prepare_target(key, info)
        results[key] = success

    print(f"\n{'='*60}")
    print("Summary:")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {key}: {status}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
