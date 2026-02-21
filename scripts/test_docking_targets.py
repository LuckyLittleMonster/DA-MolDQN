#!/usr/bin/env python3
"""Test docking on all prepared targets.

Tests UniDockScorer.batch_dock() on a set of test molecules for each target.
Verifies that scores are reasonable (negative kcal/mol for drug-like molecules).
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from reward.docking_score.unidock import UniDockScorer

TARGETS_DIR = PROJECT_ROOT / 'Data' / 'docking_targets'

# Test molecules: known drugs + simple molecules
TEST_MOLECULES = {
    'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'Ibuprofen': 'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
    'Caffeine': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
    'Celecoxib': 'Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1',
    'Benzene': 'c1ccccc1',
    'Ethanol': 'CCO',
    'Aniline': 'c1ccc(N)cc1',
    'Isoproterenol': 'CC(C)NCC(O)c1ccc(O)c(O)c1',
}


def test_target(target_name: str):
    """Test docking on a single target."""
    target_dir = TARGETS_DIR / target_name
    config_file = target_dir / 'config.json'
    pdbqt_file = target_dir / 'receptor.pdbqt'

    if not config_file.exists() or not pdbqt_file.exists():
        print(f"  SKIP: {target_name} (files missing)")
        return False

    with open(config_file) as f:
        config = json.load(f)

    print(f"\n{'='*60}")
    print(f"Target: {target_name} ({config.get('name', 'unknown')})")
    print(f"PDB: {config.get('pdb_id', 'unknown')}")
    print(f"Box center: ({config['center_x']}, {config['center_y']}, {config['center_z']})")
    print(f"{'='*60}")

    scorer = UniDockScorer(
        receptor_pdbqt=str(pdbqt_file),
        center_x=config['center_x'],
        center_y=config['center_y'],
        center_z=config['center_z'],
        size_x=config.get('size_x', 22.5),
        size_y=config.get('size_y', 22.5),
        size_z=config.get('size_z', 22.5),
        search_mode='fast',
        num_modes=1,
        verbosity=0,
    )

    smiles_list = list(TEST_MOLECULES.values())
    names = list(TEST_MOLECULES.keys())

    t0 = time.perf_counter()
    scores = scorer.batch_dock(smiles_list)
    elapsed = time.perf_counter() - t0

    print(f"\nResults ({elapsed:.1f}s for {len(smiles_list)} molecules):")
    print(f"{'Name':>15s}  {'Score (kcal/mol)':>16s}  {'Status':>8s}")
    print(f"{'-'*15}  {'-'*16}  {'-'*8}")

    n_ok = 0
    for name, score in zip(names, scores):
        if score < 0:
            status = 'OK'
            n_ok += 1
        elif score == 0.0:
            status = 'FAIL'
        else:
            status = 'WEAK'
            n_ok += 1
        print(f"{name:>15s}  {score:>16.2f}  {status:>8s}")

    print(f"\n  {n_ok}/{len(scores)} molecules docked successfully")
    print(f"  Best score: {min(scores):.2f} kcal/mol")
    print(f"  {scorer.timing_summary}")
    return n_ok > 0


def main():
    print("Testing docking on all prepared targets")
    print(f"Targets directory: {TARGETS_DIR}")

    targets = ['seh', 'drd2', 'gsk3b']
    results = {}
    for t in targets:
        try:
            results[t] = test_target(t)
        except Exception as e:
            print(f"  ERROR: {t}: {e}")
            results[t] = False

    print(f"\n{'='*60}")
    print("Summary:")
    for t, ok in results.items():
        print(f"  {t}: {'PASS' if ok else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
