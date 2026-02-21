#!/usr/bin/env python3
"""Test ADMET reward integration.

Tests:
1. compute_reward_admet() correctness on known molecules
2. Batch vs single-molecule consistency
3. FastADMETModel singleton pattern
4. Performance comparison: qed vs admet reward
5. Full reward dispatch via compute_reward(cfg_reward.name='admet')
"""

import sys
import os
import time
import pathlib

# Ensure project root on path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from omegaconf import OmegaConf


def test_admet_reward_basic():
    """Test compute_reward_admet on a few molecules."""
    from reward import compute_reward_admet, compute_reward

    cfg = OmegaConf.load(PROJECT_ROOT / 'configs' / 'reward' / 'admet.yaml')

    test_mols = [
        'c1ccc(N)cc1',        # aniline (simple)
        'CC(=O)Oc1ccccc1C(=O)O',  # aspirin
        'CC(C)NCC(O)c1ccc(O)c(O)c1',  # isoproterenol
        'CC12CCC3C(CCC4CC(=O)CCC34C)C1CCC2O',  # testosterone
        'INVALID_SMILES',
    ]

    print("=" * 60)
    print("Test 1: compute_reward_admet() on known molecules")
    print("=" * 60)

    for smi in test_mols:
        metrics = compute_reward_admet(
            smi, step=0, max_steps=5, gamma=0.9, cfg_reward=cfg)
        r = metrics['reward']
        if metrics.get('valid', False):
            print(f"\n  {smi}")
            print(f"    reward={r:.4f}, QED={metrics['qed']:.3f}, SA={metrics['sa']:.1f}")
            print(f"    HIA={metrics.get('HIA_Hou', 'N/A'):.3f}, "
                  f"hERG={metrics.get('hERG', 'N/A'):.3f}, "
                  f"AMES={metrics.get('AMES', 'N/A'):.3f}")
            print(f"    DILI={metrics.get('DILI', 'N/A'):.3f}, "
                  f"ClinTox={metrics.get('ClinTox', 'N/A'):.3f}, "
                  f"CYP3A4={metrics.get('CYP3A4_Veith', 'N/A'):.3f}")
            print(f"    Solubility={metrics.get('Solubility_AqSolDB', 'N/A'):.2f}")
        else:
            print(f"\n  {smi}: INVALID (reward={r})")

    # Test via unified dispatch
    print("\n\nTest 2: Unified dispatch (compute_reward with name='admet')")
    print("-" * 60)
    rdict = compute_reward(
        'CC(=O)Oc1ccccc1C(=O)O', step=0, max_steps=5, gamma=0.9,
        cfg_reward=cfg)
    r, qed, sa = rdict['reward'], rdict['qed'], rdict['sa']
    print(f"  Aspirin: reward={r:.4f}, qed={qed:.3f}, sa={sa:.1f}")
    assert r > 0, "Aspirin reward should be positive"


def test_admet_singleton():
    """Test that FastADMETModel is a singleton."""
    from reward.admet.reward import _get_admet_model

    print("\n" + "=" * 60)
    print("Test 3: FastADMETModel singleton")
    print("=" * 60)

    t0 = time.perf_counter()
    m1 = _get_admet_model()
    t1 = time.perf_counter()
    m2 = _get_admet_model()
    t2 = time.perf_counter()

    assert m1 is m2, "Should be same object (singleton)"
    print(f"  First call: {(t1-t0)*1000:.0f}ms (model loading)")
    print(f"  Second call: {(t2-t1)*1000:.3f}ms (cached)")
    print(f"  Singleton OK: {m1 is m2}")


def test_batch_consistency():
    """Test that batch ADMET predictions match single-molecule results."""
    from reward.admet.reward import _get_admet_model, compute_reward_admet
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(PROJECT_ROOT / 'configs' / 'reward' / 'admet.yaml')

    print("\n" + "=" * 60)
    print("Test 4: Batch vs single-molecule consistency")
    print("=" * 60)

    mols = [
        'c1ccc(N)cc1',
        'CC(=O)Oc1ccccc1C(=O)O',
        'CC(C)NCC(O)c1ccc(O)c(O)c1',
        'c1ccccc1',
        'CCO',
        'CC(=O)O',
        'c1ccc(C=O)cc1',
        'CCCC',
    ]

    admet_model = _get_admet_model()

    # Batch prediction
    batch_preds = admet_model.predict_properties(mols)

    # Single predictions
    max_diff = 0.0
    for i, smi in enumerate(mols):
        single_preds = admet_model.predict_properties(smi)
        batch_row = batch_preds.iloc[i].to_dict()

        for key in single_preds:
            diff = abs(single_preds[key] - batch_row[key])
            max_diff = max(max_diff, diff)

    print(f"  Max abs diff (batch vs single): {max_diff:.6f}")
    assert max_diff < 1e-5, f"Batch/single mismatch: {max_diff}"
    print("  PASS: batch and single predictions match")


def test_performance():
    """Compare QED-only vs ADMET reward performance."""
    from reward import compute_reward_qed
    from reward.admet.reward import compute_reward_admet, _get_admet_model
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(PROJECT_ROOT / 'configs' / 'reward' / 'admet.yaml')

    print("\n" + "=" * 60)
    print("Test 5: Performance comparison (QED vs ADMET reward)")
    print("=" * 60)

    mols = [
        'c1ccc(N)cc1', 'CC(=O)Oc1ccccc1C(=O)O',
        'CC(C)NCC(O)c1ccc(O)c(O)c1', 'c1ccccc1',
        'CCO', 'CC(=O)O', 'c1ccc(C=O)cc1', 'CCCC',
    ] * 8  # 64 molecules

    # Warm up
    _get_admet_model()

    # QED reward timing
    t0 = time.perf_counter()
    for smi in mols:
        compute_reward_qed(smi, 0, 5, 0.9)
    t_qed = time.perf_counter() - t0

    # ADMET reward timing (single-molecule calls, like ReaSyn path)
    t0 = time.perf_counter()
    for smi in mols:
        compute_reward_admet(smi, 0, 5, 0.9, cfg)
    t_admet_single = time.perf_counter() - t0

    # ADMET reward timing (batch prediction, like Route path)
    admet_model = _get_admet_model()
    t0 = time.perf_counter()
    batch_preds = admet_model.predict_properties(mols)
    for i, smi in enumerate(mols):
        preds = batch_preds.iloc[i].to_dict()
        compute_reward_admet(smi, 0, 5, 0.9, cfg, admet_preds=preds)
    t_admet_batch = time.perf_counter() - t0

    print(f"  QED-only ({len(mols)} mols): {t_qed*1000:.1f}ms "
          f"({t_qed/len(mols)*1000:.2f}ms/mol)")
    print(f"  ADMET single ({len(mols)} mols): {t_admet_single*1000:.1f}ms "
          f"({t_admet_single/len(mols)*1000:.2f}ms/mol)")
    print(f"  ADMET batch ({len(mols)} mols): {t_admet_batch*1000:.1f}ms "
          f"({t_admet_batch/len(mols)*1000:.2f}ms/mol)")
    print(f"  Overhead vs QED: single={t_admet_single/t_qed:.1f}x, "
          f"batch={t_admet_batch/t_qed:.1f}x")


def test_reward_range():
    """Verify ADMET reward produces reasonable values."""
    from reward.admet.reward import compute_reward_admet
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(PROJECT_ROOT / 'configs' / 'reward' / 'admet.yaml')

    print("\n" + "=" * 60)
    print("Test 6: Reward range sanity check")
    print("=" * 60)

    # Good drug-like molecules should have positive rewards
    good_mols = [
        ('Aspirin', 'CC(=O)Oc1ccccc1C(=O)O'),
        ('Ibuprofen', 'CC(C)Cc1ccc(C(C)C(=O)O)cc1'),
        ('Caffeine', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'),
    ]

    # "Bad" molecules (toxic or non-drug-like)
    bad_mols = [
        ('Benzene', 'c1ccccc1'),
        ('Methanol', 'CO'),
    ]

    print("\n  Good drug-like molecules:")
    for name, smi in good_mols:
        m = compute_reward_admet(smi, 0, 1, 1.0, cfg)
        r = m['reward']
        print(f"    {name:15s}: reward={r:.4f}, QED={m['qed']:.3f}, "
              f"hERG={m.get('hERG', 0):.3f}, AMES={m.get('AMES', 0):.3f}")

    print("\n  Simple/non-drug molecules:")
    for name, smi in bad_mols:
        m = compute_reward_admet(smi, 0, 1, 1.0, cfg)
        r = m['reward']
        print(f"    {name:15s}: reward={r:.4f}, QED={m['qed']:.3f}, "
              f"hERG={m.get('hERG', 0):.3f}, AMES={m.get('AMES', 0):.3f}")


if __name__ == '__main__':
    test_admet_reward_basic()
    test_admet_singleton()
    test_batch_consistency()
    test_performance()
    test_reward_range()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
