"""Benchmark proxy models vs Uni-Dock real docking for sEH.

Compares:
- sEH MPNN proxy (Bengio et al. 2021, used by RxnFlow/SynFlowNet/RGFN)
- Uni-Dock GPU docking (real physics-based scoring)

Metrics: Pearson/Spearman correlation, MAE, rank correlation, speed.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")


def load_test_molecules():
    """Load diverse test molecules for benchmarking."""
    # Mix of drug-like molecules with varying properties
    return [
        ("benzene", "c1ccccc1"),
        ("ethanol", "CCO"),
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("haloperidol", "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"),
        ("celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),
        ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("naproxen", "COc1ccc2cc(CC(C)C(=O)O)ccc2c1"),
        ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
        ("indomethacin", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1"),
        ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
        ("metformin", "CN(C)C(=N)NC(=N)N"),
        ("atorvastatin", "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1"),
        ("losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"),
        ("omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
        ("metoprolol", "COCCc1ccc(OCC(O)CNC(C)C)cc1"),
        ("amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl"),
        ("valsartan", "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)C(C(=O)O)C(C)C"),
        ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("nicotine", "CN1CCC[C@@H]1c1cccnc1"),
        ("quercetin", "Oc1cc(O)c2c(c1)oc(-c1ccc(O)c(O)c1)c(O)c2=O"),
        ("curcumin", "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O"),
        ("resveratrol", "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1"),
        ("melatonin", "COc1ccc2[nH]cc(CCNC(C)=O)c2c1"),
        ("sildenafil", "CCCc1nn(C)c2c1nc(-c1cc(S(=O)(=O)N1CC(O)CC1)ccc1OCC)nc2O"),
        ("lisinopril", "NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O"),
        ("captopril", "C[C@@H](CS)C(=O)N1CCC[C@H]1C(=O)O"),
        ("fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
        ("sertraline", "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21"),
        ("clopidogrel", "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"),
        ("tamoxifen", "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
        ("dexamethasone", "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO"),
    ]


def main():
    from reward.docking_score import UniDockScorer, ProxyScorer

    molecules = load_test_molecules()
    names = [m[0] for m in molecules]
    smiles = [m[1] for m in molecules]
    n = len(smiles)

    print(f"Benchmarking proxy vs Uni-Dock on {n} molecules")
    print("=" * 70)

    # --- Proxy scoring ---
    print("\n1. sEH Proxy (MPNN, Bengio et al. 2021)")
    t0 = time.perf_counter()
    proxy = ProxyScorer("seh", device="cuda")
    t1 = time.perf_counter()
    proxy_scores = proxy.score(smiles)
    t2 = time.perf_counter()
    print(f"   Init: {t1 - t0:.2f}s, Predict: {(t2 - t1) * 1000:.1f}ms ({n} mols)")
    print(f"   Speed: {(t2 - t1) / n * 1000:.2f} ms/mol")

    # --- Uni-Dock scoring ---
    print("\n2. Uni-Dock (GPU-accelerated real docking)")
    target_dir = Path(__file__).resolve().parent.parent / "Data" / "docking_targets" / "seh"
    config_path = target_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    receptor_pdbqt = str(target_dir / "receptor.pdbqt")
    t3 = time.perf_counter()
    dock = UniDockScorer(
        receptor_pdbqt=receptor_pdbqt,
        center_x=cfg["center_x"],
        center_y=cfg["center_y"],
        center_z=cfg["center_z"],
        size_x=cfg.get("size_x", 22.5),
        size_y=cfg.get("size_y", 22.5),
        size_z=cfg.get("size_z", 22.5),
    )
    t4 = time.perf_counter()
    dock_scores = dock.batch_dock(smiles)
    t5 = time.perf_counter()
    print(f"   Init: {t4 - t3:.2f}s, Dock: {t5 - t4:.1f}s ({n} mols)")
    print(f"   Speed: {(t5 - t4) / n * 1000:.0f} ms/mol")

    # --- Comparison ---
    print("\n3. Per-molecule comparison")
    print(f"   {'Name':20s} {'Proxy':>8s} {'Vina(kcal)':>10s} {'Vina_norm':>10s}")
    print("   " + "-" * 52)

    proxy_arr = np.array(proxy_scores)
    dock_arr = np.array(dock_scores)
    # Normalize docking: clip(-score/12, 0, 1) — same as our reward
    dock_norm = np.clip(-dock_arr / 12.0, 0, 1)

    for name, ps, ds, dn in zip(names, proxy_scores, dock_scores, dock_norm):
        print(f"   {name:20s} {ps:8.4f} {ds:10.2f} {dn:10.4f}")

    # --- Correlation metrics ---
    from scipy import stats

    # Only use molecules with valid dock scores
    valid = dock_arr != 0
    if valid.sum() < 5:
        print("\n   WARNING: Too few valid docking results for correlation!")
        return

    p_valid = proxy_arr[valid]
    d_valid = dock_norm[valid]

    pearson_r, pearson_p = stats.pearsonr(p_valid, d_valid)
    spearman_r, spearman_p = stats.spearmanr(p_valid, d_valid)
    mae = np.mean(np.abs(p_valid - d_valid))
    # Kendall tau (rank correlation)
    kendall_tau, kendall_p = stats.kendalltau(p_valid, d_valid)

    print(f"\n4. Correlation metrics (n={valid.sum()} valid molecules)")
    print(f"   Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"   Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"   Kendall τ:  {kendall_tau:.4f} (p={kendall_p:.2e})")
    print(f"   MAE:        {mae:.4f}")

    print(f"\n5. Speed comparison")
    speedup = (t5 - t4) / max(t2 - t1, 1e-6)
    print(f"   Proxy: {(t2 - t1) * 1000:.1f}ms total, {(t2 - t1) / n * 1000:.2f}ms/mol")
    print(f"   Dock:  {(t5 - t4) * 1000:.0f}ms total, {(t5 - t4) / n * 1000:.0f}ms/mol")
    print(f"   Speedup: {speedup:.0f}x")

    # --- Summary ---
    print(f"\n6. Score distributions")
    print(f"   Proxy:    min={proxy_arr.min():.4f}, max={proxy_arr.max():.4f}, "
          f"mean={proxy_arr.mean():.4f}, std={proxy_arr.std():.4f}")
    print(f"   Dock norm: min={dock_norm[valid].min():.4f}, max={dock_norm[valid].max():.4f}, "
          f"mean={dock_norm[valid].mean():.4f}, std={dock_norm[valid].std():.4f}")


if __name__ == "__main__":
    main()
