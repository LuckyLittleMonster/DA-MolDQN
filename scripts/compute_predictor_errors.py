#!/usr/bin/env python3
"""Compute MAE, MAPE, MSE, RMSE for BDE/IP predictors on proprietary and DFT validation datasets."""

import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred, name=""):
    """Compute and print error metrics."""
    diff = y_pred - y_true
    abs_diff = np.abs(diff)

    mae = np.mean(abs_diff)
    mape = np.mean(abs_diff / np.abs(y_true)) * 100
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    bias = np.mean(diff)

    print(f"\n{'=' * 50}")
    print(f"  {name}  (N={len(y_true)})")
    print(f"{'=' * 50}")
    print(f"  MAE   = {mae:.3f} kcal/mol")
    print(f"  MAPE  = {mape:.2f}%")
    print(f"  MSE   = {mse:.3f} (kcal/mol)^2")
    print(f"  RMSE  = {rmse:.3f} kcal/mol")
    print(f"  Bias  = {bias:+.3f} kcal/mol  (pred - true)")
    print(f"  Range: true [{y_true.min():.1f}, {y_true.max():.1f}], "
          f"pred [{y_pred.min():.1f}, {y_pred.max():.1f}]")
    return {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": rmse, "Bias": bias}


def main():
    base = "/shared/data1/Users/l1062811/git/DA-MolDQN"

    # Linear correction factors from hyp.py (applied to ML predictions before use in RL)
    BDE_FACTOR = 0.9  # hyp.bde_factor: ALFABET_pred * 0.9
    IP_FACTOR = 0.8   # hyp.ip_factor:  AIMNet_pred * 0.8

    # ============================================================
    # Dataset 1: Proprietary antioxidant BDE (ALFABET vs DFT)
    # ============================================================
    print("\n" + "#" * 60)
    print("# Dataset 1: Proprietary antioxidants (anti-bde.csv)")
    print("#" * 60)

    bde_df = pd.read_csv(f"{base}/Data/anti-bde.csv", sep="\t")
    bde_valid = bde_df.dropna(subset=["BDE_DFT", "BDE_ALFABET"])
    print(f"  (Dropped {len(bde_df) - len(bde_valid)} rows with NaN BDE_DFT)")

    bde_dft = bde_valid["BDE_DFT"].values
    bde_raw = bde_valid["BDE_ALFABET"].values
    bde_cal = bde_raw * BDE_FACTOR

    print("\n--- Raw ALFABET (no correction) ---")
    compute_metrics(bde_dft, bde_raw, name="BDE: ALFABET_raw vs DFT")

    print("\n--- Calibrated ALFABET × 0.9 ---")
    compute_metrics(bde_dft, bde_cal, name="BDE: ALFABET×0.9 vs DFT")

    # ============================================================
    # Dataset 2: 163-molecule DFT validation
    # ============================================================
    print("\n" + "#" * 60)
    print("# Dataset 2: 163-molecule DFT validation (dft_validation.csv)")
    print("#" * 60)

    dft_df = pd.read_csv(f"{base}/dft/public/dft_validation.csv")
    dft_df = dft_df.dropna(subset=["ml_bde", "dft_bde", "ml_ip", "dft_ip"])

    ml_bde = dft_df["ml_bde"].values
    ml_ip = dft_df["ml_ip"].values
    dft_bde = dft_df["dft_bde"].values
    dft_ip = dft_df["dft_ip"].values

    print("\n--- Raw ML predictions (no correction) ---")
    compute_metrics(dft_bde, ml_bde, name="BDE: ALFABET_raw vs DFT (163 mols)")
    compute_metrics(dft_ip, ml_ip, name="IP: AIMNet_raw vs DFT (163 mols)")

    print("\n--- Calibrated ML × factor ---")
    compute_metrics(dft_bde, ml_bde * BDE_FACTOR, name="BDE: ALFABET×0.9 vs DFT (163 mols)")
    compute_metrics(dft_ip, ml_ip * IP_FACTOR, name="IP: AIMNet×0.8 vs DFT (163 mols)")

    # ============================================================
    # Note about IP on proprietary
    # ============================================================
    print("\n" + "#" * 60)
    print("# Note: Data/anti-ip.csv has only ML-predicted IP values")
    print("# (no DFT ground truth), so error metrics cannot be computed.")
    print("#" * 60)


if __name__ == "__main__":
    main()
