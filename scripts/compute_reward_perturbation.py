#!/usr/bin/env python3
"""Compute reward perturbation from predictor errors, including error correlation analysis."""

import numpy as np
import pandas as pd

base = "/shared/data1/Users/l1062811/git/DA-MolDQN"

# Normalization ranges (from agent.py get_scaler)
BDE_MAX, BDE_MIN = 96.58618528, 59.79533261
IP_MAX, IP_MIN = 178.1623553, 110.8306396
BDE_RANGE = BDE_MAX - BDE_MIN  # 36.79
IP_RANGE = IP_MAX - IP_MIN      # 67.33

# Reward weights
W_BDE = 0.8
W_IP = 0.2

print(f"BDE range: {BDE_RANGE:.2f} kcal/mol")
print(f"IP  range: {IP_RANGE:.2f} kcal/mol")
print(f"Weights: w_BDE={W_BDE}, w_IP={W_IP}")

# ============================================================
# Dataset: 163-molecule DFT validation
# ============================================================
dft_df = pd.read_csv(f"{base}/dft/public/dft_validation.csv")
dft_df = dft_df.dropna(subset=["ml_bde", "dft_bde", "ml_ip", "dft_ip"])
N = len(dft_df)
print(f"\n163-mol DFT validation (N={N})")

# Raw errors (ML - DFT)
err_bde = dft_df["ml_bde"].values - dft_df["dft_bde"].values  # pred - true
err_ip = dft_df["ml_ip"].values - dft_df["dft_ip"].values

print(f"\nBDE error: mean={err_bde.mean():+.3f}, std={err_bde.std():.3f}, MAE={np.abs(err_bde).mean():.3f}")
print(f"IP  error: mean={err_ip.mean():+.3f}, std={err_ip.std():.3f}, MAE={np.abs(err_ip).mean():.3f}")

# ============================================================
# Error correlation
# ============================================================
corr_pearson = np.corrcoef(err_bde, err_ip)[0, 1]
from scipy import stats
spearman_r, spearman_p = stats.spearmanr(err_bde, err_ip)

print(f"\n--- Error Correlation ---")
print(f"Pearson  r(err_BDE, err_IP) = {corr_pearson:.4f}")
print(f"Spearman ρ(err_BDE, err_IP) = {spearman_r:.4f} (p={spearman_p:.2e})")

# ============================================================
# Reward perturbation calculation
# ============================================================
# Reward = -w1 * nBDE + w2 * nIP + w3 * gamma
# nBDE = (BDE - min) / range
# Perturbation in normalized space:
# delta_nBDE = err_BDE / BDE_RANGE
# delta_nIP  = err_IP  / IP_RANGE
# delta_R = -w1 * delta_nBDE + w2 * delta_nIP  (signed)
# |delta_R| = reward perturbation magnitude

delta_nBDE = err_bde / BDE_RANGE
delta_nIP = err_ip / IP_RANGE

# Per-molecule reward perturbation (signed)
# R = -w1*nBDE + w2*nIP, so error in R = -w1*delta_nBDE + w2*delta_nIP
delta_R = -W_BDE * delta_nBDE + W_IP * delta_nIP
abs_delta_R = np.abs(delta_R)

print(f"\n--- Reward Perturbation (per-molecule) ---")
print(f"Mean |δR|     = {abs_delta_R.mean():.4f}")
print(f"Median |δR|   = {np.median(abs_delta_R):.4f}")
print(f"Std δR        = {delta_R.std():.4f}")
print(f"Max |δR|      = {abs_delta_R.max():.4f}")
print(f"Mean δR       = {delta_R.mean():+.4f}  (systematic bias)")

print(f"\nAs % of reward range [0.8, 2.5] (range=1.7):")
print(f"Mean |δR| / 1.7 = {abs_delta_R.mean()/1.7*100:.1f}%")
print(f"Std δR / 1.7     = {delta_R.std()/1.7*100:.1f}%")

# ============================================================
# Method 1: Simple sum of MAEs (worst case, what paper does)
# ============================================================
simple_sum = W_BDE * np.abs(err_bde).mean() / BDE_RANGE + W_IP * np.abs(err_ip).mean() / IP_RANGE
print(f"\n--- Method 1: Simple sum of MAEs (worst case) ---")
print(f"δR = w1*MAE_BDE/range + w2*MAE_IP/range = {simple_sum:.4f}")

# ============================================================
# Method 2: Independent (RMS combination)
# ============================================================
rms_combo = np.sqrt((W_BDE * np.abs(err_bde).mean() / BDE_RANGE)**2 +
                     (W_IP * np.abs(err_ip).mean() / IP_RANGE)**2)
print(f"\n--- Method 2: RMS combination (assuming independent) ---")
print(f"δR = sqrt((w1*MAE_BDE/range)^2 + (w2*MAE_IP/range)^2) = {rms_combo:.4f}")

# ============================================================
# Method 3: Actual per-molecule calculation (ground truth)
# ============================================================
print(f"\n--- Method 3: Actual per-molecule mean |δR| ---")
print(f"Mean |δR| = {abs_delta_R.mean():.4f}")

# ============================================================
# Check: where does 0.29 come from?
# ============================================================
print(f"\n--- Checking paper's 0.29 ---")
print(f"Simple sum with MAE:  {simple_sum:.4f}")
print(f"Simple sum with RMSE: {W_BDE * np.sqrt((err_bde**2).mean()) / BDE_RANGE + W_IP * np.sqrt((err_ip**2).mean()) / IP_RANGE:.4f}")
print(f"Using max errors:     {W_BDE * np.abs(err_bde).max() / BDE_RANGE + W_IP * np.abs(err_ip).max() / IP_RANGE:.4f}")
# Maybe they added absolute values differently?
print(f"Sum of abs normalized errors: {(W_BDE * np.abs(delta_nBDE) + W_IP * np.abs(delta_nIP)).mean():.4f}")
# Or maybe 0.8*MAE_BDE/BDE_RANGE + 0.2*MAE_IP/IP_RANGE but with wrong ranges?
print(f"If BDE_RANGE=15, IP_RANGE=30: {W_BDE * 5.5/15 + W_IP * 8.5/30:.4f}")
