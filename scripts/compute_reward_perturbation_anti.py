#!/usr/bin/env python3
"""Compute reward perturbation and error correlation using the proprietary antioxidant dataset."""

import numpy as np
import pandas as pd
from scipy import stats

base = "/shared/data1/Users/l1062811/git/DA-MolDQN"

BDE_FACTOR = 0.9
IP_FACTOR = 0.8

# Normalization ranges (from agent.py get_scaler)
BDE_MAX, BDE_MIN = 96.58618528, 59.79533261
IP_MAX, IP_MIN = 178.1623553, 110.8306396
BDE_RANGE = BDE_MAX - BDE_MIN
IP_RANGE = IP_MAX - IP_MIN

W_BDE = 0.8
W_IP = 0.2

print(f"BDE range: {BDE_RANGE:.2f}, IP range: {IP_RANGE:.2f}")
print(f"BDE factor: {BDE_FACTOR}, IP factor: {IP_FACTOR}")

# Load data (same SMILES order, verified)
bde_df = pd.read_csv(f"{base}/Data/anti-bde.csv", sep="\t")
ip_df = pd.read_csv(f"{base}/Data/anti-ip.csv", sep="\t")
ip_aimnet = pd.read_csv(f"{base}/Data/anti-ip-compared.csv", sep="\t")

# anti-ip.csv has 71 empty IP rows; anti-ip-compared.csv has 445 rows (non-empty ones)
# Need to identify which rows in the original 516 have valid IP DFT and AIMNet predictions

# Build merged dataset: need BDE_DFT, BDE_ALFABET, IP_DFT, IP_AIMNet for same molecules
# ip_df has 516 rows, some with empty IP
# ip_aimnet has 445 rows (the non-empty ones, in order)

# Mark valid IP rows
ip_valid_mask = ip_df['IP'].notna() & (ip_df['IP'] != '')
# Convert IP to float where valid
ip_dft_all = pd.to_numeric(ip_df['IP'], errors='coerce')

# Build aligned arrays
smiles = bde_df['structure'].values
bde_dft = bde_df['BDE_DFT'].values
bde_alfabet = bde_df['BDE_ALFABET'].values * BDE_FACTOR  # calibrated

ip_dft = ip_dft_all.values
# ip_aimnet has 445 rows matching the 445 non-NaN IP rows
ip_aimnet_vals = np.full(516, np.nan)
aimnet_idx = 0
for i in range(516):
    if not np.isnan(ip_dft[i]):
        ip_aimnet_vals[i] = ip_aimnet['IP_AIMNet_cal'].iloc[aimnet_idx]
        aimnet_idx += 1

# Find molecules with all 4 values (BDE_DFT, BDE_ALFABET, IP_DFT, IP_AIMNet)
valid = ~np.isnan(bde_dft) & ~np.isnan(ip_dft) & ~np.isnan(ip_aimnet_vals)
N = valid.sum()
print(f"\nMolecules with all 4 values: {N}")

bde_dft_v = bde_dft[valid]
bde_pred_v = bde_alfabet[valid]  # ALFABET × 0.9
ip_dft_v = ip_dft[valid]
ip_pred_v = ip_aimnet_vals[valid]  # AIMNet × 0.8

# Errors (pred - true)
err_bde = bde_pred_v - bde_dft_v
err_ip = ip_pred_v - ip_dft_v

print(f"\n--- Individual Errors ---")
print(f"BDE: MAE={np.abs(err_bde).mean():.3f}, bias={err_bde.mean():+.3f}, std={err_bde.std():.3f}")
print(f"IP:  MAE={np.abs(err_ip).mean():.3f}, bias={err_ip.mean():+.3f}, std={err_ip.std():.3f}")

# ============================================================
# Error Correlation
# ============================================================
r_pearson, p_pearson = stats.pearsonr(err_bde, err_ip)
r_spearman, p_spearman = stats.spearmanr(err_bde, err_ip)

print(f"\n--- Error Correlation (N={N}) ---")
print(f"Pearson  r = {r_pearson:.4f}  (p={p_pearson:.2e})")
print(f"Spearman ρ = {r_spearman:.4f}  (p={p_spearman:.2e})")

# ============================================================
# Reward Perturbation
# ============================================================
# R = -w1 * nBDE + w2 * nIP
# delta_R = -w1 * (err_BDE / BDE_RANGE) + w2 * (err_IP / IP_RANGE)
delta_nBDE = err_bde / BDE_RANGE
delta_nIP = err_ip / IP_RANGE
delta_R = -W_BDE * delta_nBDE + W_IP * delta_nIP

print(f"\n--- Reward Perturbation (per-molecule, N={N}) ---")
print(f"Mean |δR|   = {np.abs(delta_R).mean():.4f}")
print(f"Median |δR| = {np.median(np.abs(delta_R)):.4f}")
print(f"Std δR      = {delta_R.std():.4f}")
print(f"Max |δR|    = {np.abs(delta_R).max():.4f}")
print(f"Mean δR     = {delta_R.mean():+.4f}  (systematic bias in reward)")

reward_range = 1.7  # typical range [0.8, 2.5]
print(f"\nAs % of reward range {reward_range}:")
print(f"  Mean |δR| = {np.abs(delta_R).mean()/reward_range*100:.1f}%")
print(f"  Std δR    = {delta_R.std()/reward_range*100:.1f}%")

# ============================================================
# Comparison of methods
# ============================================================
mae_bde = np.abs(err_bde).mean()
mae_ip = np.abs(err_ip).mean()

m1 = W_BDE * mae_bde / BDE_RANGE + W_IP * mae_ip / IP_RANGE
m2 = np.sqrt((W_BDE * mae_bde / BDE_RANGE)**2 + (W_IP * mae_ip / IP_RANGE)**2)
m3 = np.abs(delta_R).mean()

print(f"\n--- Summary ---")
print(f"Method 1 (worst-case sum):    {m1:.4f}")
print(f"Method 2 (independent RMS):   {m2:.4f}")
print(f"Method 3 (actual mean |δR|):  {m3:.4f}")
print(f"Paper claims:                 0.29  <-- ERROR")
