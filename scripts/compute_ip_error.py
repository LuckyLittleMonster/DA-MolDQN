#!/usr/bin/env python3
"""Compute AIMNet-NSE IP prediction error on the proprietary antioxidant dataset.

Reads Data/anti-ip.csv (SMILES + DFT IP), runs AIMNet-NSE single model prediction,
and computes MAE/MAPE/RMSE with and without the ip_factor=0.8 calibration.
"""

import sys
import os
import csv
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import load_models, to_numpy


def ev2kcal_per_mol(ev):
    return ev * 23.0609


def calc_react_idx(data):
    ip = data['energy'][0] - data['energy'][1]
    ea = data['energy'][1] - data['energy'][2]
    return ip, ea


def predict_ip_single(smiles, model, device, max_attempts=7):
    """Predict IP for a single SMILES using AIMNet-NSE."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    # ETKDG
    cid = -1
    for _ in range(max_attempts):
        cid = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=1)
        if cid >= 0:
            break
    if cid < 0:
        return None

    coords = mol.GetConformer(cid).GetPositions()
    coords = torch.tensor(coords, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1).to(device)
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(3, 1).to(device)
    charge = torch.tensor([1, 0, -1]).to(device)
    mult = torch.tensor([2, 1, 2]).to(device)
    data = dict(coord=coords, numbers=numbers, charge=charge, mult=mult)

    with torch.jit.optimized_execution(False), torch.no_grad():
        pred = model(data)
    pred['charges'] = pred['charges'].sum(-1)
    pred = to_numpy(pred)

    ip_ev = pred['energy'][0] - pred['energy'][1]
    ip_kcal = ev2kcal_per_mol(ip_ev)
    return ip_kcal


def compute_metrics(y_true, y_pred, name=""):
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    mae = np.mean(abs_diff)
    mape = np.mean(abs_diff / np.abs(y_true)) * 100
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    bias = np.mean(diff)
    print(f"\n{'=' * 55}")
    print(f"  {name}  (N={len(y_true)})")
    print(f"{'=' * 55}")
    print(f"  MAE   = {mae:.3f} kcal/mol")
    print(f"  MAPE  = {mape:.2f}%")
    print(f"  MSE   = {mse:.3f} (kcal/mol)^2")
    print(f"  RMSE  = {rmse:.3f} kcal/mol")
    print(f"  Bias  = {bias:+.3f} kcal/mol  (pred - true)")


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load single AIMNet-NSE model (model cv4, same as agent.py L479)
    model_path = os.path.join(base, 'aimnetnse-models/aimnet-nse-cv4.jpt')
    print(f"Loading model: {model_path}")
    model = load_models([model_path]).to(device)

    # Read anti-ip.csv
    ip_file = os.path.join(base, 'Data/anti-ip.csv')
    smiles_list = []
    dft_ips = []
    skipped = 0
    with open(ip_file) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2 or row[1].strip() == '':
                skipped += 1
                continue
            smiles_list.append(row[0])
            dft_ips.append(float(row[1]))
    print(f"Skipped {skipped} rows with empty IP")

    print(f"Total molecules: {len(smiles_list)}")

    # Predict IP for each molecule
    pred_ips = []
    dft_matched = []
    failed = 0
    for i, (smi, dft_ip) in enumerate(zip(smiles_list, dft_ips)):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(smiles_list)}, failed={failed}")
        ip = predict_ip_single(smi, model, device)
        if ip is not None:
            pred_ips.append(ip)
            dft_matched.append(dft_ip)
        else:
            failed += 1

    print(f"\nDone. Success: {len(pred_ips)}, Failed: {failed}")

    y_true = np.array(dft_matched)
    y_pred = np.array(pred_ips)

    IP_FACTOR = 0.8

    print("\n--- Raw AIMNet-NSE (no correction) ---")
    compute_metrics(y_true, y_pred, name="IP: AIMNet_raw vs DFT (proprietary)")

    print("\n--- Calibrated AIMNet × 0.8 ---")
    compute_metrics(y_true, y_pred * IP_FACTOR, name="IP: AIMNet×0.8 vs DFT (proprietary)")

    # Save results
    out_path = os.path.join(base, 'Data/anti-ip-compared.csv')
    with open(out_path, 'w') as f:
        f.write("IP_DFT\tIP_AIMNet\tIP_AIMNet_cal\n")
        for dft, pred in zip(dft_matched, pred_ips):
            f.write(f"{dft:.4f}\t{pred:.4f}\t{pred*IP_FACTOR:.4f}\n")
    print(f"\nSaved {len(dft_matched)} results to {out_path}")


if __name__ == "__main__":
    main()
