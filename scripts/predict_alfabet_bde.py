"""Predict min O-H BDE for anti_400 molecules using ALFABET.

Usage:
    pip install alfabet
    python scripts/predict_alfabet_bde.py > Experiments/anti400_bde_alfabet.csv
"""
import sys
from alfabet import model

input_path = "Data/anti_400.txt"
if len(sys.argv) > 1:
    input_path = sys.argv[1]

with open(input_path) as f:
    smiles = [line.strip() for line in f if line.strip()]

print("idx,smiles,oh_bde,valid")

n_valid = 0
bdes_valid = []

for i, smi in enumerate(smiles):
    try:
        df = model.predict([smi])
        oh_rows = df[df["bond_type"].str.contains("O-H|H-O")]
        if len(oh_rows) > 0:
            oh_bde = oh_rows["bde_pred"].min()
            print(f"{i},{smi},{oh_bde:.4f},True")
            n_valid += 1
            bdes_valid.append(oh_bde)
        else:
            print(f"{i},{smi},0.0000,False")
    except Exception as e:
        print(f"{i},{smi},0.0000,False", file=sys.stdout)
        print(f"# Error mol {i} ({smi}): {e}", file=sys.stderr)

import numpy as np
arr = np.array(bdes_valid)
print(f"# valid: {n_valid}/{len(smiles)}", file=sys.stderr)
if len(arr) > 0:
    print(f"# BDE stats: mean={arr.mean():.2f}, std={arr.std():.2f}, "
          f"min={arr.min():.2f}, max={arr.max():.2f}", file=sys.stderr)
