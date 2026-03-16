import time, os, sys
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
import logging
logging.disable(logging.INFO)

smiles_40 = [
    "C", "CC", "CCC", "c1ccccc1", "CCO", "c1ccncc1",
    "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
] * 4

device = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# === 1. unimol_tools get_repr ===
from unimol_tools import UniMolRepr
repr_model = UniMolRepr(data_type='molecule', remove_hs=False, use_cuda=True)
_ = repr_model.get_repr(["C"])

times = []
for _ in range(3):
    t0 = time.perf_counter()
    r = repr_model.get_repr(smiles_40)
    times.append(time.perf_counter() - t0)
print(f"\n1. get_repr (40 mols): {np.mean(times)*1000:.1f}ms = {np.mean(times)/40*1000:.1f}ms/mol")

# === 2. Bare-metal ===
from unimol_tools.data.conformer import ConformerGen, coords2unimol, inner_smi2coords
confgen = ConformerGen(data_type='molecule', remove_hs=False)
model = repr_model.model.to(device)
model.eval()

# A: ETKDG
t_a = []
for _ in range(3):
    t0 = time.perf_counter()
    atoms_list, coords_list = [], []
    for smi in smiles_40:
        atoms, coords, mol = inner_smi2coords(smi, seed=42, mode='fast', remove_hs=False)
        atoms_list.append(atoms)
        coords_list.append(coords)
    t_a.append(time.perf_counter() - t0)
print(f"2a. ETKDG (40 mols serial): {np.mean(t_a)*1000:.1f}ms = {np.mean(t_a)/40*1000:.1f}ms/mol")

# B: transform_raw
t_b = []
for _ in range(3):
    t0 = time.perf_counter()
    inputs = confgen.transform_raw(atoms_list, coords_list)
    t_b.append(time.perf_counter() - t0)
print(f"2b. transform_raw (40 mols): {np.mean(t_b)*1000:.1f}ms = {np.mean(t_b)/40*1000:.1f}ms/mol")

# C: Manual batch + GPU inference
def collate_batch(inputs_list, dev):
    bs = len(inputs_list)
    max_n = max(inp['src_tokens'].shape[0] for inp in inputs_list)
    src_tokens = torch.zeros(bs, max_n, dtype=torch.long, device=dev)
    src_coord = torch.zeros(bs, max_n, 3, dtype=torch.float32, device=dev)
    src_distance = torch.zeros(bs, max_n, max_n, dtype=torch.float32, device=dev)
    src_edge_type = torch.zeros(bs, max_n, max_n, dtype=torch.long, device=dev)
    for i, inp in enumerate(inputs_list):
        n = inp['src_tokens'].shape[0]
        src_tokens[i, :n] = torch.tensor(inp['src_tokens'], dtype=torch.long)
        src_coord[i, :n] = torch.tensor(inp['src_coord'], dtype=torch.float32)
        src_distance[i, :n, :n] = torch.tensor(inp['src_distance'], dtype=torch.float32)
        src_edge_type[i, :n, :n] = torch.tensor(inp['src_edge_type'], dtype=torch.long)
    return {'src_tokens': src_tokens, 'src_coord': src_coord,
            'src_distance': src_distance, 'src_edge_type': src_edge_type}

# Warmup
batch = collate_batch(inputs, device)
with torch.no_grad():
    out = model(**batch, return_repr=True, return_atomic_reprs=False)
    if isinstance(out, tuple):
        cls_repr = out[0]
    else:
        cls_repr = out
torch.cuda.synchronize()

t_c = []
for _ in range(5):
    batch = collate_batch(inputs, device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**batch, return_repr=True, return_atomic_reprs=False)
        if isinstance(out, tuple):
            cls_repr = out[0]
        else:
            cls_repr = out
    torch.cuda.synchronize()
    t_c.append(time.perf_counter() - t0)
print(f"2c. GPU inference (40 mols): {np.mean(t_c)*1000:.1f}ms  shape={cls_repr.shape}")

total = np.mean(t_a) + np.mean(t_b) + np.mean(t_c)
print(f"\n2. Bare-metal total: {total*1000:.1f}ms = {total/40*1000:.1f}ms/mol")
print(f"   Speedup vs get_repr: {np.mean(times)/total:.1f}x")

# === 3. Collation overhead ===
t_col = []
for _ in range(10):
    t0 = time.perf_counter()
    batch = collate_batch(inputs, device)
    torch.cuda.synchronize()
    t_col.append(time.perf_counter() - t0)
print(f"\n3. Collation overhead: {np.mean(t_col)*1000:.1f}ms")

print(f"\n=== Breakdown (40 mols) ===")
print(f"  ETKDG:        {np.mean(t_a)*1000:7.1f}ms ({np.mean(t_a)/total*100:.0f}%)")
print(f"  Featurize:    {np.mean(t_b)*1000:7.1f}ms ({np.mean(t_b)/total*100:.0f}%)")
print(f"  GPU infer:    {np.mean(t_c)*1000:7.1f}ms ({np.mean(t_c)/total*100:.0f}%)")
print(f"  Collation:    {np.mean(t_col)*1000:7.1f}ms")
print(f"  TOTAL:        {total*1000:7.1f}ms")
