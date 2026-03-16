"""Test ConstrainedEmbed behavior:
1. Does it modify matched atom coordinates?
2. Why does it fail? Can we improve success rate?
3. Can 3D distance filter the action space?
"""
import time
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D
RDLogger.DisableLog("rdApp.*")

import sys
sys.path.insert(0, '/shared/data1/Users/l1062811/git/SynDQN')
import src.cenv as cenv

print("=" * 60)
print("Q1: Does ConstrainedEmbed modify matched atom coords?")
print("=" * 60)

# Create parent with conformer
parent = Chem.MolFromSmiles("c1ccccc1")  # benzene
AllChem.EmbedMolecule(parent, randomSeed=42)
AllChem.MMFFOptimizeMolecule(parent)

parent_conf = parent.GetConformer()
parent_coords = np.array([parent_conf.GetAtomPosition(i) for i in range(parent.GetNumAtoms())])
print(f"\nParent: {Chem.MolToSmiles(parent)} ({parent.GetNumAtoms()} atoms)")
print(f"Parent coords:\n{parent_coords}")

# Child = toluene (benzene + methyl)
child = Chem.MolFromSmiles("Cc1ccccc1")
child_embedded = AllChem.ConstrainedEmbed(child, parent)
child_conf = child_embedded.GetConformer()
child_coords = np.array([child_conf.GetAtomPosition(i) for i in range(child_embedded.GetNumAtoms())])

# Find substructure match to see which child atoms correspond to parent
match = child_embedded.GetSubstructMatch(parent)
print(f"\nChild: {Chem.MolToSmiles(child)} ({child_embedded.GetNumAtoms()} atoms)")
print(f"Substructure match (child_idx for each parent_idx): {match}")

# Compare matched atom coordinates
print(f"\nCoord comparison for matched atoms:")
max_drift = 0
for parent_idx, child_idx in enumerate(match):
    p_pos = parent_coords[parent_idx]
    c_pos = child_coords[child_idx]
    dist = np.linalg.norm(p_pos - c_pos)
    max_drift = max(max_drift, dist)
    print(f"  Parent atom {parent_idx} → Child atom {child_idx}: drift = {dist:.4f} Å")

print(f"\nMax coordinate drift: {max_drift:.4f} Å")
print(f"Conclusion: {'Coords preserved (drift < 0.1Å)' if max_drift < 0.1 else 'Coords MODIFIED'}")

# Test with useTethers=False
child2 = Chem.MolFromSmiles("Cc1ccccc1")
child2_embedded = AllChem.ConstrainedEmbed(child2, parent, useTethers=False)
child2_conf = child2_embedded.GetConformer()
match2 = child2_embedded.GetSubstructMatch(parent)
max_drift2 = 0
for parent_idx, child_idx in enumerate(match2):
    p_pos = parent_coords[parent_idx]
    c_pos = np.array(child2_conf.GetAtomPosition(child_idx))
    max_drift2 = max(max_drift2, np.linalg.norm(p_pos - c_pos))
print(f"\nWith useTethers=False: max drift = {max_drift2:.4f} Å")

print("\n" + "=" * 60)
print("Q2: Why does ConstrainedEmbed fail? Improving success rate")
print("=" * 60)

# Setup cenv
flags = cenv.Flags()
env = cenv.Environment(["C", "O", "N"], [3, 5, 6], 3, 2048, flags)

# Run through episode, categorize failures
np.random.seed(42)
state = Chem.MolFromSmiles("C")
parent_with_conf = None

total_actions = 0
total_success = 0
total_no_match = 0
total_embed_fail = 0
failure_examples = []

for step in range(20):
    valid_actions, _ = env.get_valid_actions_and_fingerprint(state, 0, False)
    
    if parent_with_conf is not None:
        for mol in valid_actions:
            total_actions += 1
            # Check if parent is substructure of child
            match = mol.GetSubstructMatch(parent_with_conf)
            if not match:
                total_no_match += 1
                if len(failure_examples) < 5:
                    failure_examples.append((
                        "NO_MATCH",
                        Chem.MolToSmiles(parent_with_conf),
                        Chem.MolToSmiles(mol)
                    ))
                continue
            # Try ConstrainedEmbed
            try:
                result = AllChem.ConstrainedEmbed(Chem.RWMol(mol), parent_with_conf, randomseed=42)
                total_success += 1
            except Exception as e:
                total_embed_fail += 1
                if len(failure_examples) < 10:
                    failure_examples.append((
                        f"EMBED_FAIL: {str(e)[:60]}",
                        Chem.MolToSmiles(parent_with_conf),
                        Chem.MolToSmiles(mol)
                    ))
    
    # Advance state
    pick = np.random.randint(0, max(1, len(valid_actions) - 1))
    state = Chem.RWMol(valid_actions[pick])
    
    # Build parent conformer
    parent_with_conf = Chem.RWMol(state)
    res = AllChem.EmbedMolecule(parent_with_conf, randomSeed=42)
    if res != 0:
        parent_with_conf = None
        continue
    try:
        AllChem.MMFFOptimizeMolecule(parent_with_conf)
    except:
        pass

print(f"\nTotal actions tested: {total_actions}")
print(f"  Success:      {total_success} ({total_success/total_actions*100:.1f}%)")
print(f"  No match:     {total_no_match} ({total_no_match/total_actions*100:.1f}%)")
print(f"  Embed fail:   {total_embed_fail} ({total_embed_fail/total_actions*100:.1f}%)")

print(f"\nFailure examples:")
for reason, parent_smi, child_smi in failure_examples:
    print(f"  {reason}")
    print(f"    Parent: {parent_smi}")
    print(f"    Child:  {child_smi}")

# Q2b: Try fallback strategies for no-match cases
print(f"\n--- Fallback: reverse match (child as substructure of parent) ---")
state = Chem.MolFromSmiles("CCO")  
AllChem.EmbedMolecule(state, randomSeed=42)
AllChem.MMFFOptimizeMolecule(state)

# Bond removal: child is substructure of parent
child_removal = Chem.MolFromSmiles("CC")  # remove O
match_fwd = child_removal.GetSubstructMatch(state)  # child in parent? no
match_rev = state.GetSubstructMatch(child_removal)   # parent contains child? 
print(f"  Parent: CCO, Child (bond removal): CC")
print(f"  child.GetSubstructMatch(parent): {match_fwd}")
print(f"  parent.GetSubstructMatch(child): {match_rev}")
# For removals: child is smaller, so we can use child's coords from parent
if match_rev:
    print(f"  → Can extract child coords from parent conformer directly!")


print("\n" + "=" * 60)
print("Q3: 3D distance-based action filtering")
print("=" * 60)

# For a molecule with conformer, check distances between non-bonded atoms
mol = Chem.MolFromSmiles("CC(=O)C(N)=CN")  # 7 heavy atoms
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)
conf = mol.GetConformer()

print(f"\nMolecule: {Chem.MolToSmiles(mol)} ({mol.GetNumAtoms()} atoms)")
print(f"\nPairwise distances (Å) between non-bonded heavy atoms:")

# Bond lengths reference
bond_lengths = {"C-C": 1.54, "C=C": 1.34, "C-O": 1.43, "C=O": 1.23, 
                "C-N": 1.47, "C=N": 1.29, "C#N": 1.16}
max_bond_len = 1.54  # single C-C

n_atoms = mol.GetNumAtoms()
bonded_pairs = set()
for bond in mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    bonded_pairs.add((min(i,j), max(i,j)))

close_pairs = 0
far_pairs = 0
distances = []
for i in range(n_atoms):
    for j in range(i+1, n_atoms):
        if (i,j) in bonded_pairs:
            continue
        pos_i = np.array(conf.GetAtomPosition(i))
        pos_j = np.array(conf.GetAtomPosition(j))
        dist = np.linalg.norm(pos_i - pos_j)
        sym_i = mol.GetAtomWithIdx(i).GetSymbol()
        sym_j = mol.GetAtomWithIdx(j).GetSymbol()
        distances.append((dist, i, j, sym_i, sym_j))

distances.sort()
for dist, i, j, si, sj in distances:
    feasible = "✓ bondable" if dist < max_bond_len * 2.5 else "✗ too far"
    print(f"  {si}{i}-{sj}{j}: {dist:.2f}Å  {feasible}")
    if dist < max_bond_len * 2.5:
        close_pairs += 1
    else:
        far_pairs += 1

print(f"\nWith threshold {max_bond_len * 2.5:.2f}Å (2.5 × C-C bond):")
print(f"  Bondable pairs: {close_pairs}")
print(f"  Too far pairs:  {far_pairs}")
print(f"  Action space reduction: {far_pairs/(close_pairs+far_pairs)*100:.0f}% of add-bond actions filtered out")

# Larger molecule test
print(f"\n--- Larger molecule test ---")
mol2 = Chem.MolFromSmiles("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C")  # testosterone
AllChem.EmbedMolecule(mol2, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol2)
conf2 = mol2.GetConformer()
n2 = mol2.GetNumAtoms()

bonded2 = set()
for bond in mol2.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    bonded2.add((min(i,j), max(i,j)))

close2 = 0
far2 = 0
for i in range(n2):
    for j in range(i+1, n2):
        if (i,j) in bonded2:
            continue
        pos_i = np.array(conf2.GetAtomPosition(i))
        pos_j = np.array(conf2.GetAtomPosition(j))
        dist = np.linalg.norm(pos_i - pos_j)
        if dist < max_bond_len * 2.5:
            close2 += 1
        else:
            far2 += 1

print(f"Testosterone ({n2} atoms):")
print(f"  Non-bonded pairs: {close2 + far2}")
print(f"  Bondable (< {max_bond_len * 2.5:.2f}Å): {close2}")
print(f"  Too far: {far2}")
print(f"  Action space reduction: {far2/(close2+far2)*100:.0f}%")
