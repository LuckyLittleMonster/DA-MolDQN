"""Profile incremental ETKDG: reuse parent conformer coords for child molecules.

In MolDQN, each valid action differs from parent by 1 edit (add atom/bond, remove bond, etc).
If we seed ETKDG with parent coordinates via coordMap, the solver should converge faster.
"""
import time
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
RDLogger.DisableLog("rdApp.*")

import sys
sys.path.insert(0, '/shared/data1/Users/l1062811/git/SynDQN')
import src.cenv as cenv

def get_parent_coordmap(parent_mol_h, child_mol_h):
    """Build coordMap: map matching atoms in child to parent's 3D coords.
    
    Returns dict {child_atom_idx: Point3D} for atoms that exist in parent.
    """
    from rdkit.Geometry import Point3D
    # Match parent substructure in child
    match = child_mol_h.GetSubstructMatch(parent_mol_h)
    if not match:
        # Try without H
        parent_no_h = Chem.RemoveHs(parent_mol_h)
        child_no_h = Chem.RemoveHs(child_mol_h)
        match_no_h = child_no_h.GetSubstructMatch(parent_no_h)
        if not match_no_h:
            return None
        # Map back to H-containing mol indices
        # This is complex, fall back to no coordMap
        return None
    
    conf = parent_mol_h.GetConformer()
    coord_map = {}
    for parent_idx, child_idx in enumerate(match):
        pos = conf.GetAtomPosition(parent_idx)
        coord_map[child_idx] = pos
    return coord_map


def embed_from_scratch(mol_h, seed=42):
    """Standard ETKDG from scratch."""
    res = AllChem.EmbedMolecule(mol_h, randomSeed=seed)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h)
        except:
            pass
    return res


def embed_incremental(mol_h, coord_map, seed=42):
    """ETKDG with parent coordinates as seed via coordMap."""
    if coord_map is None:
        return embed_from_scratch(mol_h, seed)
    res = AllChem.EmbedMolecule(mol_h, coordMap=coord_map, randomSeed=seed)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h)
        except:
            pass
    return res


def embed_constrained(child_mol, parent_mol_with_conf):
    """Use ConstrainedEmbed: force child to match parent substructure geometry."""
    try:
        child = Chem.RWMol(child_mol)
        AllChem.ConstrainedEmbed(child, parent_mol_with_conf)
        return child
    except Exception as e:
        return None


# Setup cenv
flags = cenv.Flags()
env = cenv.Environment(["C", "O", "N"], [3, 5, 6], 3, 2048, flags)

# Build a sequence of molecules (simulating MolDQN episode)
init_mol = Chem.MolFromSmiles("C")
state = Chem.RWMol(init_mol)

print("=== Incremental ETKDG Feasibility Test ===\n")

# Step through several MolDQN steps, measuring ETKDG with and without coordMap
episode_scratch = []
episode_incremental = []
episode_constrained = []
parent_mol_h = None
parent_conf = None

for step in range(15):
    valid_actions, _ = env.get_valid_actions_and_fingerprint(state, 0, False)
    n_actions = len(valid_actions)
    
    state_smi = Chem.MolToSmiles(state)
    print(f"Step {step:2d}: {state_smi:40s} ({state.GetNumAtoms()} atoms, {n_actions} actions)")
    
    # Measure from-scratch ETKDG for all actions
    t_scratch = []
    for mol in valid_actions:
        mol_h = AllChem.AddHs(Chem.RWMol(mol))
        t0 = time.perf_counter()
        embed_from_scratch(mol_h)
        t_scratch.append(time.perf_counter() - t0)
    avg_scratch = np.mean(t_scratch) * 1000
    
    # Measure incremental ETKDG (coordMap from parent)
    t_incr = []
    t_constrained = []
    n_matched = 0
    n_constrained_ok = 0
    
    if parent_mol_h is not None and parent_conf is not None:
        for mol in valid_actions:
            mol_h = AllChem.AddHs(Chem.RWMol(mol))
            cmap = get_parent_coordmap(parent_mol_h, mol_h)
            t0 = time.perf_counter()
            embed_incremental(mol_h, cmap)
            t_incr.append(time.perf_counter() - t0)
            if cmap is not None:
                n_matched += 1
            
            # Also try ConstrainedEmbed
            t0 = time.perf_counter()
            result = embed_constrained(mol, parent_conf)
            t_constrained.append(time.perf_counter() - t0)
            if result is not None:
                n_constrained_ok += 1
    
    avg_incr = np.mean(t_incr) * 1000 if t_incr else float('nan')
    avg_constr = np.mean(t_constrained) * 1000 if t_constrained else float('nan')
    
    episode_scratch.append(avg_scratch)
    episode_incremental.append(avg_incr)
    episode_constrained.append(avg_constr)
    
    if t_incr:
        speedup = avg_scratch / avg_incr if avg_incr > 0 else 0
        print(f"  Scratch: {avg_scratch:.2f}ms/mol | CoordMap: {avg_incr:.2f}ms/mol "
              f"(match {n_matched}/{n_actions}, {speedup:.2f}x)")
        print(f"  ConstrainedEmbed: {avg_constr:.2f}ms/mol (ok {n_constrained_ok}/{n_actions})")
    else:
        print(f"  Scratch: {avg_scratch:.2f}ms/mol | (no parent yet)")
    
    # Pick an action (not last = not no-modification) and advance
    pick = np.random.randint(0, max(1, n_actions - 1))
    state = Chem.RWMol(valid_actions[pick])
    
    # Generate parent conformer for next step
    parent_mol_h = AllChem.AddHs(Chem.RWMol(state))
    embed_from_scratch(parent_mol_h)
    parent_conf = Chem.RWMol(state)
    try:
        AllChem.EmbedMolecule(AllChem.AddHs(parent_conf), randomSeed=42)
        parent_conf_h = AllChem.AddHs(parent_conf)
        AllChem.EmbedMolecule(parent_conf_h, randomSeed=42)
        try:
            AllChem.MMFFOptimizeMolecule(parent_conf_h)
        except:
            pass
        # Copy conformer back to non-H mol for ConstrainedEmbed
        parent_conf = Chem.RWMol(parent_conf)
        AllChem.EmbedMolecule(parent_conf, randomSeed=42)
    except:
        parent_conf = None

print(f"\n=== Summary ===")
valid_scratch = [x for x in episode_scratch if not np.isnan(x)]
valid_incr = [x for x in episode_incremental if not np.isnan(x)]
valid_constr = [x for x in episode_constrained if not np.isnan(x)]
print(f"Avg from-scratch:     {np.mean(valid_scratch):.2f}ms/mol")
if valid_incr:
    print(f"Avg coordMap:         {np.mean(valid_incr):.2f}ms/mol ({np.mean(valid_scratch)/np.mean(valid_incr):.2f}x)")
if valid_constr:
    print(f"Avg ConstrainedEmbed: {np.mean(valid_constr):.2f}ms/mol ({np.mean(valid_scratch)/np.mean(valid_constr):.2f}x)")
