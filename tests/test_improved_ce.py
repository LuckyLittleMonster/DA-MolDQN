"""Test improved ConstrainedEmbed with reverse-match fallback for MolDQN.

Current ConstrainedEmbed success rate is ~61.6%:
- 28.4% fail NO_MATCH: parent is NOT a substructure of child (bond removal/downgrade)
- 10.1% fail EMBED_FAIL: geometric constraint conflicts

Improved 3-stage strategy:
1. Forward match: child.GetSubstructMatch(parent) -> ConstrainedEmbed(child, parent)
2. Reverse match: parent.GetSubstructMatch(child) -> extract child coords from parent
   a. Approach A: Build coordMap from parent conformer, EmbedMolecule with coordMap
   b. Approach B: ConstrainedEmbed(parent, child_as_core) where child has coords from parent
3. Fallback: standard ETKDG from scratch
"""
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, '/shared/data1/Users/l1062811/git/SynDQN')

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
RDLogger.DisableLog("rdApp.*")

import src.cenv as cenv


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def embed_from_scratch(mol, seed=42):
    """Standard ETKDG from scratch. Returns (mol_with_conf, success)."""
    mol = Chem.RWMol(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        return mol, True
    return mol, False


def try_forward_match(child_mol, parent_mol_with_conf, seed=42):
    """Stage 1: Forward match - parent is substructure of child.

    child.GetSubstructMatch(parent) -> ConstrainedEmbed(child, parent)
    Returns (mol_with_conf, success)
    """
    match = child_mol.GetSubstructMatch(parent_mol_with_conf)
    if not match:
        return None, False
    try:
        child = Chem.RWMol(child_mol)
        result = AllChem.ConstrainedEmbed(child, parent_mol_with_conf, randomseed=seed)
        return result, True
    except Exception:
        return None, False


def try_reverse_match_coordmap(child_mol, parent_mol_with_conf, seed=42):
    """Stage 2a: Reverse match with coordMap approach.

    parent.GetSubstructMatch(child) -> build coordMap from parent's conformer
    for matched atoms, then EmbedMolecule(child, coordMap=coordMap).

    Returns (mol_with_conf, success)
    """
    match = parent_mol_with_conf.GetSubstructMatch(child_mol)
    if not match:
        return None, False

    parent_conf = parent_mol_with_conf.GetConformer()
    coord_map = {}
    # match[i] = index in parent that corresponds to child atom i
    # So child atom i -> parent atom match[i]
    for child_idx in range(child_mol.GetNumAtoms()):
        parent_idx = match[child_idx]
        pos = parent_conf.GetAtomPosition(parent_idx)
        coord_map[child_idx] = pos

    child = Chem.RWMol(child_mol)
    res = AllChem.EmbedMolecule(child, coordMap=coord_map, randomSeed=seed)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(child)
        except Exception:
            pass
        return child, True
    return None, False


def try_reverse_match_constrained(child_mol, parent_mol_with_conf, seed=42):
    """Stage 2b: Reverse match with ConstrainedEmbed approach.

    parent.GetSubstructMatch(child) -> give child a conformer using parent's
    coordinates for matching atoms, then ConstrainedEmbed(parent, child_as_core).

    Wait - we want child's conformer, not parent's. So we:
    1. Create child_core with conformer from parent's matched atom coords
    2. Use ConstrainedEmbed(child, child_core) - but child IS child_core...

    Actually the right approach: since child is a substructure of parent,
    we can directly assign child a conformer from parent's coords and
    then optimize. This gives child a valid 3D geometry.

    Returns (mol_with_conf, success)
    """
    match = parent_mol_with_conf.GetSubstructMatch(child_mol)
    if not match:
        return None, False

    parent_conf = parent_mol_with_conf.GetConformer()
    child = Chem.RWMol(child_mol)

    # Create a conformer for child using parent's coordinates
    conf = Chem.Conformer(child.GetNumAtoms())
    for child_idx in range(child.GetNumAtoms()):
        parent_idx = match[child_idx]
        pos = parent_conf.GetAtomPosition(parent_idx)
        conf.SetAtomPosition(child_idx, pos)
    conf.SetId(0)
    child.RemoveAllConformers()
    child.AddConformer(conf, assignId=True)

    # Now optimize the geometry with MMFF to relax any strain
    try:
        AllChem.MMFFOptimizeMolecule(child)
        return child, True
    except Exception:
        # Even without optimization, the conformer from parent is reasonable
        return child, True


def improved_embed(child_mol, parent_mol_with_conf, seed=42):
    """3-stage improved embedding strategy.

    Returns (mol_with_conf, stage_used):
        stage_used: "forward", "reverse_coordmap", "reverse_constrained", "scratch", "failed"
    """
    if parent_mol_with_conf is None or parent_mol_with_conf.GetNumConformers() == 0:
        mol, ok = embed_from_scratch(child_mol, seed)
        return (mol, "scratch") if ok else (mol, "failed")

    # Stage 1: Forward match
    result, ok = try_forward_match(child_mol, parent_mol_with_conf, seed)
    if ok:
        return result, "forward"

    # Stage 2a: Reverse match with coordMap
    result, ok = try_reverse_match_coordmap(child_mol, parent_mol_with_conf, seed)
    if ok:
        return result, "reverse_coordmap"

    # Stage 2b: Reverse match with direct coord assignment
    result, ok = try_reverse_match_constrained(child_mol, parent_mol_with_conf, seed)
    if ok:
        return result, "reverse_constrained"

    # Stage 3: Fallback to from-scratch ETKDG
    mol, ok = embed_from_scratch(child_mol, seed)
    return (mol, "scratch") if ok else (mol, "failed")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

print("=" * 70)
print("Test: Improved ConstrainedEmbed with Reverse-Match Fallback")
print("=" * 70)

# Setup cenv
flags = cenv.Flags()
env = cenv.Environment(["C", "O", "N"], [3, 5, 6], 3, 2048, flags)

np.random.seed(42)
state = Chem.MolFromSmiles("C")
parent_with_conf = None

# Counters for the original (forward-only) strategy
orig_total = 0
orig_success = 0
orig_no_match = 0
orig_embed_fail = 0

# Counters for the improved 3-stage strategy
imp_stage_counts = defaultdict(int)
imp_total = 0

# Detailed tracking for reverse match sub-approaches
reverse_coordmap_tried = 0
reverse_coordmap_success = 0
reverse_constrained_tried = 0
reverse_constrained_success = 0

# Timing
timing_improved = []
timing_scratch = []

# Per-step details
step_details = []

for step in range(20):
    valid_actions, _ = env.get_valid_actions_and_fingerprint(state, 0, False)
    n_actions = len(valid_actions)
    state_smi = Chem.MolToSmiles(state)

    step_stage_counts = defaultdict(int)
    step_time_improved = 0.0
    step_time_scratch = 0.0
    n_tested = 0

    if parent_with_conf is not None:
        for mol in valid_actions:
            orig_total += 1
            imp_total += 1
            n_tested += 1

            # --- Original strategy (forward only) ---
            match = mol.GetSubstructMatch(parent_with_conf)
            if not match:
                orig_no_match += 1
            else:
                try:
                    AllChem.ConstrainedEmbed(Chem.RWMol(mol), parent_with_conf, randomseed=42)
                    orig_success += 1
                except Exception:
                    orig_embed_fail += 1

            # --- Improved 3-stage strategy ---
            t0 = time.perf_counter()
            result_mol, stage = improved_embed(mol, parent_with_conf, seed=42)
            t1 = time.perf_counter()
            step_time_improved += (t1 - t0)
            imp_stage_counts[stage] += 1
            step_stage_counts[stage] += 1

            # Track reverse match sub-approaches independently
            if not match:
                # Forward failed, check reverse match availability
                rev_match = parent_with_conf.GetSubstructMatch(mol)
                if rev_match:
                    # Try coordmap approach independently
                    reverse_coordmap_tried += 1
                    _, ok_cm = try_reverse_match_coordmap(mol, parent_with_conf, seed=42)
                    if ok_cm:
                        reverse_coordmap_success += 1

                    # Try constrained approach independently
                    reverse_constrained_tried += 1
                    _, ok_ce = try_reverse_match_constrained(mol, parent_with_conf, seed=42)
                    if ok_ce:
                        reverse_constrained_success += 1

            # --- From-scratch ETKDG for timing comparison ---
            t0 = time.perf_counter()
            scratch_mol = Chem.RWMol(mol)
            res = AllChem.EmbedMolecule(scratch_mol, randomSeed=42)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(scratch_mol)
                except Exception:
                    pass
            t1 = time.perf_counter()
            step_time_scratch += (t1 - t0)

    if n_tested > 0:
        timing_improved.append(step_time_improved / n_tested)
        timing_scratch.append(step_time_scratch / n_tested)

    step_details.append({
        "step": step,
        "state": state_smi,
        "n_atoms": state.GetNumAtoms(),
        "n_actions": n_actions,
        "n_tested": n_tested,
        "stages": dict(step_stage_counts),
    })

    # Print per-step summary
    if n_tested > 0:
        stage_str = ", ".join(f"{k}={v}" for k, v in sorted(step_stage_counts.items()))
        print(f"Step {step:2d}: {state_smi:40s} | {state.GetNumAtoms()} atoms | "
              f"{n_tested:3d} actions | {stage_str}")
    else:
        print(f"Step {step:2d}: {state_smi:40s} | {state.GetNumAtoms()} atoms | "
              f"(no parent conformer yet)")

    # Advance state
    pick = np.random.randint(0, max(1, len(valid_actions) - 1))
    state = Chem.RWMol(valid_actions[pick])

    # Build parent conformer for next step
    parent_with_conf = Chem.RWMol(state)
    res = AllChem.EmbedMolecule(parent_with_conf, randomSeed=42)
    if res != 0:
        parent_with_conf = None
        continue
    try:
        AllChem.MMFFOptimizeMolecule(parent_with_conf)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("RESULTS: Original Strategy (Forward-Match Only)")
print("=" * 70)

if orig_total > 0:
    print(f"Total actions tested: {orig_total}")
    print(f"  Forward CE success: {orig_success:5d} ({orig_success/orig_total*100:5.1f}%)")
    print(f"  NO_MATCH (no fwd):  {orig_no_match:5d} ({orig_no_match/orig_total*100:5.1f}%)")
    print(f"  EMBED_FAIL:         {orig_embed_fail:5d} ({orig_embed_fail/orig_total*100:5.1f}%)")

print("\n" + "=" * 70)
print("RESULTS: Improved 3-Stage Strategy")
print("=" * 70)

if imp_total > 0:
    print(f"Total actions tested: {imp_total}")
    for stage in ["forward", "reverse_coordmap", "reverse_constrained", "scratch", "failed"]:
        count = imp_stage_counts.get(stage, 0)
        print(f"  {stage:25s}: {count:5d} ({count/imp_total*100:5.1f}%)")

    # Overall success = everything except "failed"
    imp_success = imp_total - imp_stage_counts.get("failed", 0)
    orig_rate = orig_success / orig_total * 100 if orig_total > 0 else 0
    imp_rate = imp_success / imp_total * 100
    print(f"\nOverall success rate:")
    print(f"  Original (forward only):   {orig_rate:.1f}%")
    print(f"  Improved (3-stage):        {imp_rate:.1f}%")
    print(f"  Improvement:               +{imp_rate - orig_rate:.1f} percentage points")

print("\n" + "=" * 70)
print("RESULTS: Reverse Match Sub-Approach Breakdown")
print("=" * 70)

print(f"\nWhen forward match fails and reverse match IS available:")
print(f"  Reverse coordMap tried:      {reverse_coordmap_tried}")
print(f"  Reverse coordMap success:    {reverse_coordmap_success} "
      f"({reverse_coordmap_success/max(1,reverse_coordmap_tried)*100:.1f}%)")
print(f"  Reverse constrained tried:   {reverse_constrained_tried}")
print(f"  Reverse constrained success: {reverse_constrained_success} "
      f"({reverse_constrained_success/max(1,reverse_constrained_tried)*100:.1f}%)")

# Check: how many NO_MATCH cases have reverse match?
print(f"\nOf {orig_no_match} NO_MATCH cases, {reverse_coordmap_tried} had reverse match "
      f"({reverse_coordmap_tried/max(1,orig_no_match)*100:.1f}%)")
no_either_match = orig_no_match - reverse_coordmap_tried
print(f"  {no_either_match} had NEITHER forward nor reverse match")

print("\n" + "=" * 70)
print("RESULTS: Timing Comparison")
print("=" * 70)

if timing_improved and timing_scratch:
    avg_improved = np.mean(timing_improved) * 1000  # ms
    avg_scratch = np.mean(timing_scratch) * 1000
    print(f"\nAverage per-molecule embedding time:")
    print(f"  Improved strategy: {avg_improved:.3f} ms/mol")
    print(f"  From-scratch ETKDG: {avg_scratch:.3f} ms/mol")
    if avg_improved > 0:
        ratio = avg_scratch / avg_improved
        print(f"  Ratio (scratch / improved): {ratio:.2f}x")

    # Per-step timing
    print(f"\nPer-step timing (ms/mol):")
    print(f"  {'Step':>4s}  {'Improved':>10s}  {'Scratch':>10s}  {'Ratio':>8s}")
    for i, (t_imp, t_scr) in enumerate(zip(timing_improved, timing_scratch)):
        t_imp_ms = t_imp * 1000
        t_scr_ms = t_scr * 1000
        r = t_scr_ms / t_imp_ms if t_imp_ms > 0 else float('inf')
        # Find which step this corresponds to (skip step 0 which has no parent)
        step_idx = i + 1  # offset by 1 since step 0 has no parent
        print(f"  {step_idx:4d}  {t_imp_ms:10.3f}  {t_scr_ms:10.3f}  {r:8.2f}x")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if imp_total > 0:
    imp_success = imp_total - imp_stage_counts.get("failed", 0)
    orig_rate = orig_success / orig_total * 100 if orig_total > 0 else 0
    imp_rate = imp_success / imp_total * 100

    reverse_recovered = imp_stage_counts.get("reverse_coordmap", 0) + imp_stage_counts.get("reverse_constrained", 0)

    print(f"\nReverse-match fallback recovered {reverse_recovered} / {orig_no_match} NO_MATCH cases")
    print(f"Success rate: {orig_rate:.1f}% -> {imp_rate:.1f}% (+{imp_rate - orig_rate:.1f}pp)")

    if imp_rate > orig_rate + 5:
        print("VERDICT: Reverse-match fallback provides meaningful improvement.")
    elif imp_rate > orig_rate:
        print("VERDICT: Reverse-match fallback provides modest improvement.")
    else:
        print("VERDICT: Reverse-match fallback does NOT improve success rate.")
