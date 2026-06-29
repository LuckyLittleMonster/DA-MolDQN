"""IP-path prototype: reference-conformer ETKDG over a real RL episode chain.

Unlike test_etkdg_reference.py (synthetic carbon-chain edits), this drives the
ACTUAL cenv action space: from a start molecule it walks real valid actions
(the RL transitions), and at each step embeds the new molecule both from scratch
(the current IP path) and seeded by the previous step's conformer.

Key enabler (verified): cenv preserves parent atom indices in its children, so the
parent->child heavy-atom map is the identity on 0..min(n_parent,n_child)-1 — no MCS
needed. We seed each shared heavy atom (where its element is unchanged) plus its
H's (where its H-count is unchanged); the edit site + new atoms embed fresh.

Runnable: ``python tests/test_ip_etkdg_reference.py``. Also a pytest module.
"""
import os
import random
import sys
import time

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import cxx_environment


START_SMILES = [
    "CC(C)NCc1cccc(-c2cccc(-c3nc4cc(F)ccc4[nH]3)c2)c1",   # start_molecule (27 heavy)
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",  # imatinib-like
]

MAX_ATTEMPTS = 10
N_STEPS = 10
N_EPISODES = 5
FP_MODE = 1        # 'list'
MAINTAIN_OH = -1   # 'exist'


def _h_neighbors(mol, i):
    return [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetAtomicNum() == 1]


def _coord_map(parent_h, child_h, n_shared):
    """Identity map on shared heavy atoms 0..n_shared-1 (cenv preserves order):
    seed each whose element is unchanged, plus its H's where the H-count matches."""
    conf = parent_h.GetConformer()
    cmap = {}
    for i in range(n_shared):
        if parent_h.GetAtomWithIdx(i).GetAtomicNum() != child_h.GetAtomWithIdx(i).GetAtomicNum():
            continue
        cmap[i] = conf.GetAtomPosition(i)
        p_hs, c_hs = _h_neighbors(parent_h, i), _h_neighbors(child_h, i)
        if len(p_hs) == len(c_hs):
            for c_h, p_h in zip(c_hs, p_hs):
                cmap[c_h] = conf.GetAtomPosition(p_h)
    return cmap


def _embed(mol_noH, seed, coord_map=None):
    mol = Chem.AddHs(mol_noH)
    if coord_map is None:
        cid = AllChem.EmbedMolecule(mol, useRandomCoords=True,
                                    maxAttempts=MAX_ATTEMPTS, randomSeed=seed)
    else:
        cid = AllChem.EmbedMolecule(mol, coordMap=coord_map,
                                    maxAttempts=MAX_ATTEMPTS, randomSeed=seed)
    return (mol, True) if cid >= 0 else (None, False)


def run_episode(start_smi, n_steps, rng_seed):
    """Walk real cenv transitions; embed each step scratch vs seeded. Returns
    (scratch_ms, seeded_ms, map_ms, steps, scratch_ok, seeded_ok)."""
    rng = random.Random(rng_seed)
    mol = Chem.MolFromSmiles(start_smi)
    parent_h, _ = _embed(mol, seed=1)               # the previous-step conformer

    scratch_ms = seeded_ms = map_ms = fallback_ms = 0.0
    steps = scratch_ok = seeded_ok = 0
    for st in range(n_steps):
        vas, _ = cxx_environment.get_valid_actions_and_fingerprint(mol, FP_MODE, MAINTAIN_OH)
        # drop the no-op action (the molecule itself), pick a real edit
        cands = [c for c in vas if c.GetNumAtoms() != mol.GetNumAtoms()] or vas
        child = rng.choice(cands)
        child_h = Chem.AddHs(child)
        n_shared = min(mol.GetNumAtoms(), child.GetNumAtoms())

        t = time.perf_counter()
        cmap = _coord_map(parent_h, child_h, n_shared)
        map_ms += (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        s_mol, s_ok = _embed(child, seed=st)
        scratch_step = (time.perf_counter() - t) * 1000
        scratch_ms += scratch_step

        t = time.perf_counter()
        cid = AllChem.EmbedMolecule(child_h, coordMap=cmap,
                                    maxAttempts=MAX_ATTEMPTS, randomSeed=st)
        seeded_ms += (time.perf_counter() - t) * 1000
        r_ok = cid >= 0
        if not r_ok:
            fallback_ms += scratch_step   # realistic integration retries from scratch

        steps += 1
        scratch_ok += s_ok
        seeded_ok += r_ok
        # advance: child becomes the new state; its seeded conformer is the next ref
        mol = child
        parent_h = child_h if r_ok else (s_mol if s_ok else parent_h)
    return scratch_ms, seeded_ms, map_ms, fallback_ms, steps, scratch_ok, seeded_ok


def benchmark():
    print(f"\nIP ETKDG: scratch vs reference-conformer over real cenv episodes "
          f"(maxAttempts={MAX_ATTEMPTS}, {N_STEPS} steps x {N_EPISODES} eps)\n")
    hdr = (f"{'start mol (heavy)':<22}{'scratch':>10}{'seeded':>9}{'seeded%':>9}"
           f"{'net+fallback':>14}{'raw spd':>9}{'net spd':>9}")
    print(hdr)
    print("-" * len(hdr))
    for smi in START_SMILES:
        n_heavy = Chem.MolFromSmiles(smi).GetNumAtoms()
        sc = se = mp = fb = n = sok = rok = 0
        for ep in range(N_EPISODES):
            a, b, m, f, k, so, ro = run_episode(smi, N_STEPS, rng_seed=ep)
            sc += a; se += b; mp += m; fb += f; n += k; sok += so; rok += ro
        sc_s, se_s, mp_s, fb_s = sc / n, se / n, mp / n, fb / n
        raw = sc_s / (se_s + mp_s)
        net = sc_s / (se_s + mp_s + fb_s)        # seeded + map + scratch-retry on failures
        label = f"{smi[:14]}.. ({n_heavy})"
        print(f"{label:<22}{sc_s:>10.2f}{se_s:>9.2f}{rok/n:>8.0%}"
              f"{se_s + mp_s + fb_s:>14.2f}{raw:>8.2f}x{net:>8.2f}x")
    print("\nms/step. raw = seeded+map; net = +scratch retry on the (1-seeded%) failures.")
    print("cenv gives the parent->child atom map free (map cost ~0.2 ms).\n")


# --- pytest: seeded embedding works on a real cenv transition ---

def test_seeded_embed_on_real_cenv_transition():
    mol = Chem.MolFromSmiles(START_SMILES[0])
    parent_h, ok = _embed(mol, seed=1)
    assert ok
    vas, _ = cxx_environment.get_valid_actions_and_fingerprint(mol, FP_MODE, MAINTAIN_OH)
    child = next(c for c in vas if c.GetNumAtoms() >= mol.GetNumAtoms())
    child_h = Chem.AddHs(child)
    cmap = _coord_map(parent_h, child_h, min(mol.GetNumAtoms(), child.GetNumAtoms()))
    assert len(cmap) > 0
    cid = AllChem.EmbedMolecule(child_h, coordMap=cmap,
                                maxAttempts=MAX_ATTEMPTS, randomSeed=0)
    assert cid >= 0 and child_h.GetNumConformers() == 1


if __name__ == "__main__":
    benchmark()
