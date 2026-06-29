"""Benchmark: reference-conformer ETKDG on realistic RL parent->child edits.

Motivation: in the RL loop each step's molecule is a small edit of the previous
one (an atom/bond added or changed), and IP's per-step ETKDG embedding is the
throughput bottleneck. Each child shares most of its heavy atoms with the parent,
so the parent's conformer can seed the child's embedding (via ``coordMap``).

This models that directly: build a child by appending a short carbon chain to a
parent (mirroring MolDQN-style atom additions; parent heavy-atom indices are
preserved 0..n-1 in the child, so the parent->child atom map is exact and free),
then embed the child (a) from scratch and (b) seeded by the parent's conformer for
the shared heavy atoms. The new atoms + all H's still embed fresh. We sweep the
edit size to see speedup vs the reused fraction.

Runnable benchmark: ``python tests/test_etkdg_reference.py``. Also a pytest module.
"""
import time

from rdkit import Chem
from rdkit.Chem import AllChem


PARENTS = {
    "ibuprofen (15)":  "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "start_mol (31)":  "CC(C)NCc1cccc(-c2cccc(-c3nc4cc(F)ccc4[nH]3)c2)c1",
    "imatinib (37)":   "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
}

EDIT_SIZES = [1, 2, 4, 8]   # heavy atoms added (small RL edit -> larger)
MAX_ATTEMPTS = 10
N_REPEAT = 40


def _grow(parent, k, anchor_pick=0):
    """Child = parent + a chain of k carbons on a free-valence heavy atom.

    Parent heavy-atom indices are preserved (0..n_parent-1) in the child, so the
    shared-atom map is the identity on those indices. Returns (child, anchor_idx).
    """
    anchors = [a.GetIdx() for a in parent.GetAtoms() if a.GetTotalNumHs() > 0]
    if not anchors:
        return None, None
    rw = Chem.RWMol(parent)
    anchor = anchors[anchor_pick % len(anchors)]
    prev = anchor
    for _ in range(k):
        idx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(prev, idx, Chem.BondType.SINGLE)
        prev = idx
    child = rw.GetMol()
    try:
        Chem.SanitizeMol(child)
    except Exception:
        return None, None
    return child, anchor


def _h_neighbors(mol, i):
    return [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetAtomicNum() == 1]


def _coord_map_with_hs(parent_h, child_h, n_parent_heavy, anchor):
    """Seed shared heavy atoms AND their H's from the parent conformer.

    The anchor lost an H (it now bonds the new chain), so its H's embed fresh;
    every other shared heavy atom keeps its H's (mapped parent->child in order).
    """
    conf = parent_h.GetConformer()
    cmap = {}
    for i in range(n_parent_heavy):
        cmap[i] = conf.GetAtomPosition(i)
        if i == anchor:
            continue
        for c_h, p_h in zip(_h_neighbors(child_h, i), _h_neighbors(parent_h, i)):
            cmap[c_h] = conf.GetAtomPosition(p_h)
    return cmap


def _embed_scratch(mol_noH, seed):
    mol = Chem.AddHs(mol_noH)
    cid = AllChem.EmbedMolecule(mol, useRandomCoords=True,
                                maxAttempts=MAX_ATTEMPTS, randomSeed=seed)
    return mol if cid >= 0 else None


def _embed_seeded(mol_noH, coord_map, seed):
    mol = Chem.AddHs(mol_noH)
    cid = AllChem.EmbedMolecule(mol, coordMap=coord_map,
                                maxAttempts=MAX_ATTEMPTS, randomSeed=seed)
    return mol if cid >= 0 else None


def _bench(fn, n=N_REPEAT):
    fn(0)  # warm up
    ok = 0
    t0 = time.perf_counter()
    for s in range(n):
        if fn(s) is not None:
            ok += 1
    return (time.perf_counter() - t0) / n * 1000.0, ok / n


def _parent_conformer(parent_noH, seed=0xC0FFEE):
    """Embed the parent once (the 'previous step' conformer). Heavy atoms keep
    indices 0..n-1 (AddHs appends H's after), so coordMap indices line up."""
    return _embed_scratch(parent_noH, seed)


def benchmark():
    print(f"\nRL parent->child reference-conformer ETKDG benchmark "
          f"(maxAttempts={MAX_ATTEMPTS}, {N_REPEAT} reps)\n")
    print("  scratch  = fresh ETKDG;  seed-heavy = reuse shared heavy-atom coords;"
          "  seed+H = also reuse their H coords\n")
    hdr = (f"{'parent':<17}{'+atoms':>7}{'reuse':>7}{'scratch':>9}"
           f"{'seed-heavy':>12}{'seed+H':>9}{'  speedup(heavy/+H)':>20}")
    print(hdr)
    print("-" * len(hdr))
    for name, smi in PARENTS.items():
        parent = Chem.MolFromSmiles(smi)
        n_parent = parent.GetNumAtoms()           # heavy atoms
        ref = _parent_conformer(parent)
        if ref is None:
            print(f"{name:<17}  (parent embed failed)")
            continue
        ref_conf = ref.GetConformer()
        cmap_heavy = {i: ref_conf.GetAtomPosition(i) for i in range(n_parent)}

        for k in EDIT_SIZES:
            child, anchor = _grow(parent, k)
            if child is None:
                continue
            reuse = n_parent / child.GetNumAtoms()
            cmap_full = _coord_map_with_hs(ref, Chem.AddHs(child), n_parent, anchor)

            scratch_ms, s_ok = _bench(lambda s: _embed_scratch(child, s))
            heavy_ms, h_ok = _bench(lambda s: _embed_seeded(child, cmap_heavy, s))
            full_ms, f_ok = _bench(lambda s: _embed_seeded(child, cmap_full, s))
            flag = "" if min(s_ok, h_ok, f_ok) == 1.0 else f"  (ok {s_ok:.0%}/{h_ok:.0%}/{f_ok:.0%})"
            print(f"{name:<17}{k:>7}{reuse:>6.0%}{scratch_ms:>9.2f}"
                  f"{heavy_ms:>12.2f}{full_ms:>9.2f}"
                  f"{scratch_ms / heavy_ms:>11.2f}x /{scratch_ms / full_ms:>6.2f}x{flag}")
        print()


# --- pytest: seeded embedding on a real edit must produce a valid conformer ---

def test_seeded_embed_on_edit_produces_conformer():
    parent = Chem.MolFromSmiles(PARENTS["start_mol (31)"])
    ref = _parent_conformer(parent)
    assert ref is not None
    coord_map = {i: ref.GetConformer().GetAtomPosition(i)
                 for i in range(parent.GetNumAtoms())}
    child, _ = _grow(parent, 3)
    assert child is not None
    out = _embed_seeded(child, coord_map, seed=1)
    assert out is not None and out.GetNumConformers() == 1


def test_grow_preserves_parent_indices():
    parent = Chem.MolFromSmiles(PARENTS["ibuprofen (15)"])
    child, _ = _grow(parent, 4)
    assert child.GetNumAtoms() == parent.GetNumAtoms() + 4
    # the first n_parent atoms keep their element (indices preserved)
    for i in range(parent.GetNumAtoms()):
        assert child.GetAtomWithIdx(i).GetAtomicNum() == parent.GetAtomWithIdx(i).GetAtomicNum()


if __name__ == "__main__":
    benchmark()
