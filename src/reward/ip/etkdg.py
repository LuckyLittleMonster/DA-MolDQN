"""ETKDG 3D conformer embedding for the IP predictor.

Fallback (single process / non-SLURM) mode: an intra-rank
``ThreadPoolExecutor``. RDKit's ``EmbedMolecule`` releases the GIL, so threads
parallelize embedding across the rank's mols.

Seed source (``deterministic_seed``):
  - False (default): seed = batch position ``i+1``. The SAME molecule at a
    different batch position draws a different conformer -> its IP is NOT
    reproducible across visits, which is why IP is not cached by default.
  - True: seed = crc32(canonical SMILES). A molecule's conformer (hence IP) is
    then a fixed function of its identity, so repeated/recomputed IP is bit-exact
    and IP becomes safely cacheable. (crc32 is used instead of Python ``hash`` —
    the latter is per-process randomized via PYTHONHASHSEED.)
"""
import zlib

from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent.futures import ThreadPoolExecutor


class ETKDGEmbedder:
    """3D conformer embedding via ETKDG: intra-rank thread-pool."""

    def __init__(self, device, etkdg_threads, deterministic_seed=False):
        self.device = device
        self.etkdg_threads = int(etkdg_threads)
        self.deterministic_seed = bool(deterministic_seed)
        self._etkdg_pool = (ThreadPoolExecutor(max_workers=self.etkdg_threads)
                            if self.etkdg_threads > 1 else None)

    def _seed_for(self, mol, i):
        """Per-molecule ETKDG seed. Position-based (i+1) or identity-based
        (crc32 of canonical SMILES) when ``deterministic_seed``. Always >= 0 so
        ``EmbedMolecule`` uses a call-local RNG (thread-safe)."""
        if self.deterministic_seed:
            return zlib.crc32(Chem.MolToSmiles(mol).encode()) % (2 ** 31)
        return i + 1

    def embed(self, mols, maxAttempts):
        """Embed one ETKDG conformer per molecule.

        Returns ``(embeds, success)`` parallel to ``mols``:
          - ``embeds[i]`` is ``(coords_np [natoms,3], numbers_list [natoms])`` for
            a successfully embedded molecule, else ``None``.
          - ``success[i]`` is the embed-success flag.
        The padded-batch tensor assembly (3 charge states per mol) is left to
        ``IPPredictor.predict_IP``.
        """
        seeds = [self._seed_for(m, i) for i, m in enumerate(mols)]

        # Phase 1 (CPU, GIL-released): embed each mol's conformer.
        if self._etkdg_pool is not None and len(mols) > 1:
            raw = list(self._etkdg_pool.map(
                lambda ms: self._embed_coords(ms[0], maxAttempts, ms[1]),
                zip(mols, seeds)))
        else:
            raw = [self._embed_coords(m, maxAttempts, s)
                   for m, s in zip(mols, seeds)]

        embeds = [None for _ in mols]
        success = [False for _ in mols]
        for i, (coords_np, numbers_list, _p) in enumerate(raw):
            if coords_np is not None:
                success[i] = True
                embeds[i] = (coords_np, numbers_list)
        return embeds, success

    def _embed_coords(self, mol, maxAttempts, seed):
        """CPU-only ETKDG embedding (thread-safe, no CUDA) for the fallback path.

        RDKit's ``EmbedMolecule(maxAttempts=N)`` already early-stops: it retries
        up to N times and returns on the FIRST success. ``seed`` (>= 0) selects a
        call-local RNG so concurrent workers don't race on RDKit's global one.
        """
        coords_np = None
        numbers_list = None
        cid = -1
        try:
            cid = AllChem.EmbedMolecule(mol, useRandomCoords=True,
                                        maxAttempts=maxAttempts, randomSeed=seed)
            if cid >= 0:
                coords_np = mol.GetConformer(cid).GetPositions()
                numbers_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
        except Exception as e:
            print(f"IP Exception: {e}")
        return coords_np, numbers_list, (1.0 if cid >= 0 else 0.0)
