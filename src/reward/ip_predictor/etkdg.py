"""ETKDG 3D conformer embedding for the IP predictor.

Fallback (single process / non-SLURM) mode: an intra-rank
``ThreadPoolExecutor``. RDKit's ``EmbedMolecule`` releases the GIL, so threads
parallelize embedding across the rank's mols.
"""
import torch
from rdkit.Chem import AllChem
from concurrent.futures import ThreadPoolExecutor


class ETKDGEmbedder:
    """3D conformer embedding via ETKDG: intra-rank thread-pool."""

    def __init__(self, device, etkdg_threads):
        self.device = device
        self.etkdg_threads = int(etkdg_threads)
        self._etkdg_pool = (ThreadPoolExecutor(max_workers=self.etkdg_threads)
                            if self.etkdg_threads > 1 else None)

    def rwmol2data_atts(self, mols, maxAttempts):
        """
            return values:
            data:  the data for aimnet-nse
            valid: found a valid conformer or not
        """
        data = [dict() for _ in mols]
        success = [False for _ in mols]

        # Phase 1 (CPU, GIL-released): embed each mol's conformer.
        if self._etkdg_pool is not None and len(mols) > 1:
            # Per-mol seed (i+1): seed=-1 draws from RDKit's process-global RNG,
            # which is not thread-safe; concurrent workers would race on it.
            embeds = list(self._etkdg_pool.map(
                lambda im: self._embed_coords(im[1], maxAttempts, im[0] + 1),
                enumerate(mols)))
        else:
            embeds = [self._embed_coords(m, maxAttempts, i + 1)
                      for i, m in enumerate(mols)]

        # Phase 2 (serial, GPU): build tensors and move to device on main thread.
        for i, (coords_np, numbers_list, _p) in enumerate(embeds):
            if coords_np is not None:
                success[i] = True
                coords = torch.tensor(coords_np, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1).to(self.device)
                numbers = torch.tensor(numbers_list, dtype=torch.long).unsqueeze(0).repeat(3, 1).to(self.device)
                charge = torch.tensor([1, 0, -1]).to(self.device)  # cation, neutral, anion
                mult = torch.tensor([2, 1, 2]).to(self.device)
                data[i] = dict(coord=coords, numbers=numbers, charge=charge, mult=mult)
        return data, success

    def _embed_coords(self, mol, maxAttempts, seed):
        """CPU-only ETKDG embedding (thread-safe, no CUDA) for the fallback path.

        RDKit's ``EmbedMolecule(maxAttempts=N)`` already early-stops: it retries
        up to N times and returns on the FIRST success. ``seed`` (>= 0) selects a
        call-local RNG so concurrent workers don't race on RDKit's global one.
        ``prob`` is reduced to a success flag (debug-only, unused when the IP
        cache is off).
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
