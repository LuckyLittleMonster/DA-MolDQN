"""Uni-Dock batch docking scorer for RL reward computation.

Converts SMILES → 3D SDF via RDKit, runs Uni-Dock GPU batch docking,
parses Vina scores from output SDF files.  Maintains a SMILES→score
cache to avoid redundant docking.
"""

import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers


def _generate_sdf_worker(args):
    """Worker for parallel 3D conformer generation (runs in mp.Pool).

    Uses ETKDGv3-equivalent settings with maxAttempts=7 to fail fast
    on difficult molecules (most succeed on attempt 1; rare cases need
    200-300 attempts and are not worth waiting for).
    """
    smi, tmpdir, fname, seed, max_attempts = args
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return fname, None

    try:
        mol = Chem.AddHs(mol)

        # ETKDGv3-equivalent keyword args + maxAttempts for fast failure.
        # EmbedParameters (ETKDGv3) doesn't expose maxAttempts, so we
        # use the keyword-based API and replicate ETKDGv3 settings.
        embed_kwargs = dict(
            maxAttempts=max_attempts,
            randomSeed=seed,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            useMacrocycleTorsions=True,
            useSmallRingTorsions=False,
            useMacrocycle14config=True,
            ETversion=2,
            enforceChirality=True,
        )
        status = AllChem.EmbedMolecule(mol, **embed_kwargs)
        if status != 0:
            # Fallback: random coordinates
            embed_kwargs['useRandomCoords'] = True
            status = AllChem.EmbedMolecule(mol, **embed_kwargs)
            if status != 0:
                return fname, None

        if mol.GetNumHeavyAtoms() <= 50:
            try:
                rdForceFieldHelpers.MMFFOptimizeMolecule(mol, maxIters=100)
            except Exception:
                pass

        sdf_path = os.path.join(tmpdir, f'{fname}.sdf')
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        return fname, sdf_path
    except Exception:
        return fname, None


# Default unidock binary path
_UNIDOCK_BIN = os.environ.get(
    'UNIDOCK_BIN',
    '/home/l1062811/data/envs/rl4/bin/unidock',
)

# Penalty score returned when 3D generation or docking fails
PENALTY_SCORE = 0.0  # kcal/mol (neutral — real scores are negative)


class UniDockScorer:
    """Batch molecular docking scorer using Uni-Dock GPU.

    Parameters
    ----------
    receptor_pdbqt : str
        Path to receptor PDBQT file.
    center_x, center_y, center_z : float
        Docking box center coordinates (Angstrom).
    size_x, size_y, size_z : float
        Docking box dimensions (Angstrom).
    search_mode : str
        Uni-Dock search mode: 'fast', 'balance', or 'detail'.
    seed : int
        Random seed for reproducibility.
    scoring : str
        Scoring function: 'vina', 'vinardo', or 'ad4'.
    num_modes : int
        Number of binding modes to generate.
    verbosity : int
        Uni-Dock verbosity (0=silent, 1=normal).
    """

    def __init__(self, receptor_pdbqt, center_x, center_y, center_z,
                 size_x=22.5, size_y=22.5, size_z=22.5,
                 search_mode='fast', seed=42, scoring='vina',
                 num_modes=1, verbosity=0, num_workers=0,
                 max_embed_attempts=7):
        self.receptor_pdbqt = os.path.abspath(receptor_pdbqt)
        if not os.path.isfile(self.receptor_pdbqt):
            raise FileNotFoundError(
                f"Receptor PDBQT not found: {self.receptor_pdbqt}")

        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.search_mode = search_mode
        self.seed = seed
        self.scoring = scoring
        self.num_modes = num_modes
        self.verbosity = verbosity
        # 0 = match batch size (one worker per molecule)
        self.num_workers = num_workers or os.cpu_count() or 64
        self.max_embed_attempts = max_embed_attempts

        # SMILES → score cache
        self._cache = {}
        # Timing stats
        self._timing = {'sdf': 0.0, 'dock': 0.0, 'parse': 0.0, 'calls': 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_dock(self, smiles_list):
        """Dock a batch of SMILES and return Vina scores.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to dock.

        Returns
        -------
        list[float]
            Vina docking scores (kcal/mol).  Negative = better binding.
            Returns PENALTY_SCORE for failures.
        """
        if not smiles_list:
            return []

        # Separate cached vs uncached
        uncached_smiles = []
        uncached_indices = []
        for i, smi in enumerate(smiles_list):
            canon = self._canonicalize(smi)
            if canon not in self._cache:
                uncached_smiles.append(canon)
                uncached_indices.append(i)

        # Dock uncached molecules
        if uncached_smiles:
            unique_smiles = list(dict.fromkeys(uncached_smiles))
            new_scores = self._dock_batch(unique_smiles)
            for smi, score in zip(unique_smiles, new_scores):
                self._cache[smi] = score

        # Assemble results
        results = []
        for smi in smiles_list:
            canon = self._canonicalize(smi)
            results.append(self._cache.get(canon, PENALTY_SCORE))
        return results

    @property
    def cache_size(self):
        return len(self._cache)

    def clear_cache(self):
        self._cache.clear()

    @property
    def timing_summary(self):
        """Return cumulative timing breakdown."""
        t = self._timing
        total = t['sdf'] + t['dock'] + t['parse']
        return (f"UniDockScorer timing ({t['calls']} calls): "
                f"SDF={t['sdf']:.1f}s, Dock={t['dock']:.1f}s, "
                f"Parse={t['parse']:.1f}s, Total={total:.1f}s")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _canonicalize(smi):
        """Canonicalize SMILES for cache consistency."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol)

    def _dock_batch(self, smiles_list):
        """Run Uni-Dock on a batch of unique SMILES.

        Uses parallel 3D conformer generation followed by GPU batch docking.
        Returns list of scores aligned with input smiles_list.
        """
        n = len(smiles_list)
        scores = [PENALTY_SCORE] * n

        with tempfile.TemporaryDirectory(prefix='unidock_') as tmpdir:
            # Step 1: Parallel SMILES → 3D SDF
            t0 = time.perf_counter()
            sdf_paths = []
            smi_to_filename = {}

            if n <= 4 or self.num_workers <= 1:
                # Sequential for small batches
                for idx, smi in enumerate(smiles_list):
                    fname = f"mol_{idx:04d}"
                    sdf_path = self._smiles_to_sdf(smi, tmpdir, fname)
                    if sdf_path is not None:
                        sdf_paths.append(sdf_path)
                        smi_to_filename[smi] = fname
            else:
                # Parallel 3D generation
                args_list = [
                    (smi, tmpdir, f"mol_{idx:04d}", self.seed,
                     self.max_embed_attempts)
                    for idx, smi in enumerate(smiles_list)
                ]
                pool = mp.Pool(min(self.num_workers, n))
                async_results = []
                for a in args_list:
                    ar = pool.apply_async(_generate_sdf_worker, (a,))
                    async_results.append((a[0], a[2], ar))  # smi, fname, ar
                pool.close()

                for smi, fname, ar in async_results:
                    try:
                        _, sdf_path = ar.get(timeout=5)
                        if sdf_path is not None:
                            sdf_paths.append(sdf_path)
                            smi_to_filename[smi] = fname
                    except mp.TimeoutError:
                        pass  # skip slow molecules
                    except Exception:
                        pass

                pool.terminate()
                pool.join()

            t_sdf = time.perf_counter() - t0

            if not sdf_paths:
                print(f"[UniDockScorer] all {n} 3D generations failed "
                      f"({t_sdf:.1f}s)")
                return scores

            n_ok = len(sdf_paths)
            n_fail = n - n_ok

            # Step 2: Run Uni-Dock GPU batch
            t1 = time.perf_counter()
            out_dir = os.path.join(tmpdir, 'output')
            os.makedirs(out_dir, exist_ok=True)
            success = self._run_unidock(sdf_paths, out_dir)
            t_dock = time.perf_counter() - t1

            if not success:
                print(f"[UniDockScorer] dock failed | "
                      f"SDF: {t_sdf:.1f}s ({n_ok}/{n})")
                return scores

            # Step 3: Parse output
            t2 = time.perf_counter()
            parsed = self._parse_scores(out_dir)
            t_parse = time.perf_counter() - t2

            # Map back
            for idx, smi in enumerate(smiles_list):
                fname = smi_to_filename.get(smi)
                if fname is not None and fname in parsed:
                    scores[idx] = parsed[fname]

            # Timing stats
            self._timing['sdf'] += t_sdf
            self._timing['dock'] += t_dock
            self._timing['parse'] += t_parse
            self._timing['calls'] += 1

            print(f"[UniDockScorer] {n_ok}/{n} docked "
                  f"({n_fail} 3D fail, {n_ok - len(parsed)} dock fail) | "
                  f"SDF: {t_sdf:.1f}s, Dock: {t_dock:.1f}s, "
                  f"Parse: {t_parse:.2f}s | "
                  f"Total: {t_sdf + t_dock + t_parse:.1f}s")

        return scores

    def _smiles_to_sdf(self, smi, tmpdir, fname):
        """Convert SMILES to 3D SDF file (sequential fallback for small batches).

        Returns path to SDF file, or None on failure.
        """
        result = _generate_sdf_worker(
            (smi, tmpdir, fname, self.seed, self.max_embed_attempts))
        return result[1]

    def _run_unidock(self, sdf_paths, out_dir):
        """Run Uni-Dock GPU batch docking.

        Parameters
        ----------
        sdf_paths : list[str]
            Paths to input SDF files.
        out_dir : str
            Output directory.

        Returns True on success, False on failure.
        """
        cmd = [
            _UNIDOCK_BIN,
            '--receptor', self.receptor_pdbqt,
            '--gpu_batch'] + sdf_paths + [
            '--center_x', str(self.center_x),
            '--center_y', str(self.center_y),
            '--center_z', str(self.center_z),
            '--size_x', str(self.size_x),
            '--size_y', str(self.size_y),
            '--size_z', str(self.size_z),
            '--search_mode', self.search_mode,
            '--scoring', self.scoring,
            '--num_modes', str(self.num_modes),
            '--seed', str(self.seed),
            '--dir', out_dir,
            '--verbosity', str(self.verbosity),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min max
            )
            if result.returncode != 0:
                print(f"[UniDockScorer] unidock failed (rc={result.returncode})")
                if result.stderr:
                    print(f"[UniDockScorer] stderr: {result.stderr[:500]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print("[UniDockScorer] unidock timed out (300s)")
            return False
        except FileNotFoundError:
            print(f"[UniDockScorer] unidock binary not found: {_UNIDOCK_BIN}")
            return False

    def _parse_scores(self, out_dir):
        """Parse docking scores from Uni-Dock output SDF files.

        Uni-Dock v1.1.3 writes ``{name}_out.sdf`` with a property
        ``Uni-Dock RESULT`` containing ``ENERGY=  -7.200 ...``.
        The best (most negative) energy for each molecule is returned.

        Returns
        -------
        dict[str, float]
            Mapping of base filename (without ``_out`` suffix) → best score.
        """
        scores = {}
        out_path = Path(out_dir)

        for sdf_file in out_path.glob('*.sdf'):
            basename = sdf_file.stem
            if basename.endswith('_out'):
                basename = basename[:-4]

            best_score = PENALTY_SCORE
            found_score = False

            try:
                suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
                for mol in suppl:
                    if mol is None:
                        continue
                    props = mol.GetPropsAsDict()

                    # Primary: Uni-Dock RESULT property (ENERGY= field)
                    ud_result = props.get('Uni-Dock RESULT', '')
                    m = re.search(r'ENERGY\s*=\s*([-\d.]+)', ud_result)
                    if m:
                        val = float(m.group(1))
                        if val < best_score or not found_score:
                            best_score = val
                            found_score = True
                        continue

                    # Fallback: any property with 'score' or 'energy'
                    for prop_name, prop_val in props.items():
                        key = prop_name.lower()
                        if 'score' in key or 'energy' in key or 'affinity' in key:
                            try:
                                val = float(str(prop_val).split()[0])
                                if val < best_score or not found_score:
                                    best_score = val
                                    found_score = True
                            except (ValueError, IndexError):
                                continue

            except Exception:
                pass

            # Last resort: regex on raw text
            if not found_score:
                try:
                    text = sdf_file.read_text()
                    for m in re.finditer(r'ENERGY\s*=\s*([-\d.]+)', text):
                        val = float(m.group(1))
                        if val < best_score or not found_score:
                            best_score = val
                            found_score = True
                except Exception:
                    pass

            if found_score:
                scores[basename] = best_score

        return scores
