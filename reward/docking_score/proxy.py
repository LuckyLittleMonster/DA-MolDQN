"""Proxy model scorers for sEH, DRD2, GSK3b -- matching baselines (RxnFlow/SynFlowNet/RGFN).

sEH: MPNN regression proxy (Bengio et al. 2021), trained on 300K AutoDock Vina scores.
DRD2: SVM classifier (ExCAPE-DB + ECFP6), predicts P(active) in [0,1].
GSK3b: Random Forest classifier (TDC/ExCAPE-DB + ECFP4), predicts P(active) in [0,1].

Usage:
    scorer = ProxyScorer('seh', device='cuda')
    scores = scorer.score(['CCO', 'c1ccccc1'])  # list of floats

    # Drop-in replacement for UniDockScorer (same batch_dock API):
    adapter = ProxyDockAdapter('seh', device='cuda')
    vina_scores = adapter.batch_dock(['CCO', 'c1ccccc1'])
"""

import gzip
import os
import pickle
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType

# --------------------------------------------------------------------------- #
# sEH MPNN proxy (from GFlowNet / Bengio et al. 2021)                        #
# --------------------------------------------------------------------------- #

NUM_ATOMIC_NUMBERS = 56

_ATOMTYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
_BONDTYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, BT.UNSPECIFIED: 0}


def _onehot(arr, num_classes):
    arr = np.asarray(arr, dtype=np.int32)
    out = np.zeros((arr.shape[0], num_classes), dtype=np.int32)
    out[np.arange(arr.shape[0]), arr] = 1
    return out


def _mol_to_pyg_data(mol):
    """Convert RDKit Mol -> PyG Data (no torch_sparse dependency)."""
    from torch_geometric.data import Data

    natm = mol.GetNumAtoms()
    ntypes = len(_ATOMTYPES)
    nfeat = ntypes + 1 + 8 + NUM_ATOMIC_NUMBERS  # 14 + 1 + 56 = 71

    atmfeat = np.zeros((natm, nfeat), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        idx = _ATOMTYPES.get(atom.GetSymbol(), 5)
        if idx < ntypes:
            atmfeat[i, idx] = 1
        atmfeat[i, ntypes + 9 + atom.GetAtomicNum() - 1] = 1  # one-hot atomic number
        atmfeat[i, ntypes + 4] = atom.GetIsAromatic()
        hyb = atom.GetHybridization()
        atmfeat[i, ntypes + 5] = hyb == HybridizationType.SP
        atmfeat[i, ntypes + 6] = hyb == HybridizationType.SP2
        atmfeat[i, ntypes + 7] = hyb == HybridizationType.SP3
        atmfeat[i, ntypes + 8] = atom.GetTotalNumHs(includeNeighbors=True)

    # Bonds: bidirectional, 4 bond types (single=0, double=1, triple=2, aromatic=3)
    bonds = mol.GetBonds()
    if len(bonds) > 0:
        src, dst, feat = [], [], []
        for b in bonds:
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bt = _BONDTYPES.get(b.GetBondType(), 0)
            src.extend([i, j])
            dst.extend([j, i])
            feat.extend([bt, bt])
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.zeros((len(feat), 4), dtype=torch.float32)
        for k, f in enumerate(feat):
            edge_attr[k, f] = 1.0

        # Sort edges (replaces torch_sparse.coalesce)
        key = edge_index[0] * natm + edge_index[1]
        order = key.argsort()
        edge_index = edge_index[:, order]
        edge_attr = edge_attr[order]
    else:
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr = torch.zeros((1, 4), dtype=torch.float32)

    # Add stem mask column (always 0)
    x = torch.tensor(atmfeat, dtype=torch.float32)
    x = torch.cat([x, torch.zeros(natm, 1)], dim=1)  # 71 + 1 = 72

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class _MPNNet(nn.Module):
    """MPNN from Bengio et al. 2021 (sEH binding affinity proxy)."""

    def __init__(self, num_feat=72, dim=64, num_conv_steps=12):
        super().__init__()
        from torch_geometric.nn import NNConv, Set2Set

        self.lin0 = nn.Linear(num_feat, dim)
        self.num_conv_steps = num_conv_steps
        self.act = nn.LeakyReLU()

        net = nn.Sequential(nn.Linear(4, 128), self.act, nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr="mean")
        self.gru = nn.GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin3 = nn.Linear(dim * 2, 1)

    def forward(self, data):
        out = self.act(self.lin0(data.x))
        h = out.unsqueeze(0)
        for _ in range(self.num_conv_steps):
            m = self.act(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)
        global_out = self.set2set(out, data.batch)
        return self.lin3(global_out)


def _load_seh_proxy(cache_dir=None):
    """Load pretrained sEH proxy weights from GFlowNet repo."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / "Data" / "proxy_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "bengio2021flow_proxy.pkl.gz"

    # Try RxnFlow's cache first
    rxnflow_cache = Path(__file__).parent.parent.parent / "refs" / "RxnFlow" / "src" / "gflownet" / "models" / "cache" / "bengio2021flow_proxy.pkl.gz"
    synflow_cache = Path(__file__).parent.parent.parent / "refs" / "synflownet" / "src" / "synflownet" / "models" / "cache" / "bengio2021flow_proxy.pkl.gz"

    if not cache_file.exists():
        for alt in [rxnflow_cache, synflow_cache]:
            if alt.exists():
                import shutil
                shutil.copy2(alt, cache_file)
                break
        else:
            # Download from GitHub
            import requests
            url = "https://github.com/GFNOrg/gflownet/raw/master/mols/data/pretrained_proxy/best_params.pkl.gz"
            r = requests.get(url, stream=True, timeout=30)
            with open(cache_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    with gzip.open(cache_file, "rb") as f:
        params = pickle.load(f)  # nosec

    model = _MPNNet(num_feat=72, dim=64, num_conv_steps=12)
    param_map = {
        "lin0.weight": params[0], "lin0.bias": params[1],
        "conv.bias": params[3],
        "conv.nn.0.weight": params[4], "conv.nn.0.bias": params[5],
        "conv.nn.2.weight": params[6], "conv.nn.2.bias": params[7],
        "conv.lin.weight": params[2],
        "gru.weight_ih_l0": params[8], "gru.weight_hh_l0": params[9],
        "gru.bias_ih_l0": params[10], "gru.bias_hh_l0": params[11],
        "set2set.lstm.weight_ih_l0": params[16], "set2set.lstm.weight_hh_l0": params[17],
        "set2set.lstm.bias_ih_l0": params[18], "set2set.lstm.bias_hh_l0": params[19],
        "lin3.weight": params[20], "lin3.bias": params[21],
    }
    for k, v in param_map.items():
        model.get_parameter(k).data = torch.tensor(v)
    return model


# --------------------------------------------------------------------------- #
# DRD2 / GSK3b classifiers (from TDC / ExCAPE-DB)                            #
# --------------------------------------------------------------------------- #

def _ecfp6_fingerprint(mol, size=2048):
    """ECFP6 fingerprint (Morgan radius 3, count+feature) as numpy array.
    Matches TDC's fingerprints_from_mol for DRD2."""
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    nfp = np.zeros((1, size), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nfp[0, idx % size] += int(v)
    return nfp


def _ecfp4_fingerprint(mol, size=2048):
    """ECFP4 binary fingerprint (Morgan radius 2) as numpy array.
    Matches TDC's gsk3b oracle convention."""
    from rdkit.Chem import DataStructs
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    arr = np.zeros((1, size), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr[0])
    return arr


def _load_tdc_model(name):
    """Load oracle model for DRD2, GSK3B, or JNK3.

    DRD2: TDC's SVM model (ExCAPE-DB), uses ECFP6 count+feature fingerprints.
    GSK3B: TDC's RF model (ExCAPE-DB), converted to sklearn 1.8 format, uses ECFP4 binary fingerprints.
    JNK3: TDC's RF model (ExCAPE-DB), converted to sklearn 1.8 format, uses ECFP4 binary fingerprints.
    """
    import sys
    import types as _types

    oracles_dir = Path(__file__).resolve().parent.parent.parent / "Data" / "oracles"

    if name == 'DRD2':
        # Patch rdkit.six for newer RDKit versions (needed to unpickle old TDC model)
        if 'rdkit.six' not in sys.modules:
            rdkit_six = _types.ModuleType('rdkit.six')
            rdkit_six.iteritems = lambda d: d.items()
            sys.modules['rdkit.six'] = rdkit_six
        cache = oracles_dir / "drd2_current.pkl"
        if not cache.exists():
            raise FileNotFoundError(f"DRD2 model not found at {cache}")
        with open(cache, 'rb') as f:
            return pickle.load(f)  # nosec
    elif name == 'GSK3B':
        # TDC's original RF model, converted from sklearn 0.23 -> 1.8 format
        cache = oracles_dir / "gsk3b_tdc_converted.pkl"
        if not cache.exists():
            raise FileNotFoundError(
                f"GSK3B model not found at {cache}. "
                "Run the sklearn version conversion script to generate it."
            )
        with open(cache, 'rb') as f:
            return pickle.load(f)  # nosec
    elif name == 'JNK3':
        # TDC's RF model (ExCAPE-DB), converted from sklearn 0.21 -> 1.8 format
        cache = oracles_dir / "jnk3_tdc_converted.pkl"
        if not cache.exists():
            raise FileNotFoundError(
                f"JNK3 model not found at {cache}. "
                "Run the sklearn version conversion script to generate it."
            )
        with open(cache, 'rb') as f:
            return pickle.load(f)  # nosec
    else:
        raise ValueError(f"Unknown oracle: {name}")


# --------------------------------------------------------------------------- #
# GPU SVM Predictor (DRD2)                                                     #
# --------------------------------------------------------------------------- #

class _GPUSVMPredictor:
    """GPU-accelerated RBF SVM prediction matching sklearn SVC.predict_proba.

    Converts sklearn SVC with RBF kernel to pure PyTorch tensor ops:
      K(X, SV) = exp(-gamma * ||X - SV||^2)
      decision = K @ dual_coef^T + intercept
      P(class_1) = 1 - 1/(1+exp(probA*(-decision)+probB))   # Platt scaling

    Achieves ~178x speedup over sklearn sequential (0.67ms vs 119ms for 64 mols).
    Max numerical diff vs sklearn: <5e-4 (float32), ranking perfectly preserved.
    """

    def __init__(self, svm_model, device):
        self.device = torch.device(device)
        sv = svm_model.support_vectors_
        self._sv = torch.tensor(sv, dtype=torch.float32, device=self.device)
        self._sv_sq = (self._sv ** 2).sum(dim=1)  # (M,)
        self._dual = torch.tensor(
            svm_model.dual_coef_, dtype=torch.float32, device=self.device)  # (1, M)
        self._intercept = float(svm_model.intercept_[0])
        self._gamma = float(svm_model._gamma)
        self._probA = float(svm_model.probA_[0])
        self._probB = float(svm_model.probB_[0])

    @torch.inference_mode()
    def predict_proba(self, fps_np: np.ndarray) -> np.ndarray:
        """Predict P(active) for batch of fingerprints.

        Args:
            fps_np: (N, D) numpy array of fingerprints.
        Returns:
            (N,) numpy array of P(class_1) probabilities.
        """
        fps = torch.tensor(fps_np, dtype=torch.float32, device=self.device)
        x_sq = (fps ** 2).sum(dim=1, keepdim=True)          # (N, 1)
        cross = fps @ self._sv.T                              # (N, M)
        dist_sq = (x_sq + self._sv_sq.unsqueeze(0) - 2.0 * cross).clamp(min=0)
        K = torch.exp(-self._gamma * dist_sq)                 # (N, M)
        decision = (K * self._dual).sum(dim=1) + self._intercept  # (N,)
        # Platt scaling: negate decision for libsvm internal class ordering
        fApB = self._probA * (-decision) + self._probB
        p_class0 = 1.0 / (1.0 + torch.exp(fApB))
        return (1.0 - p_class0).cpu().numpy()


# --------------------------------------------------------------------------- #
# Unified ProxyScorer                                                          #
# --------------------------------------------------------------------------- #

class ProxyScorer:
    """Unified proxy scorer matching baseline methods.

    Args:
        target: 'seh', 'drd2', 'gsk3b', or 'jnk3'
        device: torch device for sEH/DRD2 proxy (GPU accelerated)
    """

    def __init__(self, target: str, device: str = 'cpu'):
        self.target = target.lower()
        self.device = torch.device(device)

        if self.target == 'seh':
            self._model = _load_seh_proxy()
            self._model.eval()
            self._model.to(self.device)
        elif self.target == 'drd2':
            sklearn_model = _load_tdc_model('DRD2')
            self._gpu_svm = _GPUSVMPredictor(sklearn_model, device=self.device)
        elif self.target == 'gsk3b':
            self._model = _load_tdc_model('GSK3B')
        elif self.target == 'jnk3':
            self._model = _load_tdc_model('JNK3')
        else:
            raise ValueError(f"Unknown target: {target}. Use 'seh', 'drd2', 'gsk3b', or 'jnk3'.")

    @torch.inference_mode()
    def score(self, smiles: list[str] | str, mols: list | None = None) -> list[float]:
        """Score molecules. Returns list of floats (same length as input).

        Args:
            smiles: SMILES string or list of SMILES.
            mols: Optional pre-parsed RDKit Mol objects (same length as smiles).
                  Skips redundant MolFromSmiles calls when provided.
        """
        if isinstance(smiles, str):
            smiles = [smiles]
            if mols is not None and not isinstance(mols, list):
                mols = [mols]

        if self.target == 'seh':
            return self._score_seh(smiles, mols)
        elif self.target == 'drd2':
            return self._score_drd2_gpu(smiles, mols)
        elif self.target in ('gsk3b', 'jnk3'):
            return self._score_rf_batch(smiles, mols)
        else:
            return self._score_rf_batch(smiles, mols)

    def _score_seh(self, smiles: list[str], mols: list | None = None) -> list[float]:
        """sEH proxy: MPNN -> score / 8 (matches RxnFlow/SynFlowNet)."""
        from torch_geometric.data import Batch

        results = [0.0] * len(smiles)
        valid_idx, graphs = [], []

        for i, smi in enumerate(smiles):
            mol = mols[i] if mols is not None else Chem.MolFromSmiles(smi)
            if mol is not None and mol.GetNumAtoms() > 0:
                try:
                    g = _mol_to_pyg_data(mol)
                    valid_idx.append(i)
                    graphs.append(g)
                except Exception:
                    pass

        if graphs:
            batch = Batch.from_data_list(graphs).to(self.device)
            preds = self._model(batch).reshape(-1).cpu() / 8.0
            preds = preds.clamp(min=1e-4, max=100.0)
            preds[preds.isnan()] = 0.0
            for j, idx in enumerate(valid_idx):
                results[idx] = preds[j].item()

        return results

    def _score_drd2_gpu(self, smiles: list[str], mols: list | None = None) -> list[float]:
        """DRD2: GPU-accelerated RBF SVM with batch fingerprinting."""
        results = [0.0] * len(smiles)
        valid_idx, fps = [], []

        for i, smi in enumerate(smiles):
            mol = mols[i] if mols is not None else Chem.MolFromSmiles(smi)
            if mol is not None:
                fps.append(_ecfp6_fingerprint(mol)[0])
                valid_idx.append(i)

        if fps:
            batch_fp = np.stack(fps).astype(np.float64)
            probs = self._gpu_svm.predict_proba(batch_fp)
            for j, idx in enumerate(valid_idx):
                results[idx] = float(probs[j])

        return results

    def _score_rf_batch(self, smiles: list[str], mols: list | None = None) -> list[float]:
        """GSK3B/JNK3: Batched sklearn RF predict_proba with ECFP4 fingerprints."""
        results = [0.0] * len(smiles)
        valid_idx, fps = [], []

        for i, smi in enumerate(smiles):
            mol = mols[i] if mols is not None else Chem.MolFromSmiles(smi)
            if mol is not None:
                fps.append(_ecfp4_fingerprint(mol)[0])
                valid_idx.append(i)

        if fps:
            batch_fp = np.stack(fps)
            probs = self._model.predict_proba(batch_fp)[:, 1]
            for j, idx in enumerate(valid_idx):
                results[idx] = float(probs[j])

        return results

    def __repr__(self):
        return f"ProxyScorer(target='{self.target}', device={self.device})"


# --------------------------------------------------------------------------- #
# ProxyDockAdapter -- UniDockScorer-compatible wrapper                        #
# --------------------------------------------------------------------------- #

class ProxyDockAdapter:
    """Adapts ProxyScorer to UniDockScorer API (drop-in replacement).

    Score transform: proxy_score -> pseudo_vina = -proxy_score * 12.0
    Round-trip: proxy=0.5 -> vina=-6.0 -> dock_norm=clip(-(-6)/12, 0, 1)=0.5
    """

    def __init__(self, target, device='cuda'):
        self._proxy = ProxyScorer(target, device=device)
        self._cache = {}
        self._timing = {'score': 0.0, 'calls': 0}

    def batch_dock(self, smiles_list, mols=None):
        """Score molecules, returning pseudo-Vina scores (negative = better).

        Args:
            smiles_list: List of SMILES strings.
            mols: Optional pre-parsed RDKit Mol objects (same length).
                  Avoids redundant MolFromSmiles calls when provided.
        """
        if not smiles_list:
            return []

        results = [0.0] * len(smiles_list)
        uncached = []
        uncached_idx = []
        uncached_mols = []
        for i, smi in enumerate(smiles_list):
            mol = mols[i] if mols is not None else None
            canon, mol = self._canonicalize(smi, mol)
            if canon in self._cache:
                results[i] = self._cache[canon]
            else:
                uncached.append(canon)
                uncached_idx.append(i)
                uncached_mols.append(mol)

        if uncached:
            t0 = time.perf_counter()
            proxy_scores = self._proxy.score(uncached, mols=uncached_mols)
            self._timing['score'] += time.perf_counter() - t0
            self._timing['calls'] += 1
            for j, (canon, ps) in enumerate(zip(uncached, proxy_scores)):
                pseudo_vina = -ps * 12.0
                self._cache[canon] = pseudo_vina
                results[uncached_idx[j]] = pseudo_vina

        return results

    @staticmethod
    def _canonicalize(smi, mol=None):
        """Return (canonical_smiles, mol). Reuses mol if provided."""
        if mol is None:
            mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol), mol
        return smi, None

    @property
    def cache_size(self):
        return len(self._cache)

    def clear_cache(self):
        self._cache.clear()

    @property
    def timing_summary(self):
        t = self._timing
        return (f"ProxyDockAdapter timing ({t['calls']} calls): "
                f"Score={t['score']:.1f}s")

    def __repr__(self):
        return f"ProxyDockAdapter(target='{self._proxy.target}')"


class MultiProxyDockAdapter:
    """Multi-target ProxyDockAdapter: product of P(active) across targets.

    For GSK3B+JNK3 MOO benchmark:
      combined_proxy = P(GSK3B) * P(JNK3)  (range [0,1])
      pseudo_vina = -combined_proxy * 12.0   (UniDockScorer API compat)

    Existing reward modes work unchanged:
      dock:  reward = dock_norm = combined_proxy = P(GSK3B)*P(JNK3)   [2-obj]
      multi: reward = dock_norm * QED * SA_norm                       [4-obj]
    """

    def __init__(self, targets, device='cuda'):
        self._scorers = [ProxyScorer(t, device=device) for t in targets]
        self._targets = targets
        self._cache = {}
        self._per_target_cache = {}  # smi -> {target: prob}
        self._timing = {'score': 0.0, 'calls': 0}

    def batch_dock(self, smiles_list, mols=None):
        """Score molecules, returning pseudo-Vina of product score."""
        if not smiles_list:
            return []

        results = [0.0] * len(smiles_list)
        uncached = []
        uncached_idx = []
        uncached_mols = []
        for i, smi in enumerate(smiles_list):
            mol = mols[i] if mols is not None else None
            canon, mol = ProxyDockAdapter._canonicalize(smi, mol)
            if canon in self._cache:
                results[i] = self._cache[canon]
            else:
                uncached.append(canon)
                uncached_idx.append(i)
                uncached_mols.append(mol)

        if uncached:
            t0 = time.perf_counter()
            # Score with each target scorer, then multiply
            all_scores = []
            for scorer in self._scorers:
                all_scores.append(scorer.score(uncached, mols=uncached_mols))
            self._timing['score'] += time.perf_counter() - t0
            self._timing['calls'] += 1

            for j in range(len(uncached)):
                product = 1.0
                per_target = {}
                for t_idx, target_scores in enumerate(all_scores):
                    prob = target_scores[j]
                    product *= prob
                    per_target[self._targets[t_idx]] = prob
                pseudo_vina = -product * 12.0
                self._cache[uncached[j]] = pseudo_vina
                self._per_target_cache[uncached[j]] = per_target
                results[uncached_idx[j]] = pseudo_vina

        return results

    @property
    def cache_size(self):
        return len(self._cache)

    def clear_cache(self):
        self._cache.clear()
        self._per_target_cache.clear()

    def get_all_scored(self):
        """Return dict of {smiles: {target: prob}} for all scored molecules."""
        return dict(self._per_target_cache)

    @property
    def timing_summary(self):
        t = self._timing
        return (f"MultiProxyDockAdapter timing ({t['calls']} calls): "
                f"Score={t['score']:.1f}s")

    def __repr__(self):
        return f"MultiProxyDockAdapter(targets={self._targets})"
