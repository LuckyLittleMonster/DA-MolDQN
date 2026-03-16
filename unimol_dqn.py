"""UniMol-DQN: Pretrained 3D molecular encoder + trainable Q-head.

Architecture:
    Frozen Uni-Mol encoder (pretrained 209M conformations) → 512-dim CLS token
    + step_fraction → 513-dim
    → Trainable Q-head MLP → Q-value

The Uni-Mol encoder captures 3D molecular properties (conformation, shape,
electrostatics) that Morgan fingerprints fundamentally cannot represent.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


class EmbeddingCache:
    """SMILES → embedding cache with optional disk persistence.

    Avoids recomputing Uni-Mol embeddings (~10ms/mol) for seen molecules.
    """

    def __init__(self, cache_path=None):
        self._cache = {}  # smiles -> np.ndarray (512,)
        self._cache_path = cache_path
        self._hits = 0
        self._misses = 0

        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            for s, e in zip(data['smiles'].tolist(), data['embeddings']):
                self._cache[s] = e
            print(f"  EmbeddingCache: loaded {len(self._cache)} entries from {cache_path}")

    def get(self, smiles):
        """Return cached embedding or None."""
        emb = self._cache.get(smiles)
        if emb is not None:
            self._hits += 1
        else:
            self._misses += 1
        return emb

    def put(self, smiles, embedding):
        self._cache[smiles] = embedding

    def get_batch(self, smiles_list):
        """Return (embeddings, uncached_indices, uncached_smiles)."""
        result = [None] * len(smiles_list)
        uncached_idx = []
        uncached_smi = []
        for i, smi in enumerate(smiles_list):
            emb = self.get(smi)
            if emb is not None:
                result[i] = emb
            else:
                uncached_idx.append(i)
                uncached_smi.append(smi)
        return result, uncached_idx, uncached_smi

    def save(self):
        if self._cache_path and self._cache:
            smiles = list(self._cache.keys())
            embeddings = np.stack([self._cache[s] for s in smiles])
            np.savez(self._cache_path, smiles=smiles, embeddings=embeddings)

    @property
    def size(self):
        return len(self._cache)

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class UniMolEncoder:
    """Bare-metal UniMol encoder bypassing unimol_tools overhead.

    Pipeline: atoms+coords → transform_raw → GPU inference → 512-dim CLS
    Eliminates Pool creation, DataHub, tqdm, logging overhead (~176ms saved).
    """

    def __init__(self, use_gpu=True):
        from unimol_tools import UniMolRepr
        from unimol_tools.data.conformer import ConformerGen, inner_smi2coords
        from unimol_tools.predictor import MolDataset
        import logging
        logging.getLogger('unimol_tools').setLevel(logging.WARNING)

        use_cuda = use_gpu and torch.cuda.is_available()
        self._repr = UniMolRepr(data_type='molecule', remove_hs=False, use_cuda=use_cuda)
        self._confgen = ConformerGen(data_type='molecule', remove_hs=False)
        self._inner_smi2coords = inner_smi2coords
        self._model = self._repr.model
        self._device = self._repr.device
        self._model.eval()
        self.dim = 512

    @torch.no_grad()
    def encode(self, smiles_list):
        """Encode SMILES → (N, 512) numpy array via unimol_tools (fallback)."""
        if not smiles_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        result = self._repr.get_repr(smiles_list, return_atomic_reprs=False)
        return np.stack(result).astype(np.float32)

    @torch.no_grad()
    def encode_from_conformers(self, conformer_list):
        """Encode pre-computed conformers → (N, 512) numpy array.

        Args:
            conformer_list: list of (atoms, coords) tuples
                atoms: list of element symbols (with H)
                coords: np.ndarray (n_atoms, 3)

        Returns:
            np.ndarray (N, 512) float32
        """
        if not conformer_list:
            return np.zeros((0, self.dim), dtype=np.float32)

        atoms_list = [c[0] for c in conformer_list]
        coords_list = [c[1] for c in conformer_list]

        # Featurize (transform_raw: ~1.6ms for 40 mols)
        inputs = self._confgen.transform_raw(atoms_list, coords_list)

        # Collate batch manually (skip DataLoader overhead)
        batch = self._collate(inputs)

        # GPU inference (~10ms for 40 mols)
        out = self._model(**batch, return_repr=True, return_atomic_reprs=False)
        if isinstance(out, tuple):
            cls_repr = out[0]
        else:
            cls_repr = out

        return cls_repr.cpu().numpy().astype(np.float32)

    def generate_conformer(self, smiles):
        """Generate conformer for a single SMILES. Returns (atoms, coords)."""
        atoms, coords, mol = self._inner_smi2coords(
            smiles, seed=42, mode='fast', remove_hs=False
        )
        return atoms, coords

    def _collate(self, inputs):
        """Manual batch collation → GPU tensors."""
        bs = len(inputs)
        max_n = max(inp['src_tokens'].shape[0] for inp in inputs)

        src_tokens = torch.zeros(bs, max_n, dtype=torch.long, device=self._device)
        src_coord = torch.zeros(bs, max_n, 3, dtype=torch.float32, device=self._device)
        src_distance = torch.zeros(bs, max_n, max_n, dtype=torch.float32, device=self._device)
        src_edge_type = torch.zeros(bs, max_n, max_n, dtype=torch.long, device=self._device)

        for i, inp in enumerate(inputs):
            n = inp['src_tokens'].shape[0]
            src_tokens[i, :n] = torch.tensor(inp['src_tokens'], dtype=torch.long)
            src_coord[i, :n] = torch.tensor(inp['src_coord'], dtype=torch.float32)
            src_distance[i, :n, :n] = torch.tensor(inp['src_distance'], dtype=torch.float32)
            src_edge_type[i, :n, :n] = torch.tensor(inp['src_edge_type'], dtype=torch.long)

        return {
            'src_tokens': src_tokens, 'src_coord': src_coord,
            'src_distance': src_distance, 'src_edge_type': src_edge_type,
        }


class UniMolDQN(nn.Module):
    """Frozen Uni-Mol encoder + trainable Q-head.

    encode_actions(): SMILES → embeddings (uses cache, skips if random action)
    forward(): embedding tensor → Q-values (trainable Q-head only)

    Stability fixes:
    - L2-normalize embeddings before Q-head (scale from L2~27 to 1)
    - No LayerNorm (conflicts with Polyak target network)
    - Simple Linear+ReLU like MolDQN baseline
    """

    def __init__(self, encoder, cache, q_hidden=256):
        super().__init__()
        self.encoder = encoder  # UniMolEncoder (not nn.Module, runs externally)
        self.cache = cache
        self.emb_dim = encoder.dim  # 512

        # Trainable Q-head: normalized_emb + step_fraction → Q-value
        # Architecture mirrors MolDQN (Linear+ReLU, no LayerNorm/Dropout)
        input_dim = self.emb_dim + 1  # 513
        self.q_head = nn.Sequential(
            nn.Linear(input_dim, q_hidden),
            nn.ReLU(),
            nn.Linear(q_hidden, q_hidden // 2),
            nn.ReLU(),
            nn.Linear(q_hidden // 2, 1),
        )

    def forward(self, x):
        """Q-head forward. x: (batch, emb_dim+1) tensor."""
        return self.q_head(x)

    def encode_actions(self, smiles_list, step_fraction, conformers=None):
        """Encode action SMILES → observation tensor (N, 513).

        Uses cache. For uncached molecules, uses pre-computed conformers
        (if provided) or falls back to SMILES-based encoding.

        Args:
            smiles_list: list of SMILES strings
            step_fraction: float, appended as last feature
            conformers: optional list of (atoms, coords) tuples, parallel to smiles_list.
                        None entries = use cache or SMILES encoding.

        Returns:
            np.ndarray (N, 513) float32
        """
        result, uncached_idx, uncached_smi = self.cache.get_batch(smiles_list)

        if uncached_smi:
            if conformers is not None:
                # Use pre-computed conformers for uncached molecules
                uncached_confs = [conformers[i] for i in uncached_idx]
                # Filter out None entries (failed conformer generation)
                valid = [(i, c) for i, c in zip(uncached_idx, uncached_confs)
                         if c is not None]
                failed = [i for i, c in zip(uncached_idx, uncached_confs)
                          if c is None]

                if valid:
                    valid_idx, valid_confs = zip(*valid)
                    new_embs = self.encoder.encode_from_conformers(list(valid_confs))
                    for idx, emb in zip(valid_idx, new_embs):
                        smi = smiles_list[idx]
                        self.cache.put(smi, emb)
                        result[idx] = emb

                if failed:
                    # Fallback to SMILES encoding for failed conformers
                    failed_smi = [smiles_list[i] for i in failed]
                    fallback_embs = self.encoder.encode(failed_smi)
                    for idx, emb in zip(failed, fallback_embs):
                        smi = smiles_list[idx]
                        self.cache.put(smi, emb)
                        result[idx] = emb
            else:
                # No conformers provided, use SMILES encoding
                new_embs = self.encoder.encode(uncached_smi)
                for idx, smi, emb in zip(uncached_idx, uncached_smi, new_embs):
                    self.cache.put(smi, emb)
                    result[idx] = emb

        embeddings = np.stack(result)  # (N, 512)
        # L2-normalize embeddings (scale from L2~27 to 1.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms
        step_col = np.full((len(smiles_list), 1), step_fraction, dtype=np.float32)
        return np.hstack([embeddings, step_col])  # (N, 513)


class MolDQNBaseline(nn.Module):
    """Original MolDQN architecture for comparison. FP(2048) + step → Q-value."""

    def __init__(self, input_length=2049):
        super().__init__()
        self.linear_1 = nn.Linear(input_length, 1024)
        self.linear_2 = nn.Linear(1024, 512)
        self.linear_3 = nn.Linear(512, 128)
        self.linear_4 = nn.Linear(128, 32)
        self.linear_5 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        x = self.activation(self.linear_4(x))
        return self.linear_5(x)
