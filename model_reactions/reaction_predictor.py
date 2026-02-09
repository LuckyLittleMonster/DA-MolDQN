"""
Combined ReactionPredictor: Model 1 (ReactantPredictor) + Model 2 (ProductPredictor).

Provides the main get_valid_actions(mol) API for RL integration.

Pipeline:
    mol -> Model 1 (hypergraph link prediction) -> top-k co-reactants
    (mol, co_reactant) -> Model 2 (ReactionT5v2) -> products + scores

Usage:
    from model_reactions.reaction_predictor import ReactionPredictor

    predictor = ReactionPredictor(device='cuda')
    co_reactants, products, scores = predictor.get_valid_actions("CCO", top_k=5)
"""

import math
import os
import time
import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model_reactions.config import ReactionPredictorConfig
from model_reactions.product_prediction.product_predictor import ProductPredictor
from model_reactions.utils import canonicalize_smiles, tanimoto_from_mols


class HypergraphReactantPredictorWrapper:
    """
    Wrapper that loads the trained HypergraphLinkPredictor model
    and provides the ReactantPredictor.predict() interface.

    Uses pre-computed embeddings for fast nearest-neighbor retrieval.
    """

    def __init__(self, checkpoint_path: str, data_dir: str = "Data/uspto",
                 device: str = 'auto', max_db_mols: int = 100000):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.max_db_mols = max_db_mols

        self.model = None
        self.smiles_list = None
        self.embedding_matrix = None  # (n_mols, embed_dim), normalized
        self._loaded = False

    def load(self):
        """Load model and build embedding index."""
        if self._loaded:
            return

        import sys
        from model_reactions.link_prediction.hypergraph_link_predictor import (
            HypergraphLinkPredictor,
            HypergraphLinkConfig,
            smiles_to_graph,
        )
        from model_reactions.reactant_predictor import ReactionDatabase

        # Patch: checkpoint may have been saved with HypergraphLinkConfig in __main__
        main_mod = sys.modules.get('__main__')
        if main_mod and not hasattr(main_mod, 'HypergraphLinkConfig'):
            main_mod.HypergraphLinkConfig = HypergraphLinkConfig

        print("Loading HypergraphReactantPredictor...")
        start = time.time()

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', HypergraphLinkConfig.medium())
        if isinstance(config, dict):
            config = HypergraphLinkConfig(**config)

        self.model = HypergraphLinkPredictor(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Build database
        db_cache = os.path.join(self.data_dir, 'reaction_db_cache.pkl')
        db = ReactionDatabase(self.data_dir)
        if os.path.exists(db_cache):
            db.load(db_cache)
        else:
            db.build()
            db.save(db_cache)

        self.smiles_list = db.reactant_smiles
        n_mols = min(len(self.smiles_list), self.max_db_mols)
        self.smiles_list = self.smiles_list[:n_mols]

        # Pre-compute embeddings
        print(f"  Pre-computing embeddings for {n_mols} molecules...")
        embeddings = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, n_mols, batch_size):
                batch_smiles = self.smiles_list[i:i + batch_size]
                batch_graphs = [smiles_to_graph(s) for s in batch_smiles]

                # Prepare batch
                all_atoms, all_edges, all_edge_types, all_batch = [], [], [], []
                atom_offset = 0
                valid_in_batch = []

                for j, g in enumerate(batch_graphs):
                    if g is None:
                        continue
                    n_atoms = len(g['atom_types'])
                    all_atoms.extend(g['atom_types'])
                    for e in g['edge_index']:
                        all_edges.append([e[0] + atom_offset, e[1] + atom_offset])
                    all_edge_types.extend(g['edge_types'])
                    all_batch.extend([len(valid_in_batch)] * n_atoms)
                    atom_offset += n_atoms
                    valid_in_batch.append(i + j)

                if not all_atoms:
                    continue

                atoms_t = torch.tensor(all_atoms, dtype=torch.long, device=self.device)
                edges_t = torch.tensor(all_edges, dtype=torch.long, device=self.device).t().contiguous()
                etypes_t = torch.tensor(all_edge_types, dtype=torch.long, device=self.device)
                batch_t = torch.tensor(all_batch, dtype=torch.long, device=self.device)

                _, link_emb = self.model.encode_molecule(atoms_t, edges_t, etypes_t, batch_t)
                embeddings.append(link_emb.cpu().numpy())

        self.embedding_matrix = np.concatenate(embeddings, axis=0)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        self.embedding_matrix = self.embedding_matrix / np.where(norms > 0, norms, 1e-8)

        # Build O(1) lookup index for self-exclusion
        self.smiles_to_idx = {s: i for i, s in enumerate(self.smiles_list)}

        elapsed = time.time() - start
        print(f"  Loaded: {self.embedding_matrix.shape[0]} molecules, "
              f"dim={self.embedding_matrix.shape[1]}, time={elapsed:.1f}s")
        self._loaded = True

    def predict(self, mol, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict co-reactants for a given molecule.

        Args:
            mol: RDKit Mol or SMILES string
            top_k: Number of co-reactants to return

        Returns:
            List of (co_reactant_smiles, score) tuples
        """
        if not self._loaded:
            self.load()

        from model_reactions.link_prediction.hypergraph_link_predictor import smiles_to_graph

        if isinstance(mol, str):
            smiles = mol
        else:
            smiles = Chem.MolToSmiles(mol)

        graph = smiles_to_graph(smiles)
        if graph is None:
            return []

        # Encode query molecule
        with torch.no_grad():
            atoms_t = torch.tensor(graph['atom_types'], dtype=torch.long, device=self.device)
            edges_t = torch.tensor(graph['edge_index'], dtype=torch.long, device=self.device).t().contiguous()
            etypes_t = torch.tensor(graph['edge_types'], dtype=torch.long, device=self.device)
            batch_t = torch.zeros(len(graph['atom_types']), dtype=torch.long, device=self.device)

            _, query_emb = self.model.encode_molecule(atoms_t, edges_t, etypes_t, batch_t)
            query_emb = query_emb.cpu().numpy().flatten()

        # Normalize
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb = query_emb / query_norm

        # Cosine similarity with all database molecules
        sims = self.embedding_matrix @ query_emb

        # Exclude self if in database (O(1) dict lookup)
        canon = canonicalize_smiles(smiles)
        self_idx = self.smiles_to_idx.get(canon)
        if self_idx is not None:
            sims[self_idx] = -1.0

        # Top-k
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if sims[idx] > 0:
                results.append((self.smiles_list[idx], float(sims[idx])))

        return results

    def predict_batch(self, mols, top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Batch predict co-reactants for multiple molecules.

        One GNN forward pass for all molecules, one matrix multiply for all similarities.

        Args:
            mols: List of RDKit Mol objects or SMILES strings
            top_k: Number of co-reactants per molecule

        Returns:
            List of List of (co_reactant_smiles, score) tuples
        """
        if not self._loaded:
            self.load()

        from model_reactions.link_prediction.hypergraph_link_predictor import smiles_to_graph

        # Convert to SMILES
        smiles_list = []
        for mol in mols:
            if isinstance(mol, str):
                smiles_list.append(mol)
            else:
                smiles_list.append(Chem.MolToSmiles(mol))

        # Build batched graph for all query molecules
        graphs = [smiles_to_graph(s) for s in smiles_list]

        all_atoms, all_edges, all_edge_types, all_batch = [], [], [], []
        atom_offset = 0
        valid_mol_indices = []  # maps position in batch -> original index

        for i, g in enumerate(graphs):
            if g is None:
                continue
            n_atoms = len(g['atom_types'])
            all_atoms.extend(g['atom_types'])
            for e in g['edge_index']:
                all_edges.append([e[0] + atom_offset, e[1] + atom_offset])
            all_edge_types.extend(g['edge_types'])
            all_batch.extend([len(valid_mol_indices)] * n_atoms)
            atom_offset += n_atoms
            valid_mol_indices.append(i)

        # Initialize results for all molecules (including invalid ones)
        all_results = [[] for _ in range(len(mols))]

        if not all_atoms:
            return all_results

        # Batch GNN forward pass
        with torch.no_grad():
            atoms_t = torch.tensor(all_atoms, dtype=torch.long, device=self.device)
            edges_t = torch.tensor(all_edges, dtype=torch.long, device=self.device).t().contiguous()
            etypes_t = torch.tensor(all_edge_types, dtype=torch.long, device=self.device)
            batch_t = torch.tensor(all_batch, dtype=torch.long, device=self.device)

            _, query_embs = self.model.encode_molecule(atoms_t, edges_t, etypes_t, batch_t)
            query_embs = query_embs.cpu().numpy()  # (N_valid, embed_dim)

        # Normalize query embeddings
        query_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
        query_embs = query_embs / np.where(query_norms > 0, query_norms, 1e-8)

        # Batch cosine similarity: (N_valid, embed_dim) @ (embed_dim, M) = (N_valid, M)
        sims_matrix = query_embs @ self.embedding_matrix.T

        # Canonicalize smiles for self-exclusion
        canon_smiles = [canonicalize_smiles(s) for s in smiles_list]

        # Process each valid molecule
        for batch_idx, orig_idx in enumerate(valid_mol_indices):
            sims = sims_matrix[batch_idx].copy()

            # Exclude self if in database (O(1) dict lookup)
            canon = canon_smiles[orig_idx]
            if canon is not None:
                self_idx = self.smiles_to_idx.get(canon)
                if self_idx is not None:
                    sims[self_idx] = -1.0

            # Top-k
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if sims[idx] > 0:
                    results.append((self.smiles_list[idx], float(sims[idx])))
            all_results[orig_idx] = results

        return all_results


class ReactionPredictor:
    """
    Combined two-model reaction predictor.

    Step 1: Model 1 (ReactantPredictor) finds co-reactants
    Step 2: Model 2 (ProductPredictor / ReactionT5v2) predicts products

    Provides get_valid_actions() API compatible with RL environment.
    """

    def load(self):
        """Load both models."""
        if self._loaded:
            return

        print("=" * 60)
        print("Initializing ReactionPredictor (Two-Model Pipeline)")
        print("=" * 60)
        start = time.time()

        # Model 1: ReactantPredictor
        method = self.config.reactant_method
        print(f"\n[Model 1] ReactantPredictor (method={method})")

        if method == 'hypergraph':
            checkpoint = self._find_hypergraph_checkpoint()
            self.reactant_predictor = HypergraphReactantPredictorWrapper(
                checkpoint_path=checkpoint,
                data_dir=self.config.reactant_config.data_dir,
                device=self.device_str,
                max_db_mols=self.config.reactant_config.max_db_mols,
            )
            self.reactant_predictor.load()
        elif method == 'fingerprint':
            from model_reactions.reactant_predictor import (
                FingerprintReactantPredictor,
            )
            self.reactant_predictor = FingerprintReactantPredictor(
                self.config.reactant_config, strategy='partner'
            )
            self.reactant_predictor.build_database()
        else:
            raise ValueError(f"Unknown reactant method: {method}")

        # Model 2: ProductPredictor
        print(f"\n[Model 2] ProductPredictor (ReactionT5v2)")
        self.product_predictor = ProductPredictor(
            config=self.config.product_config,
            device=self.device_str,
        )
        self.product_predictor.load_model()

        elapsed = time.time() - start
        print(f"\nReactionPredictor ready. Total init time: {elapsed:.1f}s")
        print("=" * 60)
        self._loaded = True

    def _find_hypergraph_checkpoint(self) -> str:
        """Find the best hypergraph link predictor checkpoint."""
        search_paths = [
            "model_reactions/checkpoints/hypergraph_link_best.pt",
            "model_reactions/checkpoints/hypergraph_link_final.pt",
        ]
        for p in search_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"No hypergraph checkpoint found. Searched: {search_paths}"
        )

    def _apply_mw_reranking(self, reactant_results: List[Tuple[str, float]],
                            top_k: int) -> List[Tuple[str, float]]:
        """Apply MW-based penalty to co-reactant scores and re-rank.

        Smaller co-reactants get higher adjusted scores, biasing the system
        toward reactions that don't grow molecules excessively.
        """
        from rdkit.Chem.Descriptors import ExactMolWt

        alpha = self.config.mw_penalty_alpha
        threshold = self.config.mw_penalty_threshold

        adjusted = []
        for co_smiles, score in reactant_results:
            mol = Chem.MolFromSmiles(co_smiles)
            if mol is None:
                adjusted.append((co_smiles, score))
                continue
            mw = ExactMolWt(mol)
            penalty = math.exp(-alpha * max(0.0, mw - threshold))
            adjusted.append((co_smiles, score * penalty))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted[:top_k]

    def _expand_product(self, raw_product: str, canon_mol: str,
                        r_score: float) -> List[Tuple[str, float]]:
        """Process a raw T5 product, potentially expanding multi-component products.

        When expand_fragments is True and product contains '.', each valid fragment
        becomes a separate action candidate. This gives the agent "decomposition"
        options that can shrink molecules.

        Returns:
            List of (canonical_product_smiles, score) pairs.
        """
        results = []

        if '.' in raw_product:
            if self.config.expand_fragments:
                # Plan B: keep all valid fragments
                frags = raw_product.split('.')
                for frag in frags:
                    frag_mol = Chem.MolFromSmiles(frag)
                    if frag_mol is None:
                        continue
                    if frag_mol.GetNumHeavyAtoms() < self.config.fragment_min_atoms:
                        continue
                    canon_frag = canonicalize_smiles(frag)
                    if canon_frag is None or canon_frag == canon_mol:
                        continue
                    results.append((canon_frag, r_score))
            else:
                # Legacy: take largest fragment by heavy atom count
                frags = raw_product.split('.')
                frag_mols = [(f, Chem.MolFromSmiles(f)) for f in frags]
                frag_mols = [(f, m) for f, m in frag_mols if m is not None]
                if not frag_mols:
                    return results
                largest, prod_mol = max(frag_mols, key=lambda x: x[1].GetNumHeavyAtoms())
                if prod_mol is not None:
                    canon_prod = canonicalize_smiles(largest)
                    if canon_prod is not None and canon_prod != canon_mol:
                        results.append((canon_prod, r_score))
        else:
            prod_mol = Chem.MolFromSmiles(raw_product)
            if prod_mol is not None:
                canon_prod = canonicalize_smiles(raw_product)
                if canon_prod is not None and canon_prod != canon_mol:
                    results.append((canon_prod, r_score))

        return results

    def __init__(self, config: ReactionPredictorConfig = None, device: str = 'auto'):
        if config is None:
            config = ReactionPredictorConfig()
        self.config = config

        if device == 'auto':
            self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device_str = device

        self.reactant_predictor = None
        self.product_predictor = None
        self._loaded = False

        # Filter statistics
        self._filter_stats = defaultdict(int)

    def _tanimoto(self, mol_a, mol_b) -> float:
        """Compute Tanimoto similarity between two RDKit Mol objects."""
        return tanimoto_from_mols(mol_a, mol_b)

    def _filter_product_list(self, canon_mol: str, co_reactants: List[str],
                             products: List[str], scores: np.ndarray
                             ) -> Tuple[List[str], List[str], np.ndarray]:
        """Apply multi-layer product validation filter.

        Layers (chemist-calibrated thresholds from 700 real USPTO reactions):
            1. Copy detection: product == reactant or product == co_reactant
            2. Tanimoto thresholds: min similarity to reactant (0.2), max to co_reactant (0.9)
            3. MW reasonableness: ratio in [0.5, 1.3], delta < 200 Da
            4. Protecting group: reject products containing Si atoms

        If all actions are filtered out, keeps the one with highest Tanimoto
        to the reactant (from candidates that passed copy detection only).
        """
        if not self.config.filter_products or not products:
            return co_reactants, products, scores

        reactant_mol = Chem.MolFromSmiles(canon_mol)
        if reactant_mol is None:
            return co_reactants, products, scores

        reactant_mw = Descriptors.ExactMolWt(reactant_mol)

        kept_co = []
        kept_prod = []
        kept_scores = []
        # Fallback tracking: only from candidates that passed copy detection
        best_fallback_idx = None
        best_fallback_tani = -1.0

        for i, (co, prod, sc) in enumerate(zip(co_reactants, products, scores)):
            self._filter_stats['total'] += 1

            # Parse product
            prod_mol = Chem.MolFromSmiles(prod)
            if prod_mol is None:
                self._filter_stats['invalid_smiles'] += 1
                continue

            # Layer 1: Copy detection
            if self.config.filter_check_copy:
                canon_co = canonicalize_smiles(co)
                if prod == canon_mol:
                    self._filter_stats['copy_reactant'] += 1
                    continue  # skip fallback tracking for copies
                if canon_co is not None and prod == canon_co:
                    self._filter_stats['copy_coreactant'] += 1
                    continue  # skip fallback tracking for copies

            # -- Passed copy detection: eligible for fallback --
            tani_reactant = self._tanimoto(reactant_mol, prod_mol)

            # Track best fallback (only non-copy candidates reach here)
            if tani_reactant > best_fallback_tani:
                best_fallback_tani = tani_reactant
                best_fallback_idx = i

            # Layer 2: Tanimoto thresholds
            if tani_reactant < self.config.filter_min_tanimoto:
                self._filter_stats['low_tanimoto'] += 1
                continue

            if self.config.filter_max_co_tanimoto < 1.0:
                co_mol = Chem.MolFromSmiles(co)
                if co_mol is not None:
                    tani_co = self._tanimoto(prod_mol, co_mol)
                    if tani_co > self.config.filter_max_co_tanimoto:
                        self._filter_stats['high_co_tanimoto'] += 1
                        continue

            # Layer 3: MW reasonableness
            prod_mw = Descriptors.ExactMolWt(prod_mol)
            mw_delta = abs(prod_mw - reactant_mw)
            if mw_delta > self.config.filter_max_mw_delta:
                self._filter_stats['mw_delta'] += 1
                continue
            if reactant_mw > 0:
                mw_ratio = prod_mw / reactant_mw
                if mw_ratio > self.config.filter_max_mw_ratio:
                    self._filter_stats['mw_ratio_high'] += 1
                    continue
                if mw_ratio < self.config.filter_min_mw_ratio:
                    self._filter_stats['mw_ratio_low'] += 1
                    continue

            # Layer 4: Protecting group detection (Si atoms)
            if self.config.filter_reject_si:
                has_si = any(atom.GetAtomicNum() == 14 for atom in prod_mol.GetAtoms())
                if has_si:
                    self._filter_stats['si_atom'] += 1
                    continue

            kept_co.append(co)
            kept_prod.append(prod)
            kept_scores.append(sc)
            self._filter_stats['passed'] += 1

        # Fallback: if all filtered out, keep best Tanimoto from non-copy candidates
        if not kept_prod and best_fallback_idx is not None:
            kept_co = [co_reactants[best_fallback_idx]]
            kept_prod = [products[best_fallback_idx]]
            kept_scores = [scores[best_fallback_idx]]
            self._filter_stats['fallback'] += 1

        return kept_co, kept_prod, np.array(kept_scores, dtype=np.float32)

    def get_filter_stats(self) -> dict:
        """Return product filter statistics."""
        stats = dict(self._filter_stats)
        total = stats.get('total', 0)
        if total > 0:
            stats['pass_rate'] = stats.get('passed', 0) / total
        return stats

    def reset_filter_stats(self):
        """Reset filter statistics."""
        self._filter_stats.clear()

    def get_valid_actions(self, mol, top_k: int = None) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Get valid reaction actions for a molecule.

        Args:
            mol: RDKit Mol object or SMILES string
            top_k: Number of co-reactants to query from Model 1
                   (default: config.reactant_top_k)

        Returns:
            co_reactants: List of co-reactant SMILES
            products: List of product SMILES (best product per co-reactant)
            scores: np.ndarray of combined scores
        """
        if not self._loaded:
            self.load()

        if top_k is None:
            top_k = self.config.reactant_top_k

        if isinstance(mol, str):
            mol_smiles = mol
        else:
            mol_smiles = Chem.MolToSmiles(mol)

        canon_mol = canonicalize_smiles(mol_smiles)
        if canon_mol is None:
            return [], [], np.array([])

        # Step 1: Get co-reactants from Model 1 (with optional oversampling)
        initial_top_k = top_k * self.config.co_reactant_oversample
        reactant_results = self.reactant_predictor.predict(canon_mol, top_k=initial_top_k)
        if not reactant_results:
            return [], [], np.array([])

        # Plan C: Apply MW-based re-ranking if enabled
        if self.config.mw_penalty_alpha > 0:
            reactant_results = self._apply_mw_reranking(reactant_results, top_k)

        co_reactant_smiles_list = [r[0] for r in reactant_results]
        reactant_scores = np.array([r[1] for r in reactant_results])

        # Step 2: Predict products from Model 2 (batch)
        mol_list = [canon_mol] * len(co_reactant_smiles_list)

        # Use greedy decoding for speed in RL (num_beams=1, num_return=1)
        product_results = self.product_predictor.predict_batch(
            mol_list, co_reactant_smiles_list,
            num_beams=self.config.product_config.num_beams,
            num_return=1,
        )

        # Assemble results (with Plan B fragment expansion)
        co_reactants = []
        products = []
        scores = []

        for i, (co_react, prod_list) in enumerate(zip(co_reactant_smiles_list, product_results)):
            if not prod_list:
                continue

            best_product, prod_score = prod_list[0]
            r_score = float(reactant_scores[i])

            for product_smiles, score in self._expand_product(best_product, canon_mol, r_score):
                co_reactants.append(co_react)
                products.append(product_smiles)
                scores.append(score)

        # Apply product validation filter
        co_reactants, products, scores_arr = self._filter_product_list(
            canon_mol, co_reactants, products, np.array(scores, dtype=np.float32))
        return co_reactants, products, scores_arr

    def get_valid_actions_batch(self, mols, top_k: int = None) -> List[Tuple[List[str], List[str], np.ndarray]]:
        """
        Batch get valid reaction actions for multiple molecules.

        Combines all molecules into single GNN and T5 batch calls for efficiency.

        Args:
            mols: List of RDKit Mol objects or SMILES strings
            top_k: Number of co-reactants per molecule

        Returns:
            List of (co_reactants, products, scores) tuples, one per molecule.
        """
        if not self._loaded:
            self.load()

        if top_k is None:
            top_k = self.config.reactant_top_k

        # Canonicalize all molecules
        canon_mols = []
        for mol in mols:
            if isinstance(mol, str):
                mol_smiles = mol
            else:
                mol_smiles = Chem.MolToSmiles(mol)
            canon_mols.append(canonicalize_smiles(mol_smiles))

        # Step 1: Batch co-reactant prediction (Model 1) with optional oversampling
        initial_top_k = top_k * self.config.co_reactant_oversample
        if hasattr(self.reactant_predictor, 'predict_batch'):
            all_reactant_results = self.reactant_predictor.predict_batch(
                [cm if cm is not None else '' for cm in canon_mols], top_k=initial_top_k
            )
        else:
            # Fallback to serial for predictors without batch support
            all_reactant_results = []
            for cm in canon_mols:
                if cm is not None:
                    all_reactant_results.append(self.reactant_predictor.predict(cm, top_k=initial_top_k))
                else:
                    all_reactant_results.append([])

        # Plan C: Apply MW-based re-ranking if enabled
        if self.config.mw_penalty_alpha > 0:
            all_reactant_results = [
                self._apply_mw_reranking(rr, top_k) for rr in all_reactant_results
            ]

        # Step 2: Flatten all (mol, co-reactant) pairs for batch T5 inference
        flat_mol_smiles = []
        flat_co_smiles = []
        flat_reactant_scores = []
        # Track which molecule each pair belongs to: (mol_idx, pair_idx_within_mol)
        pair_to_mol = []

        for mol_idx, reactant_results in enumerate(all_reactant_results):
            if canon_mols[mol_idx] is None or not reactant_results:
                continue
            for pair_idx, (co_smiles, r_score) in enumerate(reactant_results):
                flat_mol_smiles.append(canon_mols[mol_idx])
                flat_co_smiles.append(co_smiles)
                flat_reactant_scores.append(r_score)
                pair_to_mol.append((mol_idx, pair_idx))

        # Initialize results for all molecules
        all_results = [([], [], np.array([], dtype=np.float32)) for _ in range(len(mols))]

        if not flat_mol_smiles:
            return all_results

        # Single batch T5 call for all pairs
        product_results = self.product_predictor.predict_batch(
            flat_mol_smiles, flat_co_smiles,
            num_beams=self.config.product_config.num_beams,
            num_return=1,
        )

        # Re-group results by molecule
        mol_co_reactants = [[] for _ in range(len(mols))]
        mol_products = [[] for _ in range(len(mols))]
        mol_scores = [[] for _ in range(len(mols))]

        for flat_idx, ((mol_idx, _), prod_list) in enumerate(zip(pair_to_mol, product_results)):
            if not prod_list:
                continue

            best_product, prod_score = prod_list[0]
            r_score = float(flat_reactant_scores[flat_idx])

            # Plan B: expand fragments via _expand_product
            for product_smiles, score in self._expand_product(
                    best_product, canon_mols[mol_idx], r_score):
                mol_co_reactants[mol_idx].append(flat_co_smiles[flat_idx])
                mol_products[mol_idx].append(product_smiles)
                mol_scores[mol_idx].append(score)

        # Convert to output format and apply product validation filter
        for i in range(len(mols)):
            co_r, prods, sc = mol_co_reactants[i], mol_products[i], np.array(mol_scores[i], dtype=np.float32)
            if canon_mols[i] is not None and prods:
                co_r, prods, sc = self._filter_product_list(canon_mols[i], co_r, prods, sc)
            all_results[i] = (co_r, prods, sc)

        return all_results

    def get_stats(self):
        """Get performance statistics from both models."""
        stats = {}
        if self.product_predictor:
            stats['product_predictor'] = self.product_predictor.get_stats()
        if self._filter_stats:
            stats['product_filter'] = self.get_filter_stats()
        return stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--reactant_method', type=str, default='hypergraph',
                        choices=['hypergraph', 'fingerprint'])
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_beams', type=int, default=1,
                        help="Beam size for product prediction (1=greedy, 5=beam)")
    args = parser.parse_args()

    config = ReactionPredictorConfig(
        reactant_method=args.reactant_method,
        reactant_top_k=args.top_k,
    )
    config.product_config.num_beams = args.num_beams
    config.product_config.num_return_sequences = 1

    predictor = ReactionPredictor(config=config, device=args.device)

    test_mols = ["CCO", "c1ccccc1Br", "CC(=O)O", "c1ccccc1N"]

    for mol in test_mols:
        print(f"\n{'='*60}")
        print(f"Input: {mol}")
        print(f"{'='*60}")

        co_reactants, products, scores = predictor.get_valid_actions(mol, top_k=args.top_k)

        if not co_reactants:
            print("  No reactions found.")
            continue

        for i, (co, prod, score) in enumerate(zip(co_reactants, products, scores)):
            print(f"  [{i+1}] {mol} + {co}")
            print(f"       -> {prod}  (score={score:.4f})")

    print(f"\nStats: {predictor.get_stats()}")
