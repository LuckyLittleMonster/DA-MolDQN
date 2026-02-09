"""
Model 1: ReactantPredictor
Predicts which molecules can react with a given input molecule.

Three implementations:
  1a. HypergraphReactantPredictor - Hypergraph link prediction (our method)
  1b. SEALReactantPredictor - GNN link prediction (AstraZeneca baseline, offline only)
  1c. FingerprintReactantPredictor - Morgan fingerprint similarity (simple baseline)
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model_reactions.config import ReactantPredictorConfig


# =============================================================================
# Base class
# =============================================================================

class ReactantPredictor(ABC):
    """Abstract base class for reactant prediction."""
    
    @abstractmethod
    def predict(self, mol, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict co-reactants for a given molecule.
        
        Args:
            mol: RDKit Mol object or SMILES string
            top_k: Number of co-reactants to return
            
        Returns:
            List of (co_reactant_smiles, score) tuples, sorted by score descending
        """
        pass
    
    @abstractmethod
    def build_database(self, data_dir: str):
        """Build the molecule database from reaction data."""
        pass


# =============================================================================
# Reaction Database (shared by all methods)
# =============================================================================

class ReactionDatabase:
    """
    Database of reactions from USPTO.
    
    Stores:
    - All unique reactant molecules and their SMILES
    - Reaction graph: which molecules participate in the same reaction
    - For each molecule: list of known reaction partners
    """
    
    def __init__(self, data_dir: str = "Data/uspto"):
        self.data_dir = Path(data_dir)
        self.reactant_smiles = []        # List of unique reactant SMILES
        self.smiles_to_idx = {}          # SMILES -> index
        self.reaction_partners = {}      # idx -> set of partner indices
        self.reactions = []              # List of (reactant_indices, product_smiles, info)
        
    def build(self):
        """Build database from USPTO CSV files."""
        print("Building reaction database...")
        start = time.time()
        
        all_reactants = set()
        raw_reactions = []
        
        for split in ['train', 'val', 'test']:
            csv_path = self.data_dir / f"{split}.csv"
            if not csv_path.exists():
                print(f"  Warning: {csv_path} not found, skipping")
                continue
            
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rxn_smiles = str(row.get('rxn_smiles', ''))
                if '>>' not in rxn_smiles:
                    continue
                
                parts = rxn_smiles.split('>>')
                reactants = [s.strip() for s in parts[0].split('.') if s.strip()]
                products = [s.strip() for s in parts[1].split('.') if s.strip()] if len(parts) > 1 else []
                
                # Canonicalize
                canon_reactants = []
                for r in reactants:
                    mol = Chem.MolFromSmiles(r)
                    if mol is not None:
                        canon_reactants.append(Chem.MolToSmiles(mol))
                
                if len(canon_reactants) >= 2:
                    all_reactants.update(canon_reactants)
                    raw_reactions.append({
                        'reactants': canon_reactants,
                        'products': products,
                        'split': split,
                        'rxn_class': int(row.get('rxn_class', row.get('class', 0))),
                    })
        
        # Build index
        self.reactant_smiles = sorted(list(all_reactants))
        self.smiles_to_idx = {s: i for i, s in enumerate(self.reactant_smiles)}
        
        # Build reaction partners (adjacency)
        self.reaction_partners = {i: set() for i in range(len(self.reactant_smiles))}
        
        for rxn in raw_reactions:
            indices = [self.smiles_to_idx[s] for s in rxn['reactants'] if s in self.smiles_to_idx]
            # All reactants in this reaction are partners of each other
            for i in range(len(indices)):
                for j in range(len(indices)):
                    if i != j:
                        self.reaction_partners[indices[i]].add(indices[j])
            
            self.reactions.append({
                'reactant_indices': indices,
                'products': rxn['products'],
                'split': rxn['split'],
                'rxn_class': rxn['rxn_class'],
            })
        
        elapsed = time.time() - start
        print(f"  {len(self.reactant_smiles)} unique reactants")
        print(f"  {len(self.reactions)} reactions")
        n_edges = sum(len(v) for v in self.reaction_partners.values()) // 2
        print(f"  {n_edges} reaction partnerships (edges)")
        print(f"  Built in {elapsed:.1f}s")
        
        return self
    
    def get_partners(self, smiles: str) -> List[int]:
        """Get indices of known reaction partners for a molecule."""
        idx = self.smiles_to_idx.get(smiles)
        if idx is None:
            return []
        return list(self.reaction_partners[idx])
    
    def get_train_test_edges(self):
        """
        Get train and test edges for benchmark evaluation.
        
        Returns:
            train_pos: np.array of shape (N, 2) - positive train edges
            test_pos: np.array of shape (M, 2) - positive test edges
        """
        train_edges = []
        test_edges = []
        
        for rxn in self.reactions:
            indices = rxn['reactant_indices']
            split = rxn['split']
            
            # Create pairwise edges from multi-reactant reactions
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edge = (indices[i], indices[j])
                    if split == 'test':
                        test_edges.append(edge)
                    else:
                        train_edges.append(edge)
        
        # Deduplicate
        train_edges = list(set(train_edges))
        test_edges = list(set(test_edges))
        
        return np.array(train_edges), np.array(test_edges)
    
    def sample_negative_edges(self, n_samples: int, exclude_edges: set = None, seed: int = 42) -> np.ndarray:
        """
        Sample negative edges (pairs of molecules that don't react).
        
        Args:
            n_samples: Number of negative edges to sample
            exclude_edges: Set of (i, j) tuples to exclude
            seed: Random seed
            
        Returns:
            np.array of shape (n_samples, 2) - negative edges
        """
        rng = np.random.RandomState(seed)
        n_mols = len(self.reactant_smiles)
        
        if exclude_edges is None:
            exclude_edges = set()
            for i, partners in self.reaction_partners.items():
                for j in partners:
                    if i < j:
                        exclude_edges.add((i, j))
        
        neg_edges = []
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(neg_edges) < n_samples and attempts < max_attempts:
            i = rng.randint(0, n_mols)
            j = rng.randint(0, n_mols)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (i, j) not in exclude_edges:
                neg_edges.append((i, j))
                exclude_edges.add((i, j))
            attempts += 1
        
        return np.array(neg_edges[:n_samples])
    
    def save(self, path: str):
        """Save database to pickle."""
        with open(path, 'wb') as f:
            pickle.dump({
                'reactant_smiles': self.reactant_smiles,
                'smiles_to_idx': self.smiles_to_idx,
                'reaction_partners': self.reaction_partners,
                'reactions': self.reactions,
            }, f)
    
    def load(self, path: str):
        """Load database from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.reactant_smiles = data['reactant_smiles']
        self.smiles_to_idx = data['smiles_to_idx']
        self.reaction_partners = data['reaction_partners']
        self.reactions = data['reactions']
        return self


# =============================================================================
# 1c. FingerprintReactantPredictor (simple baseline)
# =============================================================================

class FingerprintReactantPredictor(ReactantPredictor):
    """
    Predict co-reactants using Morgan fingerprint similarity.
    
    This is a simple baseline: molecules with similar structure 
    to known reaction partners are predicted as potential co-reactants.
    
    Two strategies:
    1. Direct similarity: find database molecules most similar to input
    2. Partner similarity: find database molecules whose known partners 
       are most similar to input (more chemically meaningful)
    """
    
    def __init__(self, config: ReactantPredictorConfig = None, strategy: str = 'partner'):
        if config is None:
            config = ReactantPredictorConfig()
        self.config = config
        self.strategy = strategy  # 'direct' or 'partner'
        
        self.db = None
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            fpSize=config.fp_length, radius=config.fp_radius)
        
        # Pre-computed fingerprints as numpy array for fast similarity
        self.fp_matrix = None  # shape: (n_mols, fp_length)
        self.valid_indices = None  # indices of molecules with valid fingerprints
    
    def build_database(self, data_dir: str = None):
        """Build fingerprint database from reaction data."""
        if data_dir is None:
            data_dir = self.config.data_dir
        
        # Build or load reaction database
        cache_path = os.path.join(data_dir, 'reaction_db_cache.pkl')
        self.db = ReactionDatabase(data_dir)
        
        if os.path.exists(cache_path):
            print("Loading cached reaction database...")
            self.db.load(cache_path)
        else:
            self.db.build()
            self.db.save(cache_path)
        
        # Pre-compute fingerprints
        self._compute_fingerprints()
        
        return self
    
    def _compute_fingerprints(self):
        """Pre-compute Morgan fingerprints for all molecules."""
        print("Pre-computing fingerprints...")
        start = time.time()
        
        n_mols = len(self.db.reactant_smiles)
        fp_length = self.config.fp_length
        
        self.fp_matrix = np.zeros((n_mols, fp_length), dtype=np.float32)
        self.valid_indices = []
        
        for i, smiles in enumerate(self.db.reactant_smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            try:
                fp = self.fp_gen.GetFingerprint(mol)
                arr = np.zeros(fp_length, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                self.fp_matrix[i] = arr
                self.valid_indices.append(i)
            except Exception:
                continue
        
        self.valid_indices = np.array(self.valid_indices)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.fp_matrix, axis=1, keepdims=True)
        self.fp_matrix_norm = self.fp_matrix / np.where(norms > 0, norms, 1.0)
        
        elapsed = time.time() - start
        print(f"  Computed fingerprints for {len(self.valid_indices)}/{n_mols} molecules in {elapsed:.1f}s")
    
    def _get_fingerprint(self, mol) -> Optional[np.ndarray]:
        """Get fingerprint for a molecule."""
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        try:
            fp = self.fp_gen.GetFingerprint(mol)
            arr = np.zeros(self.config.fp_length, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return None
    
    def predict(self, mol, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict co-reactants using fingerprint similarity.
        
        Args:
            mol: RDKit Mol or SMILES string
            top_k: Number of co-reactants to return
            
        Returns:
            List of (co_reactant_smiles, score) tuples
        """
        if self.db is None:
            raise RuntimeError("Database not built. Call build_database() first.")
        
        if isinstance(mol, str):
            smiles = mol
            mol = Chem.MolFromSmiles(mol)
        else:
            smiles = Chem.MolToSmiles(mol)
        
        fp = self._get_fingerprint(mol)
        if fp is None:
            return []
        
        if self.strategy == 'direct':
            return self._predict_direct(fp, smiles, top_k)
        else:
            return self._predict_partner(fp, smiles, top_k)
    
    def _predict_direct(self, fp: np.ndarray, smiles: str, top_k: int) -> List[Tuple[str, float]]:
        """Direct strategy: find most similar molecules in database."""
        fp_norm = fp / (np.linalg.norm(fp) + 1e-8)
        
        # Cosine similarity with all database molecules
        sims = self.fp_matrix_norm @ fp_norm
        
        # Exclude self
        idx = self.db.smiles_to_idx.get(smiles)
        if idx is not None:
            sims[idx] = -1.0
        
        # Top-k
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            if sims[i] > 0:
                results.append((self.db.reactant_smiles[i], float(sims[i])))
        
        return results
    
    def _predict_partner(self, fp: np.ndarray, smiles: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Partner strategy: find molecules whose known partners are similar to input.
        
        Logic: If molecule B's known partner A is similar to our input X,
        then B might also react with X.
        """
        fp_norm = fp / (np.linalg.norm(fp) + 1e-8)
        
        # Find molecules most similar to input
        sims = self.fp_matrix_norm @ fp_norm
        
        # For each similar molecule, collect their reaction partners
        # Weight by similarity
        n_candidates = min(50, len(self.valid_indices))
        top_similar = np.argsort(sims)[::-1][:n_candidates]
        
        partner_scores = {}
        
        for sim_idx in top_similar:
            sim_score = sims[sim_idx]
            if sim_score <= 0:
                break
            
            # Get partners of this similar molecule
            partners = self.db.reaction_partners.get(sim_idx, set())
            for p in partners:
                if p not in partner_scores:
                    partner_scores[p] = 0.0
                partner_scores[p] += sim_score
        
        # Exclude self
        idx = self.db.smiles_to_idx.get(smiles)
        if idx is not None and idx in partner_scores:
            del partner_scores[idx]
        
        # Sort by score
        sorted_partners = sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for p_idx, score in sorted_partners[:top_k]:
            results.append((self.db.reactant_smiles[p_idx], float(score)))
        
        return results
    
    def predict_batch(self, mols, top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Batch predict co-reactants for multiple molecules.

        One matrix multiply for all similarities.

        Args:
            mols: List of RDKit Mol objects or SMILES strings
            top_k: Number of co-reactants per molecule

        Returns:
            List of List of (co_reactant_smiles, score) tuples
        """
        if self.db is None:
            raise RuntimeError("Database not built. Call build_database() first.")

        # Compute fingerprints for all query molecules
        fps = []
        smiles_list = []
        for mol in mols:
            if isinstance(mol, str):
                smiles_list.append(mol)
                mol = Chem.MolFromSmiles(mol)
            else:
                smiles_list.append(Chem.MolToSmiles(mol))
            fp = self._get_fingerprint(mol)
            if fp is not None:
                fps.append(fp)
            else:
                fps.append(np.zeros(self.config.fp_length, dtype=np.float32))

        query_matrix = np.array(fps)  # (N, fp_length)
        query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        query_matrix_norm = query_matrix / np.where(query_norms > 0, query_norms, 1e-8)

        if self.strategy == 'direct':
            # Batch cosine similarity: (N, fp_length) @ (fp_length, M) = (N, M)
            sims_matrix = query_matrix_norm @ self.fp_matrix_norm.T

            all_results = []
            for i in range(len(mols)):
                sims = sims_matrix[i].copy()
                # Exclude self
                idx = self.db.smiles_to_idx.get(smiles_list[i])
                if idx is not None:
                    sims[idx] = -1.0
                top_indices = np.argsort(sims)[::-1][:top_k]
                results = [(self.db.reactant_smiles[j], float(sims[j]))
                           for j in top_indices if sims[j] > 0]
                all_results.append(results)
            return all_results
        else:
            # Partner strategy: fall back to per-molecule (depends on graph traversal)
            return [self.predict(mol, top_k=top_k) for mol in mols]

    def predict_link_probability(self, mol_a_smiles: str, mol_b_smiles: str) -> float:
        """
        Predict the probability that two molecules can react.
        Uses Tanimoto similarity as a proxy.
        """
        fp_a = self._get_fingerprint(mol_a_smiles)
        fp_b = self._get_fingerprint(mol_b_smiles)
        
        if fp_a is None or fp_b is None:
            return 0.0
        
        # Tanimoto similarity
        dot = np.dot(fp_a, fp_b)
        denom = np.sum(fp_a) + np.sum(fp_b) - dot
        if denom == 0:
            return 0.0
        return float(dot / denom)
    
    def evaluate_link_prediction(self, test_pos: np.ndarray, test_neg: np.ndarray) -> Dict:
        """
        Evaluate link prediction performance on test set.
        
        Args:
            test_pos: Positive test edges (N, 2)
            test_neg: Negative test edges (M, 2)
            
        Returns:
            Dict with ROC-AUC, AP, and other metrics
        """
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
        
        print(f"Evaluating on {len(test_pos)} positive and {len(test_neg)} negative edges...")
        
        # Compute similarity scores for all test edges
        y_true = []
        y_scores = []
        
        for i, j in test_pos:
            score = self._compute_pair_score(i, j)
            y_true.append(1)
            y_scores.append(score)
        
        for i, j in test_neg:
            score = self._compute_pair_score(i, j)
            y_true.append(0)
            y_scores.append(score)
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        roc_auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        # Accuracy at various thresholds
        results = {
            'roc_auc': roc_auc,
            'average_precision': ap,
            'n_pos': len(test_pos),
            'n_neg': len(test_neg),
        }
        
        for threshold in [0.1, 0.2, 0.3, 0.5]:
            y_pred = (y_scores >= threshold).astype(int)
            results[f'accuracy@{threshold}'] = accuracy_score(y_true, y_pred)
        
        return results
    
    def _compute_pair_score(self, idx_a: int, idx_b: int) -> float:
        """Compute similarity score for a pair of molecule indices."""
        fp_a = self.fp_matrix[idx_a]
        fp_b = self.fp_matrix[idx_b]
        
        # Tanimoto on binary fingerprints
        dot = np.dot(fp_a, fp_b)
        denom = np.sum(fp_a) + np.sum(fp_b) - dot
        if denom == 0:
            return 0.0
        return float(dot / denom)


# =============================================================================
# Factory function
# =============================================================================

def create_reactant_predictor(method: str, config: ReactantPredictorConfig = None) -> ReactantPredictor:
    """Create a ReactantPredictor by method name."""
    if config is None:
        config = ReactantPredictorConfig()
    
    if method == 'fingerprint' or method == 'fingerprint_direct':
        return FingerprintReactantPredictor(config, strategy='direct')
    elif method == 'fingerprint_partner':
        return FingerprintReactantPredictor(config, strategy='partner')
    elif method == 'hypergraph':
        # TODO: implement
        raise NotImplementedError("HypergraphReactantPredictor not yet implemented")
    elif method == 'seal':
        # TODO: implement
        raise NotImplementedError("SEALReactantPredictor not yet implemented")
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Benchmark script
# =============================================================================

def benchmark_fingerprint(data_dir: str = "Data/uspto"):
    """Run benchmark for FingerprintReactantPredictor."""
    print("=" * 60)
    print("Benchmarking FingerprintReactantPredictor")
    print("=" * 60)
    
    # Build database
    predictor = FingerprintReactantPredictor(
        ReactantPredictorConfig(data_dir=data_dir, fp_length=2048, fp_radius=2),
        strategy='direct'
    )
    predictor.build_database(data_dir)
    
    # Get train/test edges
    train_pos, test_pos = predictor.db.get_train_test_edges()
    print(f"\nTrain edges: {len(train_pos)}, Test edges: {len(test_pos)}")
    
    # Sample negative edges (same count as positive for balanced evaluation)
    all_pos_set = set()
    for i, j in train_pos:
        all_pos_set.add((min(i, j), max(i, j)))
    for i, j in test_pos:
        all_pos_set.add((min(i, j), max(i, j)))
    
    test_neg = predictor.db.sample_negative_edges(len(test_pos), exclude_edges=all_pos_set)
    print(f"Negative test edges: {len(test_neg)}")
    
    # Evaluate
    results = predictor.evaluate_link_prediction(test_pos, test_neg)
    
    print(f"\n{'='*60}")
    print(f"Results: FingerprintReactantPredictor (direct Tanimoto)")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data/uspto')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_fingerprint(args.data_dir)
    else:
        # Quick test
        config = ReactantPredictorConfig(data_dir=args.data_dir)
        pred = FingerprintReactantPredictor(config, strategy='partner')
        pred.build_database()
        
        test_mol = "CCO"
        print(f"\nPredicting co-reactants for: {test_mol}")
        results = pred.predict(test_mol, top_k=5)
        for smiles, score in results:
            print(f"  {smiles} (score={score:.4f})")
