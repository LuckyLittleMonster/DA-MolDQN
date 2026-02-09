"""
Model 2: ProductPredictor - Predicts reaction products using ReactionT5v2.

Given a molecule and a co-reactant, predicts the product of their reaction.
Wraps the pre-trained ReactionT5v2-forward model with:
  - Batch inference for efficiency
  - LRU cache to avoid redundant inference in RL training
  - Greedy / beam search modes

Usage:
    from model_reactions.product_prediction.product_predictor import ProductPredictor

    predictor = ProductPredictor(device='cuda')
    results = predictor.predict("CCO", "CC(=O)Cl")
    # [("CCOC(C)=O", -0.12), ...]
"""

import time
import torch
import numpy as np
from typing import List, Tuple, Dict

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model_reactions.config import ProductPredictorConfig
from model_reactions.utils import canonicalize_smiles, LRUCache  # noqa: F401 — re-exported


class ProductPredictor:
    """
    Predict reaction products using ReactionT5v2.

    Input format for ReactionT5v2-forward: "REACTANT:{mol}.{co_reactant} REAGENT:"
    Output: product SMILES
    """

    def __init__(self, config: ProductPredictorConfig = None, device: str = 'auto'):
        if config is None:
            config = ProductPredictorConfig()
        self.config = config

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.tokenizer = None
        self.model = None
        self._loaded = False

        # LRU cache: key = (mol_smiles, co_reactant_smiles, num_beams), value = results
        self.cache = LRUCache(config.cache_size)

        # Statistics
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'inference_calls': 0,
            'total_inference_time': 0.0,
        }

    def load_model(self):
        """Load the ReactionT5v2 model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = self.config.model_name
        print(f"Loading ReactionT5v2 from {model_name}...")
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_tensors="pt")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        elapsed = time.time() - start
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  Model loaded: {n_params:.1f}M params, device={self.device}, time={elapsed:.1f}s")

        self._loaded = True

    def _format_input(self, mol_smiles: str, co_reactant_smiles: str) -> str:
        """Format input for ReactionT5v2-forward."""
        return f"REACTANT:{mol_smiles}.{co_reactant_smiles} REAGENT:"

    def predict(self, mol_smiles: str, co_reactant_smiles: str,
                num_beams: int = None, num_return: int = None) -> List[Tuple[str, float]]:
        """
        Predict products for a reaction between mol and co_reactant.

        Args:
            mol_smiles: Input molecule SMILES
            co_reactant_smiles: Co-reactant SMILES
            num_beams: Number of beams for beam search (None = use config default)
            num_return: Number of products to return (None = use config default)

        Returns:
            List of (product_smiles, score) tuples, sorted by score descending.
            Score is the log-probability from beam search (higher = more confident).
        """
        if not self._loaded:
            self.load_model()

        if num_beams is None:
            num_beams = self.config.num_beams
        if num_return is None:
            num_return = self.config.num_return_sequences

        # Canonicalize inputs
        canon_mol = canonicalize_smiles(mol_smiles)
        canon_co = canonicalize_smiles(co_reactant_smiles)
        if canon_mol is None or canon_co is None:
            return []

        self.stats['total_calls'] += 1

        # Check cache
        cache_key = (canon_mol, canon_co, num_beams)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats['cache_hits'] += 1
            return cached[:num_return]

        # Run inference
        input_text = self._format_input(canon_mol, canon_co)
        results = self._inference_single(input_text, num_beams, num_return)

        # Cache results
        self.cache.put(cache_key, results)

        return results[:num_return]

    def predict_batch(self, mol_list: List[str], co_reactant_list: List[str],
                      num_beams: int = None, num_return: int = None) -> List[List[Tuple[str, float]]]:
        """
        Batch predict products for multiple (mol, co_reactant) pairs.

        Args:
            mol_list: List of molecule SMILES
            co_reactant_list: List of co-reactant SMILES (same length)
            num_beams: Number of beams
            num_return: Number of products per pair

        Returns:
            List of List of (product_smiles, score) tuples.
        """
        if not self._loaded:
            self.load_model()

        if num_beams is None:
            num_beams = self.config.num_beams
        if num_return is None:
            num_return = self.config.num_return_sequences

        assert len(mol_list) == len(co_reactant_list), \
            "mol_list and co_reactant_list must have the same length"

        # Separate cached vs uncached
        all_results = [None] * len(mol_list)
        uncached_indices = []
        uncached_inputs = []

        for i, (mol, co) in enumerate(zip(mol_list, co_reactant_list)):
            canon_mol = canonicalize_smiles(mol)
            canon_co = canonicalize_smiles(co)

            self.stats['total_calls'] += 1

            if canon_mol is None or canon_co is None:
                all_results[i] = []
                continue

            cache_key = (canon_mol, canon_co, num_beams)
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.stats['cache_hits'] += 1
                all_results[i] = cached[:num_return]
            else:
                uncached_indices.append(i)
                uncached_inputs.append(self._format_input(canon_mol, canon_co))

        # Batch inference for uncached pairs
        if uncached_inputs:
            batch_results = self._inference_batch(uncached_inputs, num_beams, num_return)

            for idx, results in zip(uncached_indices, batch_results):
                all_results[idx] = results[:num_return]

                # Cache
                canon_mol = canonicalize_smiles(mol_list[idx])
                canon_co = canonicalize_smiles(co_reactant_list[idx])
                if canon_mol and canon_co:
                    cache_key = (canon_mol, canon_co, num_beams)
                    self.cache.put(cache_key, results)

        return all_results

    def _inference_single(self, input_text: str, num_beams: int,
                          num_return: int) -> List[Tuple[str, float]]:
        """Run inference on a single input."""
        return self._inference_batch([input_text], num_beams, num_return)[0]

    def _inference_batch(self, input_texts: List[str], num_beams: int,
                         num_return: int) -> List[List[Tuple[str, float]]]:
        """
        Run batch inference with automatic sub-batching to prevent OOM.

        Splits large batches into chunks of max_sub_batch and concatenates results.
        """
        max_sub = self.config.max_sub_batch
        if len(input_texts) <= max_sub:
            return self._inference_batch_core(input_texts, num_beams, num_return)

        all_results = []
        for i in range(0, len(input_texts), max_sub):
            sub_texts = input_texts[i:i + max_sub]
            sub_results = self._inference_batch_core(sub_texts, num_beams, num_return)
            all_results.extend(sub_results)
        return all_results

    def _inference_batch_core(self, input_texts: List[str], num_beams: int,
                              num_return: int) -> List[List[Tuple[str, float]]]:
        """
        Run batch inference on a single sub-batch.

        Args:
            input_texts: List of formatted input strings
            num_beams: Number of beams
            num_return: Number of sequences to return per input

        Returns:
            List of List of (product_smiles, score) tuples
        """
        self.stats['inference_calls'] += 1
        start = time.time()

        # Tokenize
        inputs = self.tokenizer(
            input_texts,
            add_special_tokens=True,
            max_length=self.config.input_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            if num_beams > 1:
                output = self.model.generate(
                    **inputs,
                    max_length=self.config.output_max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sequences = output.sequences
                scores = output.sequences_scores.tolist()
            else:
                # Greedy decoding (faster for RL training)
                output = self.model.generate(
                    **inputs,
                    max_length=self.config.output_max_length,
                    do_sample=False,
                )
                sequences = output
                scores = [0.0] * len(sequences)

        # Decode
        decoded = [
            self.tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "").rstrip(".")
            for seq in sequences
        ]

        elapsed = time.time() - start
        self.stats['total_inference_time'] += elapsed

        # Group by input (each input produces num_return sequences)
        batch_size = len(input_texts)
        all_results = []

        for i in range(batch_size):
            results = []
            for j in range(num_return):
                seq_idx = i * num_return + j
                if seq_idx >= len(decoded):
                    break

                product_smiles = decoded[seq_idx]
                score = scores[seq_idx] if seq_idx < len(scores) else 0.0

                # Validate product
                canon = canonicalize_smiles(product_smiles)
                if canon is not None:
                    results.append((canon, float(score)))

            all_results.append(results)

        return all_results

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        stats = dict(self.stats)
        if stats['total_calls'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calls']
        if stats['inference_calls'] > 0:
            stats['avg_inference_time'] = stats['total_inference_time'] / stats['inference_calls']
        stats['cache_size'] = len(self.cache)
        return stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_beams', type=int, default=5)
    args = parser.parse_args()

    predictor = ProductPredictor(device=args.device)

    # Test single prediction
    test_pairs = [
        ("CCO", "CC(=O)Cl"),          # ethanol + acetyl chloride
        ("c1ccccc1Br", "OB(O)c1ccccc1"),  # bromobenzene + phenylboronic acid (Suzuki)
        ("CC(=O)O", "CCN"),            # acetic acid + ethylamine
        ("c1ccccc1N", "CC(=O)Cl"),     # aniline + acetyl chloride
    ]

    print("=== Single predictions ===")
    for mol, co_react in test_pairs:
        results = predictor.predict(mol, co_react, num_beams=args.num_beams, num_return=3)
        print(f"\n{mol} + {co_react}:")
        for prod, score in results:
            print(f"  -> {prod}  (score={score:.4f})")

    # Test batch prediction
    print("\n=== Batch prediction ===")
    mols = [p[0] for p in test_pairs]
    cos = [p[1] for p in test_pairs]
    batch_results = predictor.predict_batch(mols, cos, num_beams=args.num_beams, num_return=2)
    for i, (mol, co, results) in enumerate(zip(mols, cos, batch_results)):
        print(f"\n{mol} + {co}:")
        for prod, score in results:
            print(f"  -> {prod}  (score={score:.4f})")

    # Test cache
    print("\n=== Cache test ===")
    predictor.predict("CCO", "CC(=O)Cl")  # Should be cached
    print(f"Stats: {predictor.get_stats()}")
