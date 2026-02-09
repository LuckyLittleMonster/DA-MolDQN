"""
RetroPredictor - Predicts reactants from products using ReactionT5v2-retrosynthesis.

Given a product SMILES, predicts the reactants that could have produced it.
Wraps the pre-trained ReactionT5v2-retrosynthesis model with:
  - Batch inference for efficiency
  - LRU cache to avoid redundant inference
  - Round-trip validation (retro -> forward -> compare)

Usage:
    from model_reactions.product_prediction.retro_predictor import RetroPredictor

    predictor = RetroPredictor(device='cuda')
    results = predictor.predict("CCOC(C)=O")
    # [("CCO.CC(=O)Cl", -0.12), ...]
"""

import time
import torch
from typing import List, Tuple, Dict

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model_reactions.config import RetroPredictorConfig
from model_reactions.utils import canonicalize_smiles, tanimoto_similarity, LRUCache


class RetroPredictor:
    """
    Predict reactants from products using ReactionT5v2-retrosynthesis.

    Input format: raw product SMILES (no prefix needed)
    Output: reactant SMILES (dot-separated for multiple reactants)
    """

    def __init__(self, config: RetroPredictorConfig = None, device: str = 'auto'):
        if config is None:
            config = RetroPredictorConfig()
        self.config = config

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.tokenizer = None
        self.model = None
        self._loaded = False

        # LRU cache: key = (product_smiles, num_beams), value = results
        self.cache = LRUCache(config.cache_size)

        # Statistics
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'inference_calls': 0,
            'total_inference_time': 0.0,
        }

    def load_model(self):
        """Load the ReactionT5v2-retrosynthesis model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = self.config.model_name
        print(f"Loading ReactionT5v2-retrosynthesis from {model_name}...")
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_tensors="pt")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        elapsed = time.time() - start
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  Model loaded: {n_params:.1f}M params, device={self.device}, time={elapsed:.1f}s")

        self._loaded = True

    def predict(self, product_smiles: str,
                num_beams: int = None, num_return: int = None) -> List[Tuple[str, float]]:
        """
        Predict reactants for a given product.

        Args:
            product_smiles: Product SMILES
            num_beams: Number of beams for beam search (None = use config default)
            num_return: Number of predictions to return (None = use config default)

        Returns:
            List of (reactants_smiles, score) tuples, sorted by score descending.
            reactants_smiles is dot-separated if multiple reactants.
        """
        if not self._loaded:
            self.load_model()

        if num_beams is None:
            num_beams = self.config.num_beams
        if num_return is None:
            num_return = self.config.num_return_sequences

        # Canonicalize input
        canon = canonicalize_smiles(product_smiles)
        if canon is None:
            return []

        self.stats['total_calls'] += 1

        # Check cache
        cache_key = (canon, num_beams)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats['cache_hits'] += 1
            return cached[:num_return]

        # Run inference
        results = self._inference_single(canon, num_beams, num_return)

        # Cache results
        self.cache.put(cache_key, results)

        return results[:num_return]

    def predict_batch(self, product_list: List[str],
                      num_beams: int = None, num_return: int = None) -> List[List[Tuple[str, float]]]:
        """
        Batch predict reactants for multiple products.

        Args:
            product_list: List of product SMILES
            num_beams: Number of beams
            num_return: Number of predictions per product

        Returns:
            List of List of (reactants_smiles, score) tuples.
        """
        if not self._loaded:
            self.load_model()

        if num_beams is None:
            num_beams = self.config.num_beams
        if num_return is None:
            num_return = self.config.num_return_sequences

        # Separate cached vs uncached
        all_results = [None] * len(product_list)
        uncached_indices = []
        uncached_inputs = []

        for i, prod in enumerate(product_list):
            canon = canonicalize_smiles(prod)

            self.stats['total_calls'] += 1

            if canon is None:
                all_results[i] = []
                continue

            cache_key = (canon, num_beams)
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.stats['cache_hits'] += 1
                all_results[i] = cached[:num_return]
            else:
                uncached_indices.append(i)
                uncached_inputs.append(canon)

        # Batch inference for uncached
        if uncached_inputs:
            batch_results = self._inference_batch(uncached_inputs, num_beams, num_return)

            for idx, results in zip(uncached_indices, batch_results):
                all_results[idx] = results[:num_return]

                # Cache
                canon = canonicalize_smiles(product_list[idx])
                if canon:
                    cache_key = (canon, num_beams)
                    self.cache.put(cache_key, results)

        return all_results

    def _inference_single(self, input_smiles: str, num_beams: int,
                          num_return: int) -> List[Tuple[str, float]]:
        """Run inference on a single input."""
        return self._inference_batch([input_smiles], num_beams, num_return)[0]

    def _inference_batch(self, input_smiles_list: List[str], num_beams: int,
                         num_return: int) -> List[List[Tuple[str, float]]]:
        """Run batch inference with automatic sub-batching to prevent OOM."""
        max_sub = self.config.max_sub_batch
        if len(input_smiles_list) <= max_sub:
            return self._inference_batch_core(input_smiles_list, num_beams, num_return)

        all_results = []
        for i in range(0, len(input_smiles_list), max_sub):
            sub = input_smiles_list[i:i + max_sub]
            sub_results = self._inference_batch_core(sub, num_beams, num_return)
            all_results.extend(sub_results)
        return all_results

    def _inference_batch_core(self, input_smiles_list: List[str], num_beams: int,
                              num_return: int) -> List[List[Tuple[str, float]]]:
        """
        Run batch inference on a single sub-batch.

        Input: raw product SMILES (no prefix for retrosynthesis model)
        Output: dot-separated reactant SMILES
        """
        self.stats['inference_calls'] += 1
        start = time.time()

        # Tokenize - retro model takes raw SMILES, no prefix
        inputs = self.tokenizer(
            input_smiles_list,
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

        # Group by input
        batch_size = len(input_smiles_list)
        all_results = []

        for i in range(batch_size):
            results = []
            for j in range(num_return):
                seq_idx = i * num_return + j
                if seq_idx >= len(decoded):
                    break

                reactants_smiles = decoded[seq_idx]
                score = scores[seq_idx] if seq_idx < len(scores) else 0.0

                # Validate: each reactant component should parse
                parts = reactants_smiles.split('.')
                valid_parts = []
                for p in parts:
                    c = canonicalize_smiles(p)
                    if c is not None:
                        valid_parts.append(c)

                if valid_parts:
                    canon_reactants = '.'.join(valid_parts)
                    results.append((canon_reactants, float(score)))

            all_results.append(results)

        return all_results

    def round_trip_validate(self, product_smiles: str,
                            forward_predictor=None,
                            num_beams: int = None) -> List[Dict]:
        """
        Round-trip validation: product -> retro -> reactants -> forward -> compare.

        Args:
            product_smiles: Product SMILES to validate
            forward_predictor: ProductPredictor instance (optional, created if None)
            num_beams: Beam width for retro prediction

        Returns:
            List of dicts with keys:
                - retro_reactants: predicted reactants
                - retro_score: retro prediction score
                - forward_product: re-predicted product (if forward_predictor given)
                - forward_score: forward prediction score
                - tanimoto: Tanimoto similarity between original and re-predicted product
                - exact_match: whether products match exactly
        """
        # Get retro predictions
        retro_results = self.predict(product_smiles, num_beams=num_beams)
        if not retro_results:
            return []

        canon_product = canonicalize_smiles(product_smiles)
        if canon_product is None:
            return []

        # Lazy-load forward predictor if needed
        if forward_predictor is None:
            from model_reactions.product_prediction.product_predictor import ProductPredictor
            forward_predictor = ProductPredictor(device=str(self.device))

        validations = []
        for reactants_smi, retro_score in retro_results:
            entry = {
                'retro_reactants': reactants_smi,
                'retro_score': retro_score,
                'forward_product': None,
                'forward_score': None,
                'tanimoto': 0.0,
                'exact_match': False,
            }

            # Split reactants and run forward prediction
            parts = reactants_smi.split('.')
            if len(parts) >= 2:
                # Use first two components as mol + co-reactant
                # Try largest + second largest by atom count
                mols_with_size = []
                for p in parts:
                    m = Chem.MolFromSmiles(p)
                    if m is not None:
                        mols_with_size.append((p, m.GetNumHeavyAtoms()))
                mols_with_size.sort(key=lambda x: -x[1])

                if len(mols_with_size) >= 2:
                    mol_smi = mols_with_size[0][0]
                    co_smi = mols_with_size[1][0]

                    fwd_results = forward_predictor.predict(mol_smi, co_smi, num_beams=5, num_return=1)
                    if fwd_results:
                        fwd_prod, fwd_score = fwd_results[0]
                        entry['forward_product'] = fwd_prod
                        entry['forward_score'] = fwd_score
                        entry['tanimoto'] = tanimoto_similarity(canon_product, fwd_prod)
                        entry['exact_match'] = (canonicalize_smiles(fwd_prod) == canon_product)
            elif len(parts) == 1:
                # Single reactant - no co-reactant, skip forward
                entry['forward_product'] = None

            validations.append(entry)

        return validations

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
    parser = argparse.ArgumentParser(description="RetroPredictor: predict reactants from products")
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--round_trip', action='store_true', help='Run round-trip validation')
    args = parser.parse_args()

    predictor = RetroPredictor(device=args.device)

    # Test molecules: known drugs and simple products
    test_products = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
        ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("Ethyl acetate", "CCOC(C)=O"),
        ("Lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ]

    print("=== Retrosynthesis Predictions ===")
    for name, smi in test_products:
        results = predictor.predict(smi, num_beams=args.num_beams, num_return=3)
        print(f"\n{name}: {smi}")
        for reactants, score in results:
            print(f"  <- {reactants}  (score={score:.4f})")

    # Batch prediction
    print("\n=== Batch Prediction ===")
    products = [p[1] for p in test_products]
    batch_results = predictor.predict_batch(products, num_beams=args.num_beams, num_return=2)
    for (name, smi), results in zip(test_products, batch_results):
        print(f"\n{name}: {smi}")
        for reactants, score in results:
            print(f"  <- {reactants}  (score={score:.4f})")

    # Round-trip validation
    if args.round_trip:
        print("\n=== Round-Trip Validation ===")
        for name, smi in test_products:
            print(f"\n{name}: {smi}")
            validations = predictor.round_trip_validate(smi)
            for v in validations:
                print(f"  Retro: {v['retro_reactants']}  (score={v['retro_score']:.4f})")
                if v['forward_product']:
                    print(f"  Forward: {v['forward_product']}  (score={v['forward_score']:.4f})")
                    print(f"  Tanimoto: {v['tanimoto']:.4f}  Exact: {v['exact_match']}")
                else:
                    print(f"  Forward: N/A (single reactant or failed)")

    # Cache test
    print("\n=== Cache Test ===")
    predictor.predict(test_products[0][1])  # Should be cached
    print(f"Stats: {predictor.get_stats()}")
