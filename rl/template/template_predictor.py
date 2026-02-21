"""Template-based reaction predictor for DA-MolDQN.

Replaces the learned T5v2 product predictor with deterministic RDKit
reaction templates (SMARTS). Templates guarantee 100% chemical validity
and provide the reaction type for each step.

Architecture:
  Current molecule -> Match templates -> For each match:
    - Uni-molecular: RunReactants(mol) -> product directly
    - Bi-molecular: Find compatible building blocks -> RunReactants(mol, block) -> products
  -> Deduplicate -> Return top_k (co_reactants, products, scores)

The get_valid_actions() API matches ReactionPredictor / AIOReactionPredictor.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem

from .reaction import UniReaction, BiReaction, load_templates
from .building_blocks import BuildingBlockLibrary

# Default paths relative to this file
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_TEMPLATE_PATH = _THIS_DIR / "data" / "templates_v2.txt"
_DEFAULT_BB_PATH = _THIS_DIR / "data" / "building_blocks.smi.gz"


class TemplateReactionPredictor:
    """Template-based reaction predictor.

    Uses RDKit reaction SMARTS for deterministic, chemically-valid product
    generation. No GPU required.

    Args:
        template_path: Path to reaction templates file (one SMARTS per line).
        building_block_path: Path to building blocks SMILES file.
        top_k: Default number of actions to return.
        max_blocks_per_template: Max building blocks to sample per template match.
        seed: Random seed for block sampling reproducibility.
        num_workers: Number of threads for parallel batch inference.
                     0 = serial. >0 = ThreadPoolExecutor (RDKit releases GIL).
    """

    def __init__(
        self,
        template_path: str | Path | None = None,
        building_block_path: str | Path | None = None,
        top_k: int = 20,
        max_blocks_per_template: int = 100,
        seed: int = 42,
        num_workers: int = 0,
        **kwargs,  # accept extra kwargs for compatibility
    ):
        self.template_path = Path(template_path) if template_path else _DEFAULT_TEMPLATE_PATH
        self.building_block_path = Path(building_block_path) if building_block_path else _DEFAULT_BB_PATH
        self.top_k = top_k
        self.max_blocks_per_template = max_blocks_per_template
        self.rng = random.Random(seed)
        self.num_workers = num_workers

        # Populated by load()
        self.uni_reactions: list[UniReaction] = []
        self.bi_reactions: list[BiReaction] = []
        self.bb_library: BuildingBlockLibrary | None = None
        # Sparse compatibility index: bi_reaction index -> list of block indices
        self.bi_compat: dict[int, np.ndarray] = {}
        self._loaded = False
        self._executor: ThreadPoolExecutor | None = None

    def load(self) -> None:
        """Load templates, building blocks, and build compatibility mask."""
        if self._loaded:
            return

        t0 = time.perf_counter()

        # 1. Load templates
        self.uni_reactions, self.bi_reactions = load_templates(str(self.template_path))

        # 2. Load building blocks
        self.bb_library = BuildingBlockLibrary(self.building_block_path)
        self.bb_library.load()

        t1 = time.perf_counter()
        print(f"  Templates: {len(self.uni_reactions)} uni + {len(self.bi_reactions)} bi "
              f"(from {len(self.uni_reactions) + len(set(r.index for r in self.bi_reactions))} raw)")
        print(f"  Building blocks: {len(self.bb_library)} loaded in {t1 - t0:.1f}s")

        # 3. Build compatibility mask (sparse: list of compatible block indices per bi-reaction)
        self._build_compatibility_index()

        t2 = time.perf_counter()
        total_compat = sum(len(v) for v in self.bi_compat.values())
        print(f"  Compatibility index: {total_compat} pairs built in {t2 - t1:.1f}s")

        # 4. Start thread pool if requested
        if self.num_workers > 0:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
            print(f"  Thread pool: {self.num_workers} workers")

        self._loaded = True
        print(f"  TemplateReactionPredictor ready ({t2 - t0:.1f}s total)")

    def _build_compatibility_index(self) -> None:
        """Precompute which building blocks are compatible with each bi-reaction."""
        block_mols = self.bb_library.mol_list
        n_blocks = len(block_mols)

        for bi_idx, bi_rxn in enumerate(self.bi_reactions):
            compat_indices = []
            pattern = bi_rxn.reactant_pattern[bi_rxn.block_order]
            for blk_idx in range(n_blocks):
                if block_mols[blk_idx].HasSubstructMatch(pattern):
                    compat_indices.append(blk_idx)
            self.bi_compat[bi_idx] = np.array(compat_indices, dtype=np.int32)

    def get_valid_actions(
        self, mol, top_k: int | None = None, _rng: random.Random | None = None
    ) -> tuple[list[str], list[str], np.ndarray]:
        """Generate valid reaction products for a molecule.

        Args:
            mol: RDKit Mol object or SMILES string.
            top_k: Max number of actions to return (default: self.top_k).
            _rng: Optional per-call RNG (for thread safety in batch mode).

        Returns:
            (co_reactants, products, scores) where:
            - co_reactants: list of co-reactant SMILES ("" for uni-molecular)
            - products: list of product SMILES
            - scores: np.float32 array of uniform scores (1.0)
        """
        if top_k is None:
            top_k = self.top_k
        rng = _rng or self.rng

        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return [], [], np.array([], dtype=np.float32)

        mol_smi = Chem.MolToSmiles(mol)

        # Collect (co_reactant_smi, product_smi) pairs
        seen_products = set()
        actions = []  # list of (co_reactant, product) tuples

        # --- Uni-molecular reactions ---
        for uni_rxn in self.uni_reactions:
            if not uni_rxn.is_reactant(mol):
                continue
            product_smiles_list = uni_rxn.forward_smiles(mol)
            for prod_smi in product_smiles_list:
                if prod_smi and prod_smi != mol_smi and prod_smi not in seen_products:
                    seen_products.add(prod_smi)
                    actions.append(("", prod_smi))

        # --- Bi-molecular reactions ---
        for bi_idx, bi_rxn in enumerate(self.bi_reactions):
            if not bi_rxn.is_mol_reactant(mol):
                continue

            compat = self.bi_compat.get(bi_idx)
            if compat is None or len(compat) == 0:
                continue

            # Sample blocks if too many
            if len(compat) > self.max_blocks_per_template:
                sampled_indices = rng.sample(range(len(compat)), self.max_blocks_per_template)
                block_indices = compat[sampled_indices]
            else:
                block_indices = compat

            for blk_idx in block_indices:
                blk_smi, blk_mol = self.bb_library[blk_idx]
                product_smiles_list = bi_rxn.forward_smiles(mol, blk_mol)
                for prod_smi in product_smiles_list:
                    if prod_smi and prod_smi != mol_smi and prod_smi not in seen_products:
                        seen_products.add(prod_smi)
                        actions.append((blk_smi, prod_smi))

            # Early exit if we already have plenty of candidates
            if len(actions) >= top_k * 3:
                break

        # Shuffle and truncate to top_k
        if len(actions) > top_k:
            rng.shuffle(actions)
            actions = actions[:top_k]

        if not actions:
            return [], [], np.array([], dtype=np.float32)

        co_reactants = [a[0] for a in actions]
        products = [a[1] for a in actions]
        scores = np.ones(len(actions), dtype=np.float32)

        return co_reactants, products, scores

    def get_valid_actions_batch(
        self, mols, top_k: int | None = None
    ) -> list[tuple[list[str], list[str], np.ndarray]]:
        """Batch version of get_valid_actions with dedup + thread parallelism.

        Deduplicates SMILES first so identical molecules are computed once.
        Uses ThreadPoolExecutor for parallel RDKit ops (GIL released in C++).
        """
        if top_k is None:
            top_k = self.top_k

        # Convert all inputs to canonical SMILES
        smiles_list = []
        for m in mols:
            if isinstance(m, str):
                parsed = Chem.MolFromSmiles(m)
                smiles_list.append(Chem.MolToSmiles(parsed) if parsed else m)
            else:
                smiles_list.append(Chem.MolToSmiles(m))

        # Dedup: only compute unique SMILES
        unique_smiles = list(dict.fromkeys(smiles_list))

        if self._executor is not None and len(unique_smiles) > 1:
            # Parallel: dispatch unique SMILES to thread pool
            # Each thread gets its own RNG for thread safety
            base_seed = self.rng.randint(0, 2**31)

            def _call(args):
                smi, idx = args
                thread_rng = random.Random(base_seed + idx)
                return self.get_valid_actions(smi, top_k, _rng=thread_rng)

            futures = list(self._executor.map(
                _call, [(smi, i) for i, smi in enumerate(unique_smiles)]))
            cache = dict(zip(unique_smiles, futures))
        else:
            # Serial fallback
            cache = {}
            for smi in unique_smiles:
                cache[smi] = self.get_valid_actions(smi, top_k)

        return [cache[smi] for smi in smiles_list]

    def close(self):
        """Shut down thread pool if active."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


def _test():
    """Quick smoke test on common molecules."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--smiles', type=str, nargs='+',
                        default=['CCO', 'c1ccccc1O', 'CC(=O)O', 'c1ccc(N)cc1',
                                 'CC(=O)Nc1ccccc1', 'O=C(O)c1ccccc1'])
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--max_blocks', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    predictor = TemplateReactionPredictor(
        top_k=args.top_k,
        max_blocks_per_template=args.max_blocks,
        num_workers=args.num_workers,
    )
    predictor.load()

    print(f"\n{'='*60}")
    print(f"Testing {len(args.smiles)} molecules (top_k={args.top_k})")
    print(f"{'='*60}")

    for smi in args.smiles:
        t0 = time.perf_counter()
        co_reacts, products, scores = predictor.get_valid_actions(smi, top_k=args.top_k)
        elapsed = time.perf_counter() - t0

        print(f"\n[{smi}] → {len(products)} actions in {elapsed*1000:.1f}ms")
        for i, (co, prod, score) in enumerate(zip(co_reacts, products, scores)):
            co_str = co if co else "(uni)"
            print(f"  {i+1:3d}. {co_str:40s} → {prod}")
            if i >= 9:
                print(f"  ... ({len(products) - 10} more)")
                break

    # Batch timing test
    print(f"\n{'='*60}")
    print("Batch timing test: 64 molecules")
    mols_64 = [args.smiles[i % len(args.smiles)] for i in range(64)]
    t0 = time.perf_counter()
    results = predictor.get_valid_actions_batch(mols_64, top_k=args.top_k)
    elapsed = time.perf_counter() - t0
    total_actions = sum(len(r[1]) for r in results)
    print(f"  64 mols: {elapsed*1000:.0f}ms, {total_actions} total actions")

    predictor.close()


if __name__ == "__main__":
    _test()
