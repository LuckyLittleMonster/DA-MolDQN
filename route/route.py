"""Core data structures for synthesis routes.

A SynthesisRoute is an ordered list of RouteSteps, where each step is a
(template, building_block) pair. Forward execution from init_mol through
the steps yields intermediate products and a final product.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Mol as RDMol

# Module-level tp for multiprocessing fork inheritance.
# Set before creating Pool, workers inherit it via copy-on-write.
_mp_tp = None


@dataclass
class RouteStep:
    """One step in a synthesis route.

    Attributes:
        template_idx: Index into the original templates file (Reaction.index).
        bi_rxn_idx: Index into TemplateReactionPredictor.bi_reactions[].
            -1 for uni-molecular reactions.
        block_idx: Index into BuildingBlockLibrary. -1 for uni-molecular.
        block_smi: Building block SMILES (empty for uni-molecular).
        intermediate_smi: Product SMILES after this step.
        is_uni: True if this is a uni-molecular reaction (no BB to swap).
        uni_rxn_idx: Index into TemplateReactionPredictor.uni_reactions[].
            -1 for bi-molecular reactions.
    """
    template_idx: int
    bi_rxn_idx: int = -1
    block_idx: int = -1
    block_smi: str = ""
    intermediate_smi: str = ""
    is_uni: bool = False
    uni_rxn_idx: int = -1


@dataclass
class SynthesisRoute:
    """Complete synthesis route from building blocks to final product.

    Attributes:
        steps: Ordered list of RouteSteps (execution order).
        init_mol_smi: Starting molecule SMILES (input to step 0).
        final_product_smi: Final product SMILES (output of last step).
    """
    steps: list[RouteStep] = field(default_factory=list)
    init_mol_smi: str = ""
    final_product_smi: str = ""

    @property
    def modifiable_mask(self) -> list[bool]:
        """Which positions have swappable building blocks (L1 mask).

        Uni-molecular reaction positions cannot be modified.
        """
        return [not step.is_uni for step in self.steps]

    @property
    def n_modifiable(self) -> int:
        """Number of positions with swappable building blocks."""
        return sum(self.modifiable_mask)

    def __len__(self) -> int:
        return len(self.steps)

    def copy(self) -> SynthesisRoute:
        """Deep copy of this route."""
        return deepcopy(self)

    def forward_execute(self, from_step: int, tp) -> bool:
        """Re-execute route from `from_step` to end, updating intermediates.

        Args:
            from_step: Index of the first step to re-execute.
            tp: TemplateReactionPredictor instance providing reaction objects.

        Returns:
            True if all steps succeed, False if any step fails.
        """
        if from_step == 0:
            current_mol = Chem.MolFromSmiles(self.init_mol_smi)
        else:
            current_mol = Chem.MolFromSmiles(
                self.steps[from_step - 1].intermediate_smi)

        if current_mol is None:
            return False

        for j in range(from_step, len(self.steps)):
            step = self.steps[j]
            products = _execute_step(step, current_mol, tp)
            if not products:
                return False
            step.intermediate_smi = products[0]
            current_mol = Chem.MolFromSmiles(products[0])
            if current_mol is None:
                return False

        self.final_product_smi = self.steps[-1].intermediate_smi
        return True

    def validate(self, tp) -> bool:
        """Validate entire route by forward execution from step 0."""
        return self.forward_execute(0, tp)


def _validate_one_candidate(
    blk_idx: int,
    route: SynthesisRoute,
    position: int,
    input_mol: RDMol,
    tp,
) -> int | None:
    """Validate a single candidate BB (for parallel execution).

    Returns blk_idx if valid, None otherwise.
    """
    step = route.steps[position]
    bi_rxn = tp.bi_reactions[step.bi_rxn_idx]

    _blk_smi, blk_mol = tp.bb_library[int(blk_idx)]
    products = bi_rxn.forward_smiles(input_mol, blk_mol)
    if not products:
        return None

    current_mol = Chem.MolFromSmiles(products[0])
    if current_mol is None:
        return None

    for j in range(position + 1, len(route.steps)):
        next_step = route.steps[j]
        next_products = _execute_step(next_step, current_mol, tp)
        if not next_products:
            return None
        current_mol = Chem.MolFromSmiles(next_products[0])
        if current_mol is None:
            return None

    return int(blk_idx)


def cascade_validate(
    route: SynthesisRoute,
    position: int,
    candidate_block_indices: list[int],
    tp,
    executor: ThreadPoolExecutor | None = None,
) -> list[int]:
    """Check which candidate BBs produce valid downstream routes (L3 mask).

    For each candidate BB at `position`, execute from that position to the
    end of the route. Return indices of candidates that succeed.

    Args:
        route: Current synthesis route.
        position: Step index where BB is being replaced.
        candidate_block_indices: BB indices to test (from L2 mask).
        tp: TemplateReactionPredictor instance.
        executor: Optional ThreadPoolExecutor for parallel validation.

    Returns:
        List of block indices that pass cascade validation.
    """
    step = route.steps[position]

    # Get input molecule for this position
    if position == 0:
        input_mol = Chem.MolFromSmiles(route.init_mol_smi)
    else:
        input_mol = Chem.MolFromSmiles(
            route.steps[position - 1].intermediate_smi)

    if input_mol is None:
        return []

    # Parallel validation if executor provided and enough candidates
    if executor is not None and len(candidate_block_indices) > 16:
        futures = [
            executor.submit(
                _validate_one_candidate, blk_idx, route, position,
                input_mol, tp)
            for blk_idx in candidate_block_indices
        ]
        valid_blocks = []
        for f in futures:
            result = f.result()
            if result is not None:
                valid_blocks.append(result)
        return valid_blocks

    # Sequential fallback
    valid_blocks = []
    bi_rxn = tp.bi_reactions[step.bi_rxn_idx]

    for blk_idx in candidate_block_indices:
        _blk_smi, blk_mol = tp.bb_library[int(blk_idx)]

        products = bi_rxn.forward_smiles(input_mol, blk_mol)
        if not products:
            continue

        current_mol = Chem.MolFromSmiles(products[0])
        if current_mol is None:
            continue

        cascade_ok = True
        for j in range(position + 1, len(route.steps)):
            next_step = route.steps[j]
            next_products = _execute_step(next_step, current_mol, tp)
            if not next_products:
                cascade_ok = False
                break
            current_mol = Chem.MolFromSmiles(next_products[0])
            if current_mol is None:
                cascade_ok = False
                break

        if cascade_ok:
            valid_blocks.append(int(blk_idx))

    return valid_blocks


def update_route(
    route: SynthesisRoute,
    position: int,
    new_block_idx: int,
    tp,
) -> str | None:
    """Replace BB at position and re-execute from there to end.

    Args:
        route: Route to update (modified in-place).
        position: Step index to modify.
        new_block_idx: New building block index.
        tp: TemplateReactionPredictor instance.

    Returns:
        New final product SMILES, or None if execution fails.
    """
    step = route.steps[position]
    new_blk_smi, _ = tp.bb_library[new_block_idx]

    step.block_idx = new_block_idx
    step.block_smi = new_blk_smi

    if route.forward_execute(position, tp):
        return route.final_product_smi
    return None


def _execute_step(
    step: RouteStep,
    current_mol: RDMol,
    tp,
) -> list[str]:
    """Execute a single route step, returning product SMILES list.

    Args:
        step: The RouteStep to execute.
        current_mol: Input molecule (RDKit Mol).
        tp: TemplateReactionPredictor providing reaction objects.

    Returns:
        List of product SMILES (usually 1), or empty list on failure.
    """
    if step.is_uni:
        # Use dict lookup if available, else linear scan
        uni_idx_map = getattr(tp, '_uni_by_template_idx', None)
        if uni_idx_map is None:
            # Build and cache the index on tp (one-time cost)
            uni_idx_map = {r.index: r for r in tp.uni_reactions}
            tp._uni_by_template_idx = uni_idx_map
        uni_rxn = uni_idx_map.get(step.template_idx)
        if uni_rxn is not None and uni_rxn.is_reactant(current_mol):
            return uni_rxn.forward_smiles(current_mol)
        return []
    else:
        bi_rxn = tp.bi_reactions[step.bi_rxn_idx]
        if not bi_rxn.is_mol_reactant(current_mol):
            return []
        _, blk_mol = tp.bb_library[step.block_idx]
        return bi_rxn.forward_smiles(current_mol, blk_mol)


# ------------------------------------------------------------------
# Fast cascade validation (Mol-direct + last-step shortcut)
# ------------------------------------------------------------------

def _fast_forward_bi(bi_rxn, mol: RDMol, block: RDMol) -> RDMol | None:
    """Fast bi-molecular reaction: returns first valid product Mol.

    Skips full canonicalization (RemoveHs + MolToSmiles + MolFromSmiles)
    and deduplication. Uses SanitizeMol only. For cascade validation.
    """
    if bi_rxn.is_mol_first:
        reactants = (mol, block)
    else:
        reactants = (block, mol)
    ps = bi_rxn._rxn_forward.RunReactants(reactants, 1)
    for p in ps:
        if len(p) == 1:
            try:
                Chem.SanitizeMol(p[0])
                return p[0]
            except Exception:
                continue
    return None


def _fast_forward_uni(uni_rxn, mol: RDMol) -> RDMol | None:
    """Fast uni-molecular reaction: returns first valid product Mol."""
    ps = uni_rxn._rxn_forward.RunReactants((mol,), 1)
    for p in ps:
        if len(p) == 1:
            try:
                Chem.SanitizeMol(p[0])
                return p[0]
            except Exception:
                continue
    return None


def _get_uni_map(tp):
    """Get/build the template_idx → UniReaction dict on tp."""
    uni_idx_map = getattr(tp, '_uni_by_template_idx', None)
    if uni_idx_map is None:
        uni_idx_map = {r.index: r for r in tp.uni_reactions}
        tp._uni_by_template_idx = uni_idx_map
    return uni_idx_map


def _validate_one_candidate_fast(
    blk_idx: int,
    route: SynthesisRoute,
    position: int,
    input_mol: RDMol,
    tp,
) -> int | None:
    """Optimized candidate validation: Mol-direct + last-step shortcut.

    Compared to _validate_one_candidate:
    - Uses _fast_forward_bi/_fast_forward_uni (SanitizeMol only, no SMILES)
    - Skips full reaction on last downstream step (is_reactant check only)

    Returns blk_idx if valid, None otherwise.
    """
    step = route.steps[position]
    bi_rxn = tp.bi_reactions[step.bi_rxn_idx]
    _, blk_mol = tp.bb_library[int(blk_idx)]

    # Execute reaction at modified position
    current_mol = _fast_forward_bi(bi_rxn, input_mol, blk_mol)
    if current_mol is None:
        return None

    n_steps = len(route.steps)
    uni_idx_map = _get_uni_map(tp)

    for j in range(position + 1, n_steps):
        next_step = route.steps[j]
        is_last = (j == n_steps - 1)

        if next_step.is_uni:
            uni_rxn = uni_idx_map.get(next_step.template_idx)
            if uni_rxn is None or not uni_rxn.is_reactant(current_mol):
                return None
            if not is_last:
                current_mol = _fast_forward_uni(uni_rxn, current_mol)
                if current_mol is None:
                    return None
        else:
            next_bi_rxn = tp.bi_reactions[next_step.bi_rxn_idx]
            if not next_bi_rxn.is_mol_reactant(current_mol):
                return None
            if not is_last:
                _, next_blk_mol = tp.bb_library[next_step.block_idx]
                current_mol = _fast_forward_bi(
                    next_bi_rxn, current_mol, next_blk_mol)
                if current_mol is None:
                    return None

    return int(blk_idx)


# ------------------------------------------------------------------
# Multiprocessing worker
# ------------------------------------------------------------------

def _mp_validate_route(args):
    """Multiprocessing worker: validate all L2 candidates for one route.

    Uses module-level _mp_tp (fork-inherited from parent).

    Args: tuple of (route, position, l2_candidates, input_mol_smi)
    Returns: list[int] of valid block indices.
    """
    route, position, l2_candidates, input_mol_smi = args
    tp = _mp_tp
    input_mol = Chem.MolFromSmiles(input_mol_smi)
    if input_mol is None:
        return []

    valid = []
    for blk_idx in l2_candidates:
        if _validate_one_candidate_fast(
                blk_idx, route, position, input_mol, tp) is not None:
            valid.append(blk_idx)
    return valid
