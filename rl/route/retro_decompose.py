"""Template-based retrosynthetic decomposition and route conversion.

Route generation strategies:
1. build_random_routes(): Forward random construction (fast, for bootstrapping).
2. decompose_molecule(): Forward DFS exploration.
3. build_retro_routes(): True retrosynthesis using reversed SMARTS templates.
4. build_routes_from_aizynth(): Convert AiZynthFinder routes to our template+BB format.

The retro strategy reverses the 116 SMARTS templates (A.B>>C -> C>>A.B),
applies them to a target molecule to find precursors, checks which precursors
are in the 10K ZINC BB library, and recursively decomposes the main chain
until a complete synthesis route is found.
"""

from __future__ import annotations

import random as _random
import time
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from .route import RouteStep, SynthesisRoute


# ─────────────────── Retro: Reversed-Template Decomposition ───────────────────


def build_retro_routes(
    target_smi: str,
    tp,
    max_depth: int = 5,
    max_routes: int = 3,
    min_steps: int = 2,
    seed: int = 42,
    verify: bool = True,
) -> list[SynthesisRoute]:
    """Build synthesis routes by retrosynthetic decomposition using reversed templates.

    Applies reversed SMARTS templates to the target molecule to find precursors.
    For bi-molecular templates, one precursor must be a building block from the
    10K ZINC library. Recursively decomposes the main-chain precursor.

    Args:
        target_smi: Target molecule SMILES to decompose.
        tp: Loaded TemplateReactionPredictor.
        max_depth: Maximum number of retro steps.
        max_routes: Maximum routes to return.
        min_steps: Minimum steps for a valid route.
        seed: Random seed.
        verify: If True, verify routes by forward execution.

    Returns:
        List of SynthesisRoute objects (forward direction: init_mol -> target).
    """
    target_mol = Chem.MolFromSmiles(target_smi)
    if target_mol is None:
        return []
    target_canon = Chem.MolToSmiles(target_mol)

    retro_data = _get_retro_data(tp)
    routes: list[SynthesisRoute] = []
    rng = _random.Random(seed)

    _retro_dfs(
        target_mol=target_mol,
        target_smi=target_canon,
        retro_steps=[],
        retro_data=retro_data,
        tp=tp,
        max_depth=max_depth,
        max_routes=max_routes,
        min_steps=min_steps,
        routes=routes,
        visited={target_canon},
        rng=rng,
        verify=verify,
    )

    return routes[:max_routes]


def _reverse_smarts(smarts: str) -> str:
    """Swap reactants and products in a SMARTS string.

    A.B>>C becomes C>>A.B (for retrosynthetic application).
    """
    parts = smarts.split('>')
    if len(parts) == 3:
        return f"{parts[2]}>{parts[1]}>{parts[0]}"
    elif len(parts) == 2:
        return f"{parts[1]}>>{parts[0]}"
    raise ValueError(f"Cannot parse SMARTS: {smarts}")


def _get_retro_data(tp) -> dict:
    """Build and cache reversed templates + BB lookup on tp.

    Returns dict with keys:
        retro_uni: list of (rev_rxn, template_idx, uni_rxn_list_idx)
        retro_bi: list of (rev_rxn, template_idx, {is_mol_first: bi_rxn_list_idx})
        bb_lookup: dict of canonical_SMILES -> bb_library_index
        bi_compat_sets: dict of bi_rxn_list_idx -> set of compatible bb indices
    """
    if hasattr(tp, '_retro_data'):
        return tp._retro_data

    # 1. Reverse uni templates
    retro_uni = []
    for uni_list_idx, uni_rxn in enumerate(tp.uni_reactions):
        try:
            rev_smarts = _reverse_smarts(uni_rxn.template)
            rev_rxn = ReactionFromSmarts(rev_smarts)
            rev_rxn.Initialize()
            retro_uni.append((rev_rxn, uni_rxn.index, uni_list_idx))
        except Exception:
            pass

    # 2. Reverse bi templates (each SMARTS appears as 2 BiReaction objects)
    retro_bi = []
    seen_templates: dict[str, int] = {}  # template_str -> index in retro_bi

    for bi_list_idx, bi_rxn in enumerate(tp.bi_reactions):
        if bi_rxn.template in seen_templates:
            entry_idx = seen_templates[bi_rxn.template]
            retro_bi[entry_idx][2][bi_rxn.is_mol_first] = bi_list_idx
            continue

        try:
            rev_smarts = _reverse_smarts(bi_rxn.template)
            rev_rxn = ReactionFromSmarts(rev_smarts)
            rev_rxn.Initialize()
            bi_idx_map = {bi_rxn.is_mol_first: bi_list_idx}
            retro_bi.append([rev_rxn, bi_rxn.index, bi_idx_map])
            seen_templates[bi_rxn.template] = len(retro_bi) - 1
        except Exception:
            pass

    # 3. BB canonical SMILES -> library index
    bb_lookup = {}
    for i, smi in enumerate(tp.bb_library.smiles_list):
        bb_lookup[smi] = i  # already canonical from BuildingBlockLibrary.load()

    # 4. bi_compat as sets for O(1) lookup
    bi_compat_sets = {}
    for bi_idx, compat_arr in tp.bi_compat.items():
        bi_compat_sets[bi_idx] = set(int(x) for x in compat_arr)

    retro_data = {
        'retro_uni': retro_uni,
        'retro_bi': retro_bi,
        'bb_lookup': bb_lookup,
        'bi_compat_sets': bi_compat_sets,
    }
    tp._retro_data = retro_data
    return retro_data


def _canon_precursor(mol: RDMol) -> tuple[str | None, RDMol | None]:
    """Canonicalize a precursor from RunReactants."""
    try:
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)
        smi = Chem.MolToSmiles(mol)
        # Re-parse for clean mol
        clean_mol = Chem.MolFromSmiles(smi)
        if clean_mol is not None:
            return smi, clean_mol
    except Exception:
        pass
    return None, None


def _try_retro_bi_step(
    chain_smi: str,
    chain_mol: RDMol,
    frag_smi: str,
    template_idx: int,
    bi_rxn_idx: int,
    target_smi: str,
    bb_lookup: dict[str, int],
    bi_compat_sets: dict[int, set[int]],
    tp,
    rng: _random.Random,
    max_fallback_tries: int = 20,
) -> RouteStep | None:
    """Try to create a RouteStep for a retro bi-molecular decomposition.

    Two-tier strategy:
    1. Fast path: Check if the fragment precursor is an exact BB match AND
       compatible with the template.
    2. Fallback: Verify that chain_mol matches the forward template's mol pattern,
       then sample compatible BBs and try forward execution.

    Returns a RouteStep if successful, None otherwise.
    """
    compat_set = bi_compat_sets.get(bi_rxn_idx, set())

    # --- Fast path: exact BB match ---
    bb_idx = bb_lookup.get(frag_smi)
    if bb_idx is not None and bb_idx in compat_set:
        return RouteStep(
            template_idx=template_idx,
            bi_rxn_idx=bi_rxn_idx,
            block_idx=bb_idx,
            block_smi=frag_smi,
            intermediate_smi=target_smi,
            is_uni=False,
        )

    # --- Fallback: try compatible BBs from library ---
    if not compat_set:
        return None

    bi_rxn = tp.bi_reactions[bi_rxn_idx]
    if not bi_rxn.is_mol_reactant(chain_mol):
        return None

    # Sample a subset of compatible BBs and try forward execution
    compat_list = list(compat_set)
    n_try = min(max_fallback_tries, len(compat_list))
    sample_indices = rng.sample(compat_list, n_try)

    for blk_idx in sample_indices:
        blk_smi, blk_mol = tp.bb_library[blk_idx]
        products = bi_rxn.forward_smiles(chain_mol, blk_mol)
        if products and products[0]:
            return RouteStep(
                template_idx=template_idx,
                bi_rxn_idx=bi_rxn_idx,
                block_idx=blk_idx,
                block_smi=blk_smi,
                intermediate_smi=target_smi,  # will be updated by forward verify
                is_uni=False,
            )

    return None


def _retro_dfs(
    target_mol: RDMol,
    target_smi: str,
    retro_steps: list[tuple[RouteStep, str]],
    retro_data: dict,
    tp,
    max_depth: int,
    max_routes: int,
    min_steps: int,
    routes: list[SynthesisRoute],
    visited: set[str],
    rng: _random.Random,
    verify: bool,
):
    """DFS retrosynthetic search.

    retro_steps: list of (forward_RouteStep, chain_precursor_smi) in retro order.
    The forward_RouteStep.intermediate_smi = the target decomposed in that step.
    chain_precursor_smi = the main-chain precursor to decompose further.
    """
    if len(routes) >= max_routes:
        return

    depth = len(retro_steps)
    bb_lookup = retro_data['bb_lookup']
    bi_compat_sets = retro_data['bi_compat_sets']

    # Check if current target is a BB -> complete route
    n_bi = sum(1 for s, _ in retro_steps if not s.is_uni)
    if target_smi in bb_lookup and depth >= min_steps and n_bi >= 1:
        _assemble_retro_route(retro_steps, target_smi, tp, routes, verify)
        if len(routes) >= max_routes:
            return

    # Save incomplete route if enough steps (init_mol may not be a BB)
    if depth >= min_steps and n_bi >= 1 and target_smi not in bb_lookup:
        _assemble_retro_route(retro_steps, target_smi, tp, routes, verify)
        if len(routes) >= max_routes:
            return

    if depth >= max_depth:
        return

    # --- Try reversed bi templates first (produce modifiable positions) ---
    bi_items = list(retro_data['retro_bi'])
    rng.shuffle(bi_items)

    for rev_rxn, template_idx, bi_idx_map in bi_items:
        if len(routes) >= max_routes:
            return

        try:
            if not rev_rxn.IsMoleculeReactant(target_mol):
                continue
        except Exception:
            continue

        try:
            precursor_sets = rev_rxn.RunReactants((target_mol,), 5)
        except Exception:
            continue

        seen_combos: set[tuple[str, str]] = set()
        for prec_set in precursor_sets:
            if len(prec_set) != 2:
                continue

            p0_smi, p0_mol = _canon_precursor(prec_set[0])
            p1_smi, p1_mol = _canon_precursor(prec_set[1])
            if p0_smi is None or p1_smi is None:
                continue

            combo = (p0_smi, p1_smi)
            if combo in seen_combos:
                continue
            seen_combos.add(combo)

            # Try orientation 1: p1 is BB, p0 is chain (is_mol_first=True)
            if (True in bi_idx_map
                    and p0_smi not in visited and p0_smi != target_smi):
                bi_rxn_idx = bi_idx_map[True]
                step = _try_retro_bi_step(
                    chain_smi=p0_smi, chain_mol=p0_mol,
                    frag_smi=p1_smi, template_idx=template_idx,
                    bi_rxn_idx=bi_rxn_idx, target_smi=target_smi,
                    bb_lookup=bb_lookup, bi_compat_sets=bi_compat_sets,
                    tp=tp, rng=rng,
                )
                if step is not None:
                    retro_steps.append((step, p0_smi))
                    visited.add(p0_smi)
                    _retro_dfs(p0_mol, p0_smi, retro_steps, retro_data, tp,
                               max_depth, max_routes, min_steps, routes,
                               visited, rng, verify)
                    retro_steps.pop()
                    visited.discard(p0_smi)

            # Try orientation 2: p0 is BB, p1 is chain (is_mol_first=False)
            if (False in bi_idx_map
                    and p1_smi not in visited and p1_smi != target_smi):
                bi_rxn_idx = bi_idx_map[False]
                step = _try_retro_bi_step(
                    chain_smi=p1_smi, chain_mol=p1_mol,
                    frag_smi=p0_smi, template_idx=template_idx,
                    bi_rxn_idx=bi_rxn_idx, target_smi=target_smi,
                    bb_lookup=bb_lookup, bi_compat_sets=bi_compat_sets,
                    tp=tp, rng=rng,
                )
                if step is not None:
                    retro_steps.append((step, p1_smi))
                    visited.add(p1_smi)
                    _retro_dfs(p1_mol, p1_smi, retro_steps, retro_data, tp,
                               max_depth, max_routes, min_steps, routes,
                               visited, rng, verify)
                    retro_steps.pop()
                    visited.discard(p1_smi)

    # --- Try reversed uni templates ---
    uni_items = list(retro_data['retro_uni'])
    rng.shuffle(uni_items)

    for rev_rxn, template_idx, uni_list_idx in uni_items:
        if len(routes) >= max_routes:
            return

        try:
            if not rev_rxn.IsMoleculeReactant(target_mol):
                continue
        except Exception:
            continue

        try:
            precursor_sets = rev_rxn.RunReactants((target_mol,), 5)
        except Exception:
            continue

        seen_prec: set[str] = set()
        for prec_set in precursor_sets:
            if len(prec_set) != 1:
                continue

            p_smi, p_mol = _canon_precursor(prec_set[0])
            if p_smi is None or p_smi in seen_prec:
                continue
            if p_smi in visited or p_smi == target_smi:
                continue
            seen_prec.add(p_smi)

            step = RouteStep(
                template_idx=template_idx,
                bi_rxn_idx=-1,
                block_idx=-1,
                block_smi="",
                intermediate_smi=target_smi,
                is_uni=True,
                uni_rxn_idx=uni_list_idx,
            )

            retro_steps.append((step, p_smi))
            visited.add(p_smi)
            _retro_dfs(p_mol, p_smi, retro_steps, retro_data, tp,
                       max_depth, max_routes, min_steps, routes,
                       visited, rng, verify)
            retro_steps.pop()
            visited.discard(p_smi)


def _assemble_retro_route(
    retro_steps: list[tuple[RouteStep, str]],
    init_smi: str,
    tp,
    routes: list[SynthesisRoute],
    verify: bool,
):
    """Convert retro decomposition path into a forward SynthesisRoute.

    retro_steps[0] = shallowest (decomposes the original target).
    retro_steps[-1] = deepest (closest to init_mol).
    init_smi = the deepest chain precursor (= route init_mol).

    Forward order is reversed retro order.
    """
    if not retro_steps:
        return

    # Build forward steps (reverse of retro order)
    forward_steps = [deepcopy(s[0]) for s in reversed(retro_steps)]
    final_smi = retro_steps[0][0].intermediate_smi

    route = SynthesisRoute(
        steps=forward_steps,
        init_mol_smi=init_smi,
        final_product_smi=final_smi,
    )

    if verify:
        # Forward execution verifies the route and updates intermediate_smi
        if not route.validate(tp):
            return

    routes.append(route)


# ─────────────────── Forward: Random Route Building ───────────────────


def decompose_molecule(
    mol_smi: str,
    tp,
    max_depth: int = 5,
    max_routes: int = 3,
    max_candidates_per_step: int = 50,
) -> list[SynthesisRoute]:
    """Decompose a molecule into synthesis routes using forward template matching.

    Builds routes FORWARD from the molecule by applying reaction templates.
    Each applied template = one route step.

    Args:
        mol_smi: Target molecule SMILES.
        tp: Loaded TemplateReactionPredictor.
        max_depth: Maximum number of synthesis steps.
        max_routes: Maximum number of routes to return.
        max_candidates_per_step: Max BB candidates to try per template match.

    Returns:
        List of SynthesisRoute objects (may be empty if no decomposition found).
    """
    mol = Chem.MolFromSmiles(mol_smi)
    if mol is None:
        return []

    canonical_smi = Chem.MolToSmiles(mol)
    routes = []

    _build_routes_dfs(
        current_mol=mol,
        current_smi=canonical_smi,
        init_mol_smi=canonical_smi,
        steps_so_far=[],
        tp=tp,
        max_depth=max_depth,
        max_routes=max_routes,
        max_candidates=max_candidates_per_step,
        routes=routes,
        visited_smiles={canonical_smi},
    )

    return routes[:max_routes]


def _build_routes_dfs(
    current_mol: RDMol,
    current_smi: str,
    init_mol_smi: str,
    steps_so_far: list[RouteStep],
    tp,
    max_depth: int,
    max_routes: int,
    max_candidates: int,
    routes: list[SynthesisRoute],
    visited_smiles: set[str],
):
    """DFS to build synthesis routes from init_mol."""
    if len(routes) >= max_routes:
        return

    depth = len(steps_so_far)

    if depth >= 2:
        route = SynthesisRoute(
            steps=list(steps_so_far),
            init_mol_smi=init_mol_smi,
            final_product_smi=current_smi,
        )
        routes.append(route)
        if len(routes) >= max_routes:
            return

    if depth >= max_depth:
        return

    # Try uni-molecular reactions
    for idx, uni_rxn in enumerate(tp.uni_reactions):
        if len(routes) >= max_routes:
            return
        if not uni_rxn.is_reactant(current_mol):
            continue

        products = uni_rxn.forward_smiles(current_mol)
        for prod_smi in products:
            if not prod_smi or prod_smi == current_smi:
                continue
            if prod_smi in visited_smiles:
                continue

            prod_mol = Chem.MolFromSmiles(prod_smi)
            if prod_mol is None:
                continue

            step = RouteStep(
                template_idx=uni_rxn.index,
                bi_rxn_idx=-1,
                block_idx=-1,
                block_smi="",
                intermediate_smi=prod_smi,
                is_uni=True,
                uni_rxn_idx=idx,
            )

            steps_so_far.append(step)
            visited_smiles.add(prod_smi)

            _build_routes_dfs(
                current_mol=prod_mol,
                current_smi=prod_smi,
                init_mol_smi=init_mol_smi,
                steps_so_far=steps_so_far,
                tp=tp,
                max_depth=max_depth,
                max_routes=max_routes,
                max_candidates=max_candidates,
                routes=routes,
                visited_smiles=visited_smiles,
            )

            steps_so_far.pop()
            visited_smiles.discard(prod_smi)

    # Try bi-molecular reactions
    for bi_idx, bi_rxn in enumerate(tp.bi_reactions):
        if len(routes) >= max_routes:
            return
        if not bi_rxn.is_mol_reactant(current_mol):
            continue

        compat = tp.bi_compat.get(bi_idx)
        if compat is None or len(compat) == 0:
            continue

        if len(compat) > max_candidates:
            import numpy as np
            sampled = np.random.choice(compat, max_candidates, replace=False)
        else:
            sampled = compat

        for blk_idx in sampled:
            if len(routes) >= max_routes:
                return

            blk_smi, blk_mol = tp.bb_library[int(blk_idx)]
            products = bi_rxn.forward_smiles(current_mol, blk_mol)

            for prod_smi in products:
                if not prod_smi or prod_smi == current_smi:
                    continue
                if prod_smi in visited_smiles:
                    continue

                prod_mol = Chem.MolFromSmiles(prod_smi)
                if prod_mol is None:
                    continue

                step = RouteStep(
                    template_idx=bi_rxn.index,
                    bi_rxn_idx=bi_idx,
                    block_idx=int(blk_idx),
                    block_smi=blk_smi,
                    intermediate_smi=prod_smi,
                    is_uni=False,
                )

                steps_so_far.append(step)
                visited_smiles.add(prod_smi)

                _build_routes_dfs(
                    current_mol=prod_mol,
                    current_smi=prod_smi,
                    init_mol_smi=init_mol_smi,
                    steps_so_far=steps_so_far,
                    tp=tp,
                    max_depth=max_depth,
                    max_routes=max_routes,
                    max_candidates=max_candidates,
                    routes=routes,
                    visited_smiles=visited_smiles,
                )

                steps_so_far.pop()
                visited_smiles.discard(prod_smi)


def build_random_routes(
    mol_smi: str,
    tp,
    n_routes: int = 1,
    n_steps: int = 3,
    seed: int = 42,
) -> list[SynthesisRoute]:
    """Build random routes by greedily applying templates.

    Simpler than DFS decomposition -- just applies random templates/BBs
    at each step. Useful for bootstrapping RL training.

    Args:
        mol_smi: Starting molecule SMILES.
        tp: Loaded TemplateReactionPredictor.
        n_routes: Number of routes to generate.
        n_steps: Number of steps per route.
        seed: Random seed.

    Returns:
        List of SynthesisRoute objects.
    """
    rng = _random.Random(seed)
    mol = Chem.MolFromSmiles(mol_smi)
    if mol is None:
        return []

    canonical_smi = Chem.MolToSmiles(mol)
    routes = []

    for route_idx in range(n_routes * 5):  # oversample due to failures
        if len(routes) >= n_routes:
            break

        current_mol = mol
        current_smi = canonical_smi
        steps = []

        for step_idx in range(n_steps):
            # Collect applicable bi-reactions (prioritize over uni)
            applicable = []
            for bi_idx, bi_rxn in enumerate(tp.bi_reactions):
                if not bi_rxn.is_mol_reactant(current_mol):
                    continue
                compat = tp.bi_compat.get(bi_idx)
                if compat is not None and len(compat) > 0:
                    applicable.append(('bi', bi_idx, bi_rxn, compat))

            # Also collect uni-reactions
            for idx, uni_rxn in enumerate(tp.uni_reactions):
                if uni_rxn.is_reactant(current_mol):
                    applicable.append(('uni', idx, uni_rxn, None))

            if not applicable:
                break

            rng.shuffle(applicable)
            success = False

            for rxn_type, rxn_idx, rxn_obj, compat in applicable:
                if rxn_type == 'uni':
                    products = rxn_obj.forward_smiles(current_mol)
                    if products:
                        prod_smi = products[0]
                        if prod_smi and prod_smi != current_smi:
                            step = RouteStep(
                                template_idx=rxn_obj.index,
                                bi_rxn_idx=-1,
                                block_idx=-1,
                                block_smi="",
                                intermediate_smi=prod_smi,
                                is_uni=True,
                                uni_rxn_idx=rxn_idx,
                            )
                            steps.append(step)
                            current_mol = Chem.MolFromSmiles(prod_smi)
                            current_smi = prod_smi
                            success = True
                            break
                else:  # bi
                    n_try = min(10, len(compat))
                    blk_indices = rng.sample(list(compat), n_try)
                    for blk_idx in blk_indices:
                        blk_smi, blk_mol = tp.bb_library[int(blk_idx)]
                        products = rxn_obj.forward_smiles(
                            current_mol, blk_mol)
                        if products:
                            prod_smi = products[0]
                            if prod_smi and prod_smi != current_smi:
                                step = RouteStep(
                                    template_idx=rxn_obj.index,
                                    bi_rxn_idx=rxn_idx,
                                    block_idx=int(blk_idx),
                                    block_smi=blk_smi,
                                    intermediate_smi=prod_smi,
                                    is_uni=False,
                                )
                                steps.append(step)
                                current_mol = Chem.MolFromSmiles(prod_smi)
                                current_smi = prod_smi
                                success = True
                                break
                    if success:
                        break

            if not success:
                break

        if len(steps) >= 2:
            route = SynthesisRoute(
                steps=steps,
                init_mol_smi=canonical_smi,
                final_product_smi=current_smi,
            )
            routes.append(route)

    return routes[:n_routes]


def build_test_routes(tp) -> list[SynthesisRoute]:
    """Build hardcoded test routes for smoke testing."""
    test_mols = [
        'c1ccc(N)cc1',    # aniline
        'c1ccc(O)cc1',    # phenol
        'CCO',            # ethanol
        'CC(=O)O',        # acetic acid
        'c1ccccc1',       # benzene
        'CC(C)N',         # isopropylamine
    ]

    routes = []
    for mol_smi in test_mols:
        mol_routes = build_random_routes(
            mol_smi, tp, n_routes=1, n_steps=3, seed=42)
        routes.extend(mol_routes)

    return routes


# ─────────────── AiZynthFinder Route Conversion ───────────────


def _build_native_steps(node: dict) -> tuple[str, list[dict]]:
    """Recursively build forward-order steps from AiZynth route tree.

    Returns:
        (current_smi, steps) where:
        - current_smi: SMILES available after executing all steps.
        - steps: list of step dicts in forward synthesis order, each with:
          template_code, chain_smi, bb_smiles, product_smi, n_reactants.
    """
    if node['type'] != 'mol' or 'children' not in node or not node['children']:
        # Leaf molecule (in_stock or terminal)
        return node['smiles'], []

    rxn_node = node['children'][0]
    child_mols = rxn_node['children']
    meta = rxn_node.get('metadata', {})
    template_code = meta.get('template_code', -1)

    # Separate stock (BB) vs non-stock (chain) children
    stock_children = [c for c in child_mols if c.get('in_stock', False)]
    non_stock_children = [c for c in child_mols
                          if not c.get('in_stock', False)]

    # Recursively process non-stock children (0 or 1 for linear routes)
    all_steps: list[dict] = []
    chain_smiles: list[str] = []
    for nsc in non_stock_children:
        child_smi, child_steps = _build_native_steps(nsc)
        all_steps.extend(child_steps)
        chain_smiles.append(child_smi)

    # Determine chain_mol and BB(s)
    if non_stock_children:
        chain_smi = chain_smiles[0]
        bb_smiles = [c['smiles'] for c in stock_children]
    else:
        # All children in_stock → pick first as init_mol/chain, rest as BBs
        chain_smi = stock_children[0]['smiles']
        bb_smiles = [c['smiles'] for c in stock_children[1:]]

    all_steps.append({
        'template_code': template_code,
        'chain_smi': chain_smi,
        'bb_smiles': bb_smiles,
        'product_smi': node['smiles'],
        'n_reactants': len(child_mols),
    })
    return node['smiles'], all_steps


def build_routes_from_aizynth(
    aizynth_entries: list[dict],
    tp,
    code_to_idx: dict[int, int] | None = None,
) -> list[SynthesisRoute]:
    """Build routes directly from AiZynthFinder data using native templates.

    Requires tp loaded with AiZynth-compatible forward templates
    (template/data/templates_aizynth.txt) and AiZynth BBs
    (template/data/building_blocks_aizynth.smi).

    Preserves the EXACT original routes: same templates, same BBs, same products.

    Args:
        aizynth_entries: List of AiZynth result dicts (with 'first_route').
        tp: TemplateReactionPredictor loaded with AiZynth templates.
        code_to_idx: Mapping from AiZynth template_code to template file line index.
                     If None, loads from template/data/aizynth_code_to_idx.pkl.

    Returns:
        List of SynthesisRoute objects.
    """
    if code_to_idx is None:
        import pickle
        from pathlib import Path
        pkl_path = (Path(__file__).resolve().parent.parent
                    / 'template' / 'data' / 'aizynth_code_to_idx.pkl')
        with open(pkl_path, 'rb') as f:
            code_to_idx = pickle.load(f)

    # Build BB lookup: canonical SMILES -> library index
    bb_lookup: dict[str, int] = {}
    for i, smi in enumerate(tp.bb_library.smiles_list):
        bb_lookup[smi] = i

    routes: list[SynthesisRoute] = []
    for entry in aizynth_entries:
        if not entry.get('is_solved', False):
            continue

        _, linear_steps = _build_native_steps(entry['first_route'])
        if not linear_steps:
            continue

        route = _convert_native_route(linear_steps, tp, code_to_idx, bb_lookup)
        if route is not None:
            routes.append(route)

    return routes


def build_routes_from_paroutes(
    annotated_routes: list[dict],
    tp,
    code_to_idx: dict[int, int] | None = None,
) -> list[SynthesisRoute]:
    """Build routes from PaRoutes data with classified template_codes.

    Requires tp loaded with PaRoutes-compatible forward templates
    (template/data/templates_paroutes.txt) and PaRoutes BBs
    (template/data/building_blocks_paroutes.smi).

    Args:
        annotated_routes: List of PaRoutes route trees (each a dict with
            template_code injected into reaction node metadata).
        tp: TemplateReactionPredictor loaded with PaRoutes templates.
        code_to_idx: Mapping from AiZynth template_code to template file
                     line index. If None, loads from
                     template/data/paroutes_code_to_idx.pkl.

    Returns:
        List of SynthesisRoute objects.
    """
    if code_to_idx is None:
        import pickle
        from pathlib import Path
        pkl_path = (Path(__file__).resolve().parent.parent
                    / 'template' / 'data' / 'paroutes_code_to_idx.pkl')
        with open(pkl_path, 'rb') as f:
            code_to_idx = pickle.load(f)

    # Build BB lookup: canonical SMILES -> library index
    bb_lookup: dict[str, int] = {}
    for i, smi in enumerate(tp.bb_library.smiles_list):
        bb_lookup[smi] = i

    routes: list[SynthesisRoute] = []
    n_failed = 0
    for route_tree in annotated_routes:
        _, linear_steps = _build_native_steps(route_tree)
        if not linear_steps:
            n_failed += 1
            continue

        route = _convert_native_route(linear_steps, tp, code_to_idx, bb_lookup)
        if route is not None:
            routes.append(route)
        else:
            n_failed += 1

    if n_failed > 0:
        print(f"  PaRoutes: {len(routes)} routes built, {n_failed} failed")

    return routes


def _convert_native_route(
    steps: list[dict],
    tp,
    code_to_idx: dict[int, int],
    bb_lookup: dict[str, int],
) -> SynthesisRoute | None:
    """Convert linearized AiZynth steps to a SynthesisRoute using native templates.

    For each step:
    1. Look up template_code → template_idx → bi_rxn_idx (correct orientation).
    2. Look up BB SMILES → bb_library index.
    3. Forward-execute to verify and get canonical product.
    """
    if not steps:
        return None

    init_smi = steps[0]['chain_smi']
    init_mol = Chem.MolFromSmiles(init_smi)
    if init_mol is None:
        return None
    init_smi = Chem.MolToSmiles(init_mol)

    current_mol = init_mol
    current_smi = init_smi
    route_steps: list[RouteStep] = []

    for step_data in steps:
        template_code = step_data['template_code']
        bb_smiles_list = step_data['bb_smiles']
        n_reactants = step_data['n_reactants']

        # Skip tri-molecular reactions (mark as non-modifiable passthrough)
        if n_reactants > 2 or not bb_smiles_list:
            # Use the AiZynth product directly (no template execution)
            product_smi = step_data['product_smi']
            product_mol = Chem.MolFromSmiles(product_smi)
            if product_mol is None:
                break
            product_smi = Chem.MolToSmiles(product_mol)
            route_steps.append(RouteStep(
                template_idx=-1, bi_rxn_idx=-1, block_idx=-1,
                block_smi="", intermediate_smi=product_smi,
                is_uni=True, uni_rxn_idx=-1,
            ))
            current_mol = product_mol
            current_smi = product_smi
            continue

        # Uni-molecular (n_reactants == 1): no BB
        template_idx = code_to_idx.get(template_code, -1)
        if template_idx < 0:
            break

        if n_reactants == 1:
            # Find matching uni reaction
            uni_rxn_idx = -1
            for ui, uni_rxn in enumerate(tp.uni_reactions):
                if uni_rxn.index == template_idx:
                    uni_rxn_idx = ui
                    break
            if uni_rxn_idx < 0:
                break
            uni_rxn = tp.uni_reactions[uni_rxn_idx]
            products = uni_rxn.forward_smiles(current_mol)
            if not products:
                break
            route_steps.append(RouteStep(
                template_idx=template_idx, bi_rxn_idx=-1, block_idx=-1,
                block_smi="", intermediate_smi=products[0],
                is_uni=True, uni_rxn_idx=uni_rxn_idx,
            ))
            current_mol = Chem.MolFromSmiles(products[0])
            current_smi = products[0]
            continue

        # Bi-molecular: find correct orientation and BB
        bb_smi = bb_smiles_list[0]
        bb_mol = Chem.MolFromSmiles(bb_smi)
        if bb_mol is None:
            break
        bb_canon = Chem.MolToSmiles(bb_mol)
        bb_idx = bb_lookup.get(bb_canon, -1)
        if bb_idx < 0:
            break

        # Find the bi_rxn_idx with correct orientation
        matching_bi = [(i, br) for i, br in enumerate(tp.bi_reactions)
                       if br.index == template_idx]

        product_smi = None
        chosen_bi_idx = -1
        for bi_idx, bi_rxn in matching_bi:
            if bi_rxn.is_mol_reactant(current_mol):
                products = bi_rxn.forward_smiles(current_mol, bb_mol)
                if products:
                    product_smi = products[0]
                    chosen_bi_idx = bi_idx
                    break

        if product_smi is None:
            break

        route_steps.append(RouteStep(
            template_idx=template_idx,
            bi_rxn_idx=chosen_bi_idx,
            block_idx=bb_idx,
            block_smi=bb_canon,
            intermediate_smi=product_smi,
            is_uni=False,
        ))
        current_mol = Chem.MolFromSmiles(product_smi)
        if current_mol is None:
            break
        current_smi = product_smi

    if not route_steps:
        return None

    return SynthesisRoute(
        steps=route_steps,
        init_mol_smi=init_smi,
        final_product_smi=current_smi,
    )


# ─────────────────── CLI ───────────────────

if __name__ == "__main__":
    import argparse
    from rl.template.template_predictor import TemplateReactionPredictor

    parser = argparse.ArgumentParser(
        description="Test retrosynthetic decomposition")
    parser.add_argument(
        '--smiles', type=str, nargs='+',
        default=['c1ccc(N)cc1', 'CCO', 'c1ccc(O)cc1', 'CC(=O)O'])
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--max_routes', type=int, default=3)
    parser.add_argument('--min_steps', type=int, default=2)
    parser.add_argument('--method', type=str, default='retro',
                        choices=['random', 'dfs', 'retro'])
    args = parser.parse_args()

    tp = TemplateReactionPredictor(num_workers=0)
    tp.load()

    print(f"\n{'='*60}")
    print(f"Method: {args.method}")

    for smi in args.smiles:
        t0 = time.perf_counter()
        if args.method == 'random':
            routes = build_random_routes(smi, tp, n_routes=args.max_routes,
                                         n_steps=args.max_depth)
        elif args.method == 'dfs':
            routes = decompose_molecule(
                smi, tp, max_depth=args.max_depth,
                max_routes=args.max_routes)
        else:  # retro
            routes = build_retro_routes(
                smi, tp, max_depth=args.max_depth,
                max_routes=args.max_routes,
                min_steps=args.min_steps)
        elapsed = time.perf_counter() - t0

        print(f"\n[{smi}] -> {len(routes)} routes in {elapsed*1000:.0f}ms")
        for i, route in enumerate(routes):
            n_bi = sum(1 for s in route.steps if not s.is_uni)
            print(f"  Route {i}: {len(route)} steps, "
                  f"{route.n_modifiable} modifiable ({n_bi} bi)")
            print(f"    Init: {route.init_mol_smi}")
            for j, step in enumerate(route.steps):
                rxn_type = "uni" if step.is_uni else "bi"
                bb_str = f" + {step.block_smi}" if step.block_smi else ""
                print(f"    Step {j}: [{rxn_type}] T{step.template_idx}"
                      f"{bb_str} -> {step.intermediate_smi}")
            print(f"    Final: {route.final_product_smi}")
    print(f"\n{'='*60}")
