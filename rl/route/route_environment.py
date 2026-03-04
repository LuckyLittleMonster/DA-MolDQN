"""RL environment for synthesis route optimization.

The agent modifies synthesis routes by swapping building blocks at selected
positions. Each episode starts from initial routes and runs max_steps
BB-swap actions. The reward is based on the final product's properties,
computed via the unified reward functions in the reward package.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED as QEDModule

from .route import (
    SynthesisRoute, cascade_validate, update_route,
    extend_route, truncate_route,
)


class RouteEnvironment:
    """RL environment for route-level BB optimization.

    Each step: select a position in the route, select a BB, update the route.
    Reward is computed from the final product using the unified reward dispatch
    (supports qed, dock, and multi-objective with Lipinski/PAINS penalties).

    Args:
        routes: List of initial synthesis routes.
        tp: Loaded TemplateReactionPredictor.
        max_steps: Maximum BB-swap steps per episode.
        discount: Deprecated, kept for API compatibility (DQN handles γ).
        cfg_reward: OmegaConf reward config (from configs/reward/*.yaml).
        dock_scorer: UniDockScorer instance or None.
    """

    def __init__(self, routes: list[SynthesisRoute], tp,
                 max_steps: int = 5, discount: float = 0.9,
                 cfg_reward=None, dock_scorer=None):
        self.init_routes = routes
        self.routes = [r.copy() for r in routes]
        self.tp = tp
        self.max_steps = max_steps
        self.cfg_reward = cfg_reward
        self.dock_scorer = dock_scorer
        self.step_count = 0

    @property
    def n_routes(self) -> int:
        return len(self.routes)

    def reset(self) -> list[SynthesisRoute]:
        """Reset all routes to initial state."""
        self.routes = [r.copy() for r in self.init_routes]
        self.step_count = 0
        return self.routes

    def step(self, positions: list[int], block_indices: list[int],
             action_types: list[str] | None = None,
             extend_bi_rxn_indices: list[int] | None = None):
        """Execute one step: swap/extend/truncate each route.

        Args:
            positions: Per-route position index to modify.
            block_indices: Per-route new BB index (-1 for truncate).
            action_types: Per-route action ('swap', 'extend', 'truncate').
                Defaults to all 'swap' for backward compatibility.
            extend_bi_rxn_indices: Per-route bi_rxn_idx for extend actions.

        Returns:
            dict with 'rewards', 'done', 'QED', 'SA_score', 'dock_score',
            'success'.
        """
        rewards = []
        qeds = []
        sas = []
        dock_scores = []
        successes = []

        # Batch docking: collect all products that need docking first
        products_to_dock = []
        product_mols_to_dock = []
        product_indices = []
        new_products = []
        new_mols = []  # Pre-parsed mol objects (created once, reused)

        for i, (route, pos, blk) in enumerate(
                zip(self.routes, positions, block_indices)):
            atype = action_types[i] if action_types else 'swap'

            if atype == 'truncate':
                new_product = truncate_route(route)
            elif atype == 'extend':
                bi_rxn_idx = (extend_bi_rxn_indices[i]
                              if extend_bi_rxn_indices else -1)
                if bi_rxn_idx >= 0:
                    new_product = extend_route(
                        route, bi_rxn_idx, blk, self.tp)
                else:
                    new_product = None
            else:
                new_product = update_route(route, pos, blk, self.tp)

            new_products.append(new_product)

            # Parse mol once, reuse for docking + reward
            if new_product is not None:
                mol = Chem.MolFromSmiles(new_product)
                new_mols.append(mol)
                if mol is not None:
                    products_to_dock.append(new_product)
                    product_mols_to_dock.append(mol)
                    product_indices.append(i)
            else:
                new_mols.append(None)

        # Batch dock all successful products at once
        dock_results = {}
        if products_to_dock and self.dock_scorer is not None:
            scores = self.dock_scorer.batch_dock(
                products_to_dock, mols=product_mols_to_dock)
            for idx, smi, score in zip(product_indices, products_to_dock, scores):
                dock_results[idx] = score

        # Batch ADMET prediction for all successful products
        admet_results = {}
        if (products_to_dock and self.cfg_reward is not None
                and self.cfg_reward.name == 'admet'):
            from reward.admet.reward import _get_admet_model
            admet_model = _get_admet_model()
            batch_preds = admet_model.predict_properties(products_to_dock)
            if hasattr(batch_preds, 'iloc'):
                for j, idx in enumerate(product_indices):
                    admet_results[idx] = batch_preds.iloc[j].to_dict()
            else:
                # Single molecule → dict
                admet_results[product_indices[0]] = batch_preds

        # Compute rewards
        for i, (route, new_product) in enumerate(
                zip(self.routes, new_products)):
            if new_product is not None:
                dock_score = dock_results.get(i)
                admet_preds = admet_results.get(i)
                rdict = self._calc_reward(
                    new_product, mol=new_mols[i],
                    dock_score=dock_score,
                    admet_preds=admet_preds)
                reward = rdict['reward']
                qed = rdict['qed']
                sa = rdict['sa']
                dock_val = rdict['dock_score']
                successes.append(True)
            else:
                reward = -0.1
                mol = Chem.MolFromSmiles(route.final_product_smi)
                if mol:
                    from reward.core import _load_sascorer
                    qed = QEDModule.qed(mol)
                    sa = _load_sascorer().calculateScore(mol)
                else:
                    qed = 0.0
                    sa = 10.0
                dock_val = 0.0
                successes.append(False)

            # No discount here — DQN handles γ via Bellman equation
            rewards.append(reward)
            qeds.append(qed)
            sas.append(sa)
            dock_scores.append(dock_val)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return {
            'rewards': rewards,
            'done': done,
            'QED': qeds,
            'SA_score': sas,
            'dock_score': dock_scores,
            'success': successes,
        }

    def get_candidate_blocks(self, route: SynthesisRoute,
                             position: int) -> list[int]:
        """Get L2 + L3 validated candidate BB indices for a position.

        L2: Template compatibility (precomputed, O(1) lookup).
        L3: Cascade validation (forward execution from position to end).

        Args:
            route: Current route.
            position: Step index to modify.

        Returns:
            List of valid block indices after L2 + L3 filtering.
        """
        step = route.steps[position]
        if step.is_uni:
            return []

        # L2: template compatibility
        l2_candidates = self.tp.bi_compat.get(step.bi_rxn_idx)
        if l2_candidates is None or len(l2_candidates) == 0:
            return []

        # L3: cascade validation
        l3_valid = cascade_validate(
            route, position, l2_candidates.tolist(), self.tp)

        return l3_valid

    def _calc_reward(self, product_smi: str, mol=None,
                     dock_score: float | None = None,
                     admet_preds: dict | None = None,
                     ) -> dict:
        """Calculate reward using unified reward dispatch.

        Uses cfg_reward to determine mode (qed/dock/multi/admet) and apply
        appropriate penalties (Lipinski, PAINS, LogP, ADMET toxicity).

        Returns:
            Unified reward dict (reward, qed, sa, dock_score, valid, ...).
            Discount applied by caller.
        """
        from reward import compute_reward

        try:
            # step=0, max_steps=1, gamma=1.0 → no discount here (caller applies)
            return compute_reward(
                product_smi, step=0, max_steps=1, gamma=1.0,
                cfg_reward=self.cfg_reward,
                dock_scorer=self.dock_scorer,
                dock_score=dock_score,
                admet_preds=admet_preds,
                mol=mol)
        except Exception:
            return {'reward': -0.5, 'qed': 0.0, 'sa': 10.0,
                    'dock_score': 0.0, 'valid': False}
