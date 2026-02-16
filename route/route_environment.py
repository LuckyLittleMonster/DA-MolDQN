"""RL environment for synthesis route optimization.

The agent modifies synthesis routes by swapping building blocks at selected
positions. Each episode starts from initial routes and runs max_steps
BB-swap actions. The reward is based on the final product's properties.
"""

from __future__ import annotations

import os
import sys

from rdkit import Chem
from rdkit.Chem import QED as QEDModule

from route.route import SynthesisRoute, cascade_validate, update_route


class RouteEnvironment:
    """RL environment for route-level BB optimization.

    Each step: select a position in the route, select a BB, update the route.
    Reward is computed from the final product after the route is re-executed.

    Args:
        routes: List of initial synthesis routes.
        tp: Loaded TemplateReactionPredictor.
        max_steps: Maximum BB-swap steps per episode.
        discount: Discount factor for reward.
        qed_weight: Weight for QED in reward.
        sa_weight: Weight for SA score in reward.
    """

    def __init__(self, routes: list[SynthesisRoute], tp,
                 max_steps: int = 5, discount: float = 0.9,
                 qed_weight: float = 0.8, sa_weight: float = 0.2,
                 dock_scorer=None, dock_weight: float = 0.7):
        self.init_routes = routes
        self.routes = [r.copy() for r in routes]
        self.tp = tp
        self.max_steps = max_steps
        self.discount = discount
        self.qed_weight = qed_weight
        self.sa_weight = sa_weight
        self.dock_scorer = dock_scorer
        self.dock_weight = dock_weight
        self.step_count = 0

        # SA scorer (loaded once)
        from rdkit import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        self._sascorer = sascorer

    @property
    def n_routes(self) -> int:
        return len(self.routes)

    def reset(self) -> list[SynthesisRoute]:
        """Reset all routes to initial state."""
        self.routes = [r.copy() for r in self.init_routes]
        self.step_count = 0
        return self.routes

    def step(self, positions: list[int], block_indices: list[int]):
        """Execute one step: swap BB at selected position in each route.

        Args:
            positions: Per-route position index to modify.
            block_indices: Per-route new BB index.

        Returns:
            dict with 'rewards', 'done', 'QED', 'SA_score', 'success'.
        """
        rewards = []
        qeds = []
        sas = []
        successes = []

        for i, (route, pos, blk) in enumerate(
                zip(self.routes, positions, block_indices)):
            old_product = route.final_product_smi

            # Attempt to update route
            new_product = update_route(route, pos, blk, self.tp)

            if new_product is not None:
                reward, qed, sa = self._calc_reward(new_product)
                successes.append(True)
            else:
                # Failed — revert by re-executing from scratch
                route.forward_execute(0, self.tp)
                reward = -0.1  # small penalty for failed action
                mol = Chem.MolFromSmiles(route.final_product_smi)
                if mol:
                    qed = QEDModule.qed(mol)
                    sa = self._sascorer.calculateScore(mol)
                else:
                    qed = 0.0
                    sa = 10.0
                successes.append(False)

            # Apply discount
            reward *= (self.discount ** (self.max_steps - self.step_count - 1))
            rewards.append(reward)
            qeds.append(qed)
            sas.append(sa)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return {
            'rewards': rewards,
            'done': done,
            'QED': qeds,
            'SA_score': sas,
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

    def _calc_reward(self, product_smi: str) -> tuple[float, float, float]:
        """Calculate reward from product SMILES.

        Returns:
            (reward, qed, sa_score)
        """
        mol = Chem.MolFromSmiles(product_smi)
        if mol is None:
            return -0.5, 0.0, 10.0

        try:
            Chem.SanitizeMol(mol)
            qed = QEDModule.qed(mol)
            sa = self._sascorer.calculateScore(mol)
            sa_normalized = (10 - sa) / 9

            if self.dock_scorer is not None:
                scores = self.dock_scorer.batch_dock([product_smi])
                dock_score = scores[0]
                dock_norm = max(0.0, min(1.0, -dock_score / 12.0))
                reward = (self.dock_weight * dock_norm +
                          self.sa_weight * sa_normalized +
                          (1.0 - self.dock_weight - self.sa_weight) * qed)
            else:
                reward = self.qed_weight * qed + self.sa_weight * sa_normalized
            return reward, qed, sa
        except Exception:
            return -0.5, 0.0, 10.0
