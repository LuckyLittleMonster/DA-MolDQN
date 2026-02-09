"""ActionGenerator implementations for the two-model and hybrid pipelines."""

import numpy as np
from typing import List, Tuple

from action_generator import ActionGenerator


class TwoModelActionGenerator(ActionGenerator):
    """Two-model pipeline: link prediction (Model 1) + ReactionT5v2 (Model 2).

    Wraps ReactionPredictor to implement the ActionGenerator interface.
    """

    name = "hypergraph"

    def __init__(self, config=None, device='auto', **kwargs):
        from model_reactions.reaction_predictor import ReactionPredictor
        from model_reactions.config import ReactionPredictorConfig
        if config is None:
            config = ReactionPredictorConfig(**kwargs)
        self._predictor = ReactionPredictor(config=config, device=device)

    def load(self) -> None:
        self._predictor.load()

    def get_valid_actions(
        self, mol, top_k: int = None
    ) -> Tuple[List[str], List[str], np.ndarray]:
        return self._predictor.get_valid_actions(mol, top_k=top_k)

    def get_valid_actions_batch(self, mols, top_k: int = None) -> list:
        return self._predictor.get_valid_actions_batch(mols, top_k=top_k)


class HybridActionGenerator(ActionGenerator):
    """AIO initial retrieval + V3 re-ranking.

    Wraps HybridReactionPredictor to implement the ActionGenerator interface.
    """

    name = "hybrid"

    def __init__(self, **kwargs):
        from model_reactions.hybrid_predictor import HybridReactionPredictor
        self._predictor = HybridReactionPredictor(**kwargs)

    def load(self) -> None:
        self._predictor.load()

    def get_valid_actions(
        self, mol, top_k: int = None
    ) -> Tuple[List[str], List[str], np.ndarray]:
        return self._predictor.get_valid_actions(mol, top_k=top_k)

    def get_valid_actions_batch(self, mols, top_k: int = None) -> list:
        return self._predictor.get_valid_actions_batch(mols, top_k=top_k)
