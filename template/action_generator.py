"""TemplateActionGenerator: RDKit SMARTS template-based action generator.

Wraps TemplateReactionPredictor to implement the ActionGenerator interface.
"""

import numpy as np
from typing import List, Tuple

from action_generator import ActionGenerator
from template.template_predictor import TemplateReactionPredictor


class TemplateActionGenerator(ActionGenerator):
    """RDKit SMARTS template-based reaction predictor.

    Uses deterministic RDKit reaction templates for 100% chemically valid
    product generation. No GPU required.
    """

    name = "template"

    def __init__(self, **kwargs):
        self._predictor = TemplateReactionPredictor(**kwargs)

    def load(self) -> None:
        self._predictor.load()

    def get_valid_actions(
        self, mol, top_k: int = None
    ) -> Tuple[List[str], List[str], np.ndarray]:
        return self._predictor.get_valid_actions(mol, top_k=top_k)

    def get_valid_actions_batch(
        self, mols, top_k: int = None
    ) -> list:
        return self._predictor.get_valid_actions_batch(mols, top_k=top_k)
