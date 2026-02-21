"""Template-based reaction predictor package.

Uses RDKit reaction SMARTS (from RxnFlow/SynFlowNet) for deterministic,
100% chemically valid molecular transformations.

Includes DQN-guided predictor that uses learned Q-networks for
template and building block selection.
"""

from .reaction import Reaction, UniReaction, BiReaction, load_templates
from .building_blocks import BuildingBlockLibrary
from .template_predictor import TemplateReactionPredictor

__all__ = [
    "Reaction",
    "UniReaction",
    "BiReaction",
    "load_templates",
    "BuildingBlockLibrary",
    "TemplateReactionPredictor",
]
