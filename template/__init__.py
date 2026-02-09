"""Template-based reaction predictor package.

Uses RDKit reaction SMARTS (from RxnFlow/SynFlowNet) for deterministic,
100% chemically valid molecular transformations.
"""

from template.reaction import Reaction, UniReaction, BiReaction, load_templates
from template.building_blocks import BuildingBlockLibrary
from template.template_predictor import TemplateReactionPredictor
from template.action_generator import TemplateActionGenerator

__all__ = [
    "Reaction",
    "UniReaction",
    "BiReaction",
    "load_templates",
    "BuildingBlockLibrary",
    "TemplateReactionPredictor",
    "TemplateActionGenerator",
]
