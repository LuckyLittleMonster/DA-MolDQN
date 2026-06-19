"""Reward component registry.

Each reward is implemented in its own module under ``src.reward``:
qed / plogp / sa / bde / ip. Combination + weighting + per-step discount remain
the responsibility of the environment-level reward methods (MultiMolecules),
which call these per-molecule primitives.
"""
from src.reward.qed import qed_value
from src.reward.sa import sa_score
from src.reward.plogp import plogp_value

# Per-molecule scoring primitives, keyed by component name.
COMPONENTS = {
    "qed": qed_value,
    "sa": sa_score,
    "plogp": plogp_value,
}


def get_component(name):
    return COMPONENTS[name]
