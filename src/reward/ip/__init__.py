"""IP (ionization potential) reward component.

The AIMNet-NSE IP model + reactivity-index features live in ``model``; the
predictor orchestrator in ``predictor``. Re-exported here so all reward
components are reachable under ``src.reward.ip``.
"""
from src.reward.ip.model import AimnetNseModel, calc_react_idx, ev2kcal_per_mol
from src.reward.ip.predictor import IPPredictor

__all__ = ["IPPredictor", "AimnetNseModel", "calc_react_idx", "ev2kcal_per_mol"]
