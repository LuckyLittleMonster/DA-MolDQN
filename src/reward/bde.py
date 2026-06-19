"""BDE reward component.

Canonical home for the bond-dissociation-energy model (torch ALFABET).
``BDEModel`` lives in the vendored ``bde_predictor`` package; re-exported here so
all reward components are reachable under ``src.reward``.
"""
from src.reward.bde_predictor.predict import BDEModel

WEIGHTS = "src/reward/bde_predictor/weights/alfabet.npz"
PREPROCESSOR = "src/reward/bde_predictor/weights/alfabet_preprocessor.json"

__all__ = ["BDEModel", "WEIGHTS", "PREPROCESSOR"]
