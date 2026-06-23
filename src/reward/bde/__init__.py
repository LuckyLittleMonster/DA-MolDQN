"""BDE reward component.

Canonical home for the bond-dissociation-energy model (torch ALFABET).
``BDEModel`` (inference) lives in ``model``; the ``BDENet`` nn.Module it loads is
in ``net``. Re-exported here together with the ``BDEPredictor`` orchestrator and
the default weight/preprocessor paths so all reward components are reachable
under ``src.reward.bde``.
"""
from src.reward.bde.model import BDEModel
from src.reward.bde.predictor import BDEPredictor

WEIGHTS = "src/reward/bde/weights/alfabet.npz"
PREPROCESSOR = "src/reward/bde/weights/alfabet_preprocessor.json"

__all__ = ["BDEModel", "BDEPredictor", "WEIGHTS", "PREPROCESSOR"]
