"""BDE predictor orchestrator for the RL reward.

Owns the BDE model, its scaler, and the BDE reward factor. A *pure* predictor:
caching is handled externally by ``cached`` (composed in the BDE_IP reward),
so this class no longer knows about caches.
"""
from src.config import RewardCfg
from src.reward.bde.model import BDEModel
from src.reward.scaler import get_scaler


class BDEPredictor:
    """Predicts O-H bond-dissociation energies (pure; no cache)."""

    def __init__(self, device, bde_factor=RewardCfg.bde_factor):
        self.device = device
        self.bde_factor = bde_factor

        self.bde_scaler = get_scaler('./Data/anti-bde.csv')

        self.bde_model = BDEModel(
            'src/reward/bde/weights/alfabet.npz',
            preprocessor_path='src/reward/bde/weights/alfabet_preprocessor.json',
            device=str(self.device))

    def predict_BDE(self, smiles, mols):
        """Pure prediction. Returns ``(values, valids)`` parallel to ``smiles``."""
        return self.bde_model.predict_oh_bde(smiles)
