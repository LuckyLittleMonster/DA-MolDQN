"""Docking-based reward computation."""

from rdkit import Chem
from rdkit.Chem import QED as QEDModule

from ..core import _load_sascorer


def compute_reward_dock(smiles, step, max_steps, gamma,
                        dock_scorer=None, dock_weight=0.7, sa_weight=0.2,
                        qed_weight=0.0, dock_score=None, mol=None):
    """Docking-based reward. Returns unified reward dict.

    If *dock_score* is provided, the scorer is not called.
    Set qed_weight > 0 for QED+dock combination (default dock mode).
    """
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'reward': 0.0, 'qed': 0.0, 'sa': 10.0, 'dock_score': 0.0, 'valid': False}
    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    if dock_score is None:
        scores = dock_scorer.batch_dock([smiles], mols=[mol])
        dock_score = scores[0]
    dock_norm = max(0.0, min(1.0, -dock_score / 12.0))
    qed = QEDModule.qed(mol)  # always compute real QED
    raw = dock_weight * dock_norm + sa_weight * sa_norm + qed_weight * qed
    return {'reward': raw, 'qed': qed, 'sa': sa, 'dock_score': dock_score, 'valid': True}
