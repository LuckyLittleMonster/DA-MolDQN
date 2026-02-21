"""QED-based reward computation."""

from rdkit import Chem
from rdkit.Chem import QED as QEDModule

from .core import _load_sascorer


def compute_reward_qed(smiles, step, max_steps, gamma,
                       qed_weight=0.8, sa_weight=0.2):
    """QED-based reward. Returns unified reward dict."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'reward': 0.0, 'qed': 0.0, 'sa': 10.0, 'dock_score': 0.0, 'valid': False}
    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    qed = QEDModule.qed(mol)
    raw = qed_weight * qed + sa_weight * sa_norm
    return {'reward': raw, 'qed': qed, 'sa': sa, 'dock_score': 0.0, 'valid': True}
