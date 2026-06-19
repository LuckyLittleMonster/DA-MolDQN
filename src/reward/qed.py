"""QED reward component."""
from rdkit.Chem import QED


def qed_value(mol) -> float:
    return QED.qed(mol)
