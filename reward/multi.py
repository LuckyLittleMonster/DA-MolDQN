"""Multi-objective reward computation."""

from rdkit import Chem
from rdkit.Chem import QED as QEDModule, Descriptors

from .core import _load_sascorer, _check_lipinski, _check_pains


def compute_reward_multi(smiles, step, max_steps, gamma,
                         cfg_reward, dock_scorer=None, dock_score=None):
    """Multi-objective reward.

    Supports two strategies:
      'product'  -- baseline-matching: dock_norm x QED x SA_norm (RxnFlow vina_moo)
      'layered'  -- our version: hard constraints + weighted-sum scalarization

    Returns unified reward dict.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'reward': 0.0, 'qed': 0.0, 'sa': 10.0, 'dock_score': 0.0,
                'logp': 0.0, 'valid': False, 'lipinski': False, 'pains': False}

    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    qed = QEDModule.qed(mol)

    # Compute dock_norm
    dock_norm = 0.0
    if dock_score is not None:
        dock_norm = max(0.0, min(1.0, -dock_score / 12.0))
    elif dock_scorer is not None:
        scores = dock_scorer.batch_dock([smiles])
        dock_score = scores[0]
        dock_norm = max(0.0, min(1.0, -dock_score / 12.0))

    strategy = cfg_reward.get('strategy', 'layered')

    if strategy == 'product':
        # Baseline-matching: product of objectives (RxnFlow vina_moo)
        raw = dock_norm * qed * sa_norm
        return {'reward': raw, 'qed': qed, 'sa': sa,
                'dock_score': dock_score or 0.0,
                'logp': Descriptors.MolLogP(mol),
                'valid': True, 'lipinski': True, 'pains': False}

    # --- Layered strategy (our constrained version) ---
    logp = Descriptors.MolLogP(mol)
    metrics = {'qed': qed, 'sa': sa, 'dock_score': dock_score or 0.0,
               'logp': logp, 'valid': True, 'lipinski': True, 'pains': False}

    # Layer 1: Hard constraints
    penalty = 0.0

    # SA threshold
    if sa > cfg_reward.sa_threshold:
        penalty += 0.3

    # Lipinski
    if cfg_reward.lipinski and not _check_lipinski(mol, cfg_reward):
        penalty += 0.2
        metrics['lipinski'] = False

    # PAINS filter
    if _check_pains(mol):
        penalty += 0.3
        metrics['pains'] = True

    # LogP soft penalty (greasy molecule prevention)
    max_logp = cfg_reward.get('max_logp', 5.0)
    if logp > max_logp:
        penalty += 0.1 * (logp - max_logp)

    # Layer 2: Scalarize
    primary = cfg_reward.get('primary', 'qed')
    if primary == 'dock' and (dock_score is not None or dock_scorer is not None):
        raw = (cfg_reward.dock_weight * dock_norm +
               cfg_reward.sa_weight * sa_norm +
               cfg_reward.get('primary_weight', 0.6) * qed)
    else:
        raw = (cfg_reward.get('primary_weight', 0.6) * qed +
               cfg_reward.sa_weight * sa_norm)
        if dock_score is not None or dock_scorer is not None:
            raw += cfg_reward.dock_weight * dock_norm

    raw = max(0.0, raw - penalty)
    metrics['reward'] = raw
    return metrics
