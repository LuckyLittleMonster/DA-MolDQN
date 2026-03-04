"""ADMET-based reward computation."""

import torch
from rdkit import Chem
from rdkit.Chem import QED as QEDModule

from ..core import _load_sascorer, _check_lipinski, _check_pains


# ---------------------------------------------------------------------------
# ADMET reward (FastADMETModel)
# ---------------------------------------------------------------------------

_admet_model = None


def _get_admet_model(device=None):
    """Lazy-init FastADMETModel singleton (shared across all reward calls)."""
    global _admet_model
    if _admet_model is None:
        from .model import FastADMETModel
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _admet_model = FastADMETModel(
            include_physchem=False,      # QED/SA computed separately via RDKit
            drugbank_percentiles=False,  # Not needed for reward
            cache_size=4096,
            device=device,
        )
    return _admet_model


def compute_reward_admet(smiles, step, max_steps, gamma, cfg_reward,
                         dock_scorer=None, dock_score=None,
                         admet_preds=None, mol=None):
    """ADMET multi-property reward using FastADMETModel predictions.

    Combines QED + SA + ADMET absorption/toxicity/metabolism into a single
    scalar reward with hard constraints and soft penalties.

    Returns unified reward dict.
    """
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'reward': 0.0, 'qed': 0.0, 'sa': 10.0, 'dock_score': 0.0, 'valid': False}

    sascorer = _load_sascorer()
    sa = sascorer.calculateScore(mol)
    sa_norm = (10.0 - sa) / 9.0
    qed = QEDModule.qed(mol)

    metrics = {'qed': qed, 'sa': sa, 'dock_score': 0.0, 'valid': True,
               'lipinski': True, 'pains': False}

    # --- Hard constraints (penalty) ---
    penalty = 0.0

    # SA threshold
    if sa > cfg_reward.sa_threshold:
        penalty += 0.3

    # Lipinski
    if cfg_reward.lipinski and not _check_lipinski(mol, cfg_reward):
        penalty += 0.2
        metrics['lipinski'] = False

    # PAINS
    if _check_pains(mol):
        penalty += 0.3
        metrics['pains'] = True

    # --- ADMET predictions (batch-cached or computed) ---
    if admet_preds is None:
        admet_model = _get_admet_model()
        admet_preds = admet_model.predict_properties(smiles)

    # Extract ADMET scores (with safe defaults)
    hia = admet_preds.get('HIA_Hou', 0.5)
    herg = admet_preds.get('hERG', 0.5)
    ames = admet_preds.get('AMES', 0.5)
    dili = admet_preds.get('DILI', 0.5)
    clintox = admet_preds.get('ClinTox', 0.5)
    cyp3a4 = admet_preds.get('CYP3A4_Veith', 0.5)
    solubility = admet_preds.get('Solubility_AqSolDB', -4.0)

    metrics.update({
        'HIA_Hou': hia, 'hERG': herg, 'AMES': ames,
        'DILI': dili, 'ClinTox': clintox, 'CYP3A4_Veith': cyp3a4,
        'Solubility_AqSolDB': solubility,
    })

    # --- Scalarize: positive terms + negative penalty terms ---
    # Positive: QED, SA, HIA (higher=better)
    raw = (cfg_reward.qed_weight * qed +
           cfg_reward.sa_weight * sa_norm +
           cfg_reward.hia_weight * hia)

    # Solubility bonus/penalty
    sol_threshold = cfg_reward.get('solubility_threshold', -4.0)
    if solubility > sol_threshold:
        raw += cfg_reward.solubility_weight  # full bonus
    else:
        raw -= cfg_reward.solubility_weight * min(1.0, (sol_threshold - solubility) / 2.0)

    # Toxicity penalties (prob->0 is good, so subtract prob*weight)
    raw -= cfg_reward.herg_weight * herg
    raw -= cfg_reward.ames_weight * ames
    raw -= cfg_reward.dili_weight * dili
    raw -= cfg_reward.clintox_weight * clintox
    raw -= cfg_reward.cyp3a4_weight * cyp3a4

    # Apply hard-constraint penalty
    raw = max(0.0, raw - penalty)

    # Docking (optional)
    if dock_score is not None:
        dock_norm = max(0.0, min(1.0, -dock_score / 12.0))
        raw += cfg_reward.get('dock_weight', 0.0) * dock_norm
        metrics['dock_score'] = dock_score
    elif dock_scorer is not None:
        scores = dock_scorer.batch_dock([smiles], mols=[mol])
        dock_val = scores[0]
        dock_norm = max(0.0, min(1.0, -dock_val / 12.0))
        raw += cfg_reward.get('dock_weight', 0.0) * dock_norm
        metrics['dock_score'] = dock_val

    metrics['reward'] = raw
    return metrics
