"""Reward computation for DA-MolDQN."""

from .core import _load_sascorer, _check_lipinski, _check_pains
from .qed import compute_reward_qed
from .multi import compute_reward_multi
from .docking_score.reward import compute_reward_dock
from .docking_score.factory import make_dock_config, make_dock_scorer
from .admet.reward import compute_reward_admet, _get_admet_model


def compute_reward(smiles, step, max_steps, gamma, cfg_reward,
                   dock_scorer=None, dock_score=None,
                   admet_preds=None):
    """Unified reward dispatch based on cfg_reward.name."""
    name = cfg_reward.name
    if name == 'qed':
        return compute_reward_qed(
            smiles, step, max_steps, gamma,
            qed_weight=cfg_reward.qed_weight,
            sa_weight=cfg_reward.sa_weight)
    elif name in ('dock', 'dock_rxnflow', 'dock_deprecated'):
        return compute_reward_dock(
            smiles, step, max_steps, gamma,
            dock_scorer=dock_scorer,
            dock_weight=cfg_reward.dock_weight,
            sa_weight=cfg_reward.sa_weight,
            qed_weight=cfg_reward.get('qed_weight', 0.0),
            dock_score=dock_score)
    elif name in ('multi', 'multi_deprecated'):
        return compute_reward_multi(
            smiles, step, max_steps, gamma,
            cfg_reward=cfg_reward,
            dock_scorer=dock_scorer,
            dock_score=dock_score)
    elif name == 'admet':
        return compute_reward_admet(
            smiles, step, max_steps, gamma,
            cfg_reward=cfg_reward,
            dock_scorer=dock_scorer,
            dock_score=dock_score,
            admet_preds=admet_preds)
    else:
        raise ValueError(f"Unknown reward mode: {name}")
