"""Reward strategies: dispatch on ``config.reward.type`` to a concrete ``Reward``.

``make_reward`` builds the right strategy (BDE_IP / QED / pLogP); the
``MultiMolecules`` env delegates reward computation to ``Reward.compute``.
"""
from src.config import RewardType
from src.reward.rewards.base import Reward
from src.reward.rewards.bde_ip import BdeIpReward
from src.reward.rewards.qed import QedReward
from src.reward.rewards.plogp import PlogpReward


def make_reward(config, device, init_mols, bde_cache) -> Reward:
    """Construct the reward strategy selected by ``config.reward.type``."""
    rtype = config.reward.type
    if rtype is RewardType.BDE_IP:
        return BdeIpReward(config, device, init_mols, bde_cache)
    elif rtype is RewardType.QED:
        return QedReward(config)
    elif rtype is RewardType.PLOGP:
        return PlogpReward(config)
    raise ValueError(f"Unknown reward type: {rtype!r}")


__all__ = ["Reward", "make_reward"]
